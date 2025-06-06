import asyncio
import json
import os
import sys

import httpx

from config.load_config import RESOURCE_PATH, UI_TYPE
from env_manager import envs
from event_handler import manager
from history_manager import downloadHistory
from log_manager import log

PYTHON = sys.executable

semaphore = asyncio.Semaphore(5)

forge_types_mapping = {
    "checkpoints": "ckpts",
    "vae": "vae",
    "text-encoder": "text-encoder",
    "upscale_models": "esrgan",
    "unet": "ckpts",
    "clip": "text-encoder",
    "embeddings": "embeddings",
    "controlnet": "controlnet",
    "hypernetworks": "hypernetwork",
}


async def download_async(id: str, name: str, url: str, t: str) -> bool:

    async with semaphore:
        type_name = t
        start = {
            "type": "download",
            "data": {
                "id": id,
                "name": name,
                "url": str(url),
                "model_type": type_name,
                "status": "DOWNLOADING",
            },
        }

        downloadHistory.update_status(id, "DOWNLOADING")

        await manager.broadcast(json.dumps(start))

        if "envs" not in globals():
            envs.get_enviroment_variable()

        # adjust 'checkpoints' to 'ckpts' if needed

        if t == "checkpoints" and UI_TYPE != "FORGE":
            t = "ckpts"
        elif UI_TYPE == "FORGE":
            if t in forge_types_mapping.keys():
                t = forge_types_mapping[t]

        destination = os.path.join(RESOURCE_PATH, t)
        os.makedirs(destination, exist_ok=True)

        log.info(f"model will download into {destination}")
        log.info(f"Starting download: {name}")

        # attach tokens for civitai or huggingface if present
        if "civitai" in url and getattr(envs, "CIVITAI_TOKEN", ""):
            token_str = (
                f"&token={envs.CIVITAI_TOKEN}"
                if "?" in url
                else f"?token={envs.CIVITAI_TOKEN}"
            )
            url += token_str

        # build base aria2c command
        aria2_cmd = [
            "aria2c",
            "--console-log-level=error",
            "-c",
            "-x",
            "8",
            "-s",
            "8",
            "-k",
            "1M",
            "--retry-wait=5",  # Wait 5 seconds between retries
            "--max-tries=3",  # Reduce retry attempts
            url,
            f"--dir={destination}",
            "--download-result=hide",
        ]

        # if huggingface, add Authorization header and set output filename
        if "huggingface" in url and getattr(envs, "HUGGINGFACE_TOKEN", ""):
            aria2_cmd.append(f"--header=Authorization: Bearer {envs.HUGGINGFACE_TOKEN}")

        if "huggingface" in url:
            filename = url.split("/")[-1]
            aria2_cmd.extend(["-o", filename])

        # if civitai, let aria2c use content-disposition
        if "civitai" in url:
            aria2_cmd.append("--content-disposition=true")

        # if it's a Google Drive link, delegate to your google_drive_download script
        is_gdrive = "drive.google.com" in url
        if is_gdrive:
            # note: here we switch to calling a separate Python script
            gd_cmd = [
                PYTHON,
                "./scripts/google_drive_download.py",
                "--path",
                destination,
                "--url",
                url,
            ]
            cmd = gd_cmd
        else:
            cmd = aria2_cmd

        log.info(f"executing command: {cmd}")

        # create subprocess, redirecting stdout/stderr
        proc = await asyncio.create_subprocess_exec(
            *cmd,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.STDOUT,
        )

        prev_line = ""
        try:
            # read lines as they come in
            assert proc.stdout is not None
            async for raw_line in proc.stdout:
                line = raw_line.decode("utf-8").strip("\n")
                print(line, flush=True)

        except asyncio.CancelledError:
            # if someone cancels the task, kill the subprocess
            proc.kill()
            await proc.wait()
            log.warning(f"Download cancelled: {name}")
            return False

        return_code = await proc.wait()

        res = {
            "type": "download",
            "data": {
                "id": id,
                "name": name,
                "url": str(url),
                "model_type": type_name,
                "status": "COMPLETED",
            },
        }

        if return_code == 0:
            downloadHistory.update_status(id, "COMPLETED")
            await manager.broadcast(json.dumps(res))
            log.info(f"Download completed: {name}")
            return True
        else:
            res["data"]["status"] = "FAILED"
            downloadHistory.update_status(id, "FAILED")
            await manager.broadcast(json.dumps(res))
            log.error(f"Download failed: {name} (exit code {return_code})")
            return False


async def download_multiple(packs):
    dl_lst = []

    for j in packs:

        log.info(f"Start download {j['name']}")

        async with httpx.AsyncClient() as client:
            r = await client.get(str(j["url"]), follow_redirects=True)

        for i in r.json():

            if (UI_TYPE == "INVOKEAI") and (
                i["type"] in ["text_encoders", "clip", "vae"]
            ):
                log.warning(
                    f"download {i['name']} skip because InvokeAI does not support"
                )
                continue

            id = hex(hash(str(i["url"])))

            exists = downloadHistory.is_exists(id)

            if exists:
                log.warning(f"download {id} skip because it exists")
                continue

            dl_lst.append(download_async(id, i["name"], str(i["url"]), i["type"]))
            inqueue = {
                "type": "download",
                "data": {
                    "id": id,
                    "name": i["name"],
                    "url": str(i["url"]),
                    "model_type": i["type"],
                    "status": "IN_QUEUE",
                },
            }

            downloadHistory.put(inqueue["data"])

            await manager.broadcast(json.dumps(inqueue))

    await asyncio.gather(*dl_lst)
