import os
import asyncio
from config.load_config import RESOURCE_PATH
from event_handler import manager
import json
from log_manager import log
from env_manager import envs
import httpx
import sys
from uuid import uuid4
from history_manager import downloadHistory

PYTHON = sys.executable

semaphore = asyncio.Semaphore(5)


async def download_async(id: str, name: str, url: str, t: str) -> bool:

    async with semaphore:

        start = {
            "type": "download",
            "data": {
                "id": id,
                "name": name,
                "url": str(url),
                "model_type": t,
                "status": "DOWNLOADING",
            },
        }

        downloadHistory.update_status(id, "DOWNLOADING")

        await manager.broadcast(json.dumps(start))

        if "envs" not in globals():
            envs.get_enviroment_variable()

        # adjust 'checkpoints' to 'ckpts' if needed
        if t == "checkpoints":
            t = "ckpts"

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
            "16",
            "-s",
            "16",
            "-k",
            "1M",
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
                "model_type": t,
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
    for j in packs:

        log.info(f"Start download {j['name']}")

        async with httpx.AsyncClient() as client:
            r = await client.get(str(j["url"]), follow_redirects=True)

        dl_lst = []
        for i in r.json():
            id = str(uuid4())
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
