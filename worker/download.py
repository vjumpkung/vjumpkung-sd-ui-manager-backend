import asyncio
import json
import os
import re
import sys
import urllib.parse as urlparse
from urllib.parse import urlencode

import aiofiles
import httpx
from curl_cffi.requests import AsyncSession
from tqdm import tqdm

from config.load_config import RESOURCE_PATH, UI_TYPE
from env_manager import envs
from event_handler import manager
from history_manager import downloadHistory
from log_manager import log
from utils.generate_uuid import generate_uuid

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


def check_file_extensions(f: str):
    root, extension = os.path.splitext(f)
    return extension


_PARALLEL_CONNECTIONS = 8
_CHUNK_SIZE = 1024 * 1024  # 1 MB


def _extract_filename_from_cd(cd: str) -> str | None:
    match = re.findall(
        r"filename\*?=(?:UTF-8''|\"?)([^\";\r\n]+)\"?", cd, re.IGNORECASE
    )
    return match[0].strip().strip('"') if match else None


async def _download_http(
    url: str,
    destination: str,
    filename: str | None = None,
    headers: dict | None = None,
) -> str:
    base_headers = headers or {}
    content_length = 0
    accepts_ranges = False

    async with AsyncSession() as session:
        # HEAD to probe range support and resolve filename from Content-Disposition
        try:
            head = await session.head(
                url, impersonate="chrome", headers=base_headers, allow_redirects=True
            )
            try:
                if head.status_code < 400:
                    content_length = int(head.headers.get("content-length", 0))
                    accepts_ranges = (
                        head.headers.get("accept-ranges", "none").lower() == "bytes"
                    )
                    if not filename:
                        filename = _extract_filename_from_cd(
                            head.headers.get("content-disposition", "")
                        )
            finally:
                await head.aclose()
        except Exception:
            pass

        if not filename:
            filename = url.split("?")[0].split("/")[-1]

        filepath = os.path.join(destination, filename)

        desc = filename[:40] + "…" if len(filename) > 40 else filename
        pbar = tqdm(
            total=content_length if content_length > 0 else None,
            unit="B",
            unit_scale=True,
            unit_divisor=1024,
            desc=desc,
            dynamic_ncols=True,
        )

        try:
            if accepts_ranges and content_length > 0:
                # parallel chunked download
                part = content_length // _PARALLEL_CONNECTIONS
                ranges = [
                    (
                        i * part,
                        (i + 1) * part - 1
                        if i < _PARALLEL_CONNECTIONS - 1
                        else content_length - 1,
                    )
                    for i in range(_PARALLEL_CONNECTIONS)
                ]

                # pre-allocate file so concurrent seeks are safe
                async with aiofiles.open(filepath, "wb") as f:
                    await f.seek(content_length - 1)
                    await f.write(b"\x00")

                async def fetch_chunk(start: int, end: int) -> None:
                    chunk_headers = {**base_headers, "Range": f"bytes={start}-{end}"}
                    resp = await session.get(
                        url, stream=True, impersonate="chrome", headers=chunk_headers
                    )
                    try:
                        resp.raise_for_status()
                        async with aiofiles.open(filepath, "r+b") as f:
                            await f.seek(start)
                            async for data in resp.aiter_content(
                                chunk_size=_CHUNK_SIZE
                            ):
                                await f.write(data)
                                pbar.update(len(data))
                    finally:
                        await resp.aclose()

                await asyncio.gather(*[fetch_chunk(s, e) for s, e in ranges])

            else:
                # fallback: single-stream GET
                resp = await session.get(
                    url, stream=True, impersonate="chrome", headers=base_headers
                )
                try:
                    resp.raise_for_status()

                    # try Content-Disposition from response if HEAD didn't give us a filename
                    if not filename or filename == url.split("?")[0].split("/")[-1]:
                        cd_name = _extract_filename_from_cd(
                            resp.headers.get("content-disposition", "")
                        )
                        if cd_name:
                            filename = cd_name
                            filepath = os.path.join(destination, filename)
                            pbar.set_description(
                                filename[:40] + "…" if len(filename) > 40 else filename
                            )

                    async with aiofiles.open(filepath, "wb") as f:
                        async for data in resp.aiter_content(chunk_size=_CHUNK_SIZE):
                            await f.write(data)
                            pbar.update(len(data))
                finally:
                    await resp.aclose()
        finally:
            pbar.close()

    return filename


async def download_async(
    id: str, name: str, url: str, t: str, from_model_pack=False
) -> bool:
    async with semaphore:
        type_name = t
        original_url = str(url)
        start = {
            "type": "download",
            "data": {
                "id": id,
                "name": name,
                "url": original_url,
                "model_type": type_name,
                "status": "DOWNLOADING",
            },
        }

        await downloadHistory.update_status(id, "DOWNLOADING")

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

        parsed_url = list(urlparse.urlparse(url))

        # attach tokens for civitai or huggingface if present
        if parsed_url[1] == "civitai.com" and getattr(envs, "CIVITAI_TOKEN", ""):
            url_parts = list(urlparse.urlparse(url))
            query = dict(urlparse.parse_qsl(url_parts[4]))
            query.setdefault("token", envs.CIVITAI_TOKEN)

            url_parts[4] = urlencode(query)

            url = urlparse.urlunparse(url_parts)

        # CivitAI / HuggingFace: use curl_cffi
        if parsed_url[1] in ("civitai.com", "huggingface.co"):
            hf_filename = None
            hf_headers = {}

            if parsed_url[1] == "huggingface.co":
                if getattr(envs, "HUGGINGFACE_TOKEN", ""):
                    hf_headers["Authorization"] = f"Bearer {envs.HUGGINGFACE_TOKEN}"

                url_filename = url.split("/")[-1]
                get_extension = check_file_extensions(url_filename)
                hf_filename = url_filename

                if ("diffusion_pytorch_model" in url_filename) and (name == t):
                    root, extension = os.path.splitext(url_filename)
                    hf_filename = f"{root}-{id}{extension}"
                elif (name != t) and (not from_model_pack):
                    if get_extension in name:
                        root, extension = os.path.splitext(name)
                        hf_filename = f"{root}{extension}"
                    else:
                        hf_filename = f"{name}{get_extension}"

            try:
                await _download_http(
                    url, destination, filename=hf_filename, headers=hf_headers
                )
                await downloadHistory.update_status(id, "COMPLETED")
                res = {
                    "type": "download",
                    "data": {
                        "id": id,
                        "name": name,
                        "url": original_url,
                        "model_type": type_name,
                        "status": "COMPLETED",
                    },
                }
                await manager.broadcast(json.dumps(res))
                log.info(f"Download completed: {name}")
                return True
            except Exception as e:
                res = {
                    "type": "download",
                    "data": {
                        "id": id,
                        "name": name,
                        "url": original_url,
                        "model_type": type_name,
                        "status": "FAILED",
                    },
                }
                await downloadHistory.update_status(id, "FAILED")
                await manager.broadcast(json.dumps(res))
                log.error(f"Download failed: {name} ({e})")
                return False

        # build base aria2c command (Google Drive falls through to subprocess below)
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
            "--retry-wait=5",
            "--max-tries=3",
            url,
            f"--dir={destination}",
            "--download-result=hide",
        ]

        # if it's a Google Drive link, delegate to your google_drive_download script
        if parsed_url[1] == "drive.google.com":
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
        try:
            proc = await asyncio.create_subprocess_exec(
                *cmd,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.STDOUT,
            )
        except Exception as e:
            res = {
                "type": "download",
                "data": {
                    "id": id,
                    "name": name,
                    "url": original_url,
                    "model_type": type_name,
                    "status": "FAILED",
                },
            }
            await downloadHistory.update_status(id, "FAILED")
            await manager.broadcast(json.dumps(res))
            log.error(f"Download failed: {name} (exit code {e})")
            return False

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
                "url": original_url,
                "model_type": type_name,
                "status": "COMPLETED",
            },
        }

        if return_code == 0:
            await downloadHistory.update_status(id, "COMPLETED")
            await manager.broadcast(json.dumps(res))
            log.info(f"Download completed: {name}")
            return True
        else:
            res["data"]["status"] = "FAILED"
            await downloadHistory.update_status(id, "FAILED")
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

            id = generate_uuid(str(i["url"]))

            exists = await downloadHistory.is_exists(id)

            if exists:
                get_data = await downloadHistory.get_by_id(id)

                if get_data["status"] == "FAILED":
                    dl_lst.append(
                        download_async(id, i["name"], str(i["url"]), i["type"], True)
                    )

                    retry_msg = {
                        "type": "download",
                        "data": {
                            "id": id,
                            "name": i["name"],
                            "url": str(i["url"]),
                            "model_type": i["type"],
                            "status": "RETRYING",
                        },
                    }

                    await manager.broadcast(json.dumps(retry_msg))
                    await downloadHistory.update_status(id, "RETRYING...")
                    log.info(
                        f"retry downloading {i['name']} from {str(i['url'])} again..."
                    )
                    continue

                log.warning(
                    f"download {i['name']} was skipped because it exists in download history"
                )
                continue

            dl_lst.append(download_async(id, i["name"], str(i["url"]), i["type"], True))
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

            await downloadHistory.put(inqueue["data"])

            await manager.broadcast(json.dumps(inqueue))

    await asyncio.gather(*dl_lst)
