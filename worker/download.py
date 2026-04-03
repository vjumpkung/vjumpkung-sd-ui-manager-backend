import asyncio
import os
import re
import shutil
import sys
import urllib.parse as urlparse
from urllib.parse import urlencode

import httpx
from curl_cffi.requests import AsyncSession

from config.load_config import RESOURCE_PATH, UI_TYPE
from env_manager import envs
from event_handler import manager
from history_manager import downloadHistory
from log_manager import log
from utils.checksum import compute_sha256, fetch_civitai_sha256, fetch_hf_sha256
from utils.enums import DownloadStatus
from utils.generate_uuid import generate_uuid
from utils.ws_messages import DownloadData, DownloadMessage

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
    expected_sha256: str | None = None,
) -> str:
    # HEAD with default agent to check auth/response before downloading
    content_length = 0
    try:
        async with AsyncSession() as session:
            head = await session.head(
                url,
                headers=headers or {},
                allow_redirects=True,
            )
            try:
                if head.status_code == 401:
                    raise RuntimeError(
                        f"Authentication required (HTTP 401): API key/token is missing or invalid"
                    )
                if head.status_code == 403:
                    raise RuntimeError(
                        f"Access denied (HTTP 403): you may need an API key or lack permission to download this file"
                    )
                if head.status_code < 400:
                    content_type = head.headers.get("content-type", "")
                    if "text/html" in content_type:
                        raise RuntimeError(
                            "Server returned an HTML page instead of a file — "
                            "the URL may require an API key or the model may be behind a login/paywall"
                        )
                    content_length = int(head.headers.get("content-length", 0))
                    if not filename:
                        filename = _extract_filename_from_cd(
                            head.headers.get("content-disposition", "")
                        )
            finally:
                await head.aclose()
    except RuntimeError:
        raise
    except Exception:
        pass

    if not filename:
        filename = url.split("?")[0].split("/")[-1]

    if content_length > 0:
        free = shutil.disk_usage(destination).free
        if free < content_length:
            raise RuntimeError(
                f"Not enough disk space: need {content_length / 1024**3:.2f} GB, "
                f"free {free / 1024**3:.2f} GB"
            )

    filepath = os.path.join(destination, filename)

    if os.path.exists(filepath):
        if expected_sha256:
            print(f"Verifying checksum for existing file: {filename}", flush=True)
            local_sha256 = await compute_sha256(filepath)
            if local_sha256 == expected_sha256.lower():
                print(f"Checksum matches, skipping: {filename}", flush=True)
                return filename
            print(f"Checksum mismatch, re-downloading: {filename}", flush=True)
        else:
            existing_size = os.path.getsize(filepath)
            if content_length == 0 or existing_size == content_length:
                print(f"File already exists, skipping: {filepath}", flush=True)
                return filename
            print(
                f"File exists but size mismatch (local {existing_size}, remote {content_length}), re-downloading",
                flush=True,
            )

    cmd = [
        "curl",
        "-L",
        "--retry",
        "3",
        "--retry-delay",
        "5",
        "-o",
        filepath,
        "-w",
        "\n%{http_code}",
    ]

    for key, value in (headers or {}).items():
        if key.lower() != "user-agent":
            cmd.extend(["-H", f"{key}: {value}"])

    cmd.append(url)

    proc = await asyncio.create_subprocess_exec(
        *cmd,
        stdout=asyncio.subprocess.PIPE,
        stderr=asyncio.subprocess.STDOUT,
    )

    assert proc.stdout is not None
    last_line = ""
    async for raw_line in proc.stdout:
        line = raw_line.decode("utf-8").strip()
        if line:
            print(line, flush=True)
            last_line = line

    return_code = await proc.wait()
    if return_code != 0:
        raise RuntimeError(f"curl exited with code {return_code}")

    http_code = last_line.strip()
    if http_code.isdigit():
        code = int(http_code)
        if code == 401:
            raise RuntimeError(
                "Authentication required (HTTP 401): API key/token is missing or invalid"
            )
        if code == 403:
            raise RuntimeError(
                "Access denied (HTTP 403): you may need an API key or lack permission to download this file"
            )
        if code >= 400:
            raise RuntimeError(f"Download failed: server returned HTTP {code}")

    if expected_sha256:
        print(f"Verifying checksum after download: {filename}", flush=True)
        actual_sha256 = await compute_sha256(filepath)
        if actual_sha256 != expected_sha256.lower():
            os.remove(filepath)
            raise RuntimeError(
                f"Checksum mismatch after download — file deleted. "
                f"Expected {expected_sha256.lower()}, got {actual_sha256}"
            )
        print(f"Checksum verified: {filename}", flush=True)

    return filename


async def download_async(
    id: str, name: str, url: str, t: str, from_model_pack=False
) -> bool:
    async with semaphore:
        type_name = t
        original_url = str(url)
        start = DownloadMessage(
            data=DownloadData(
                id=id,
                name=name,
                url=original_url,
                model_type=type_name,
                status=DownloadStatus.DOWNLOADING,
            )
        )

        await downloadHistory.update_status(id, DownloadStatus.DOWNLOADING)

        await manager.broadcast(start.model_dump_json())

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
            expected_sha256 = None

            if parsed_url[1] == "civitai.com":
                path_parts = parsed_url[2].split("/")
                if "models" in path_parts:
                    idx = path_parts.index("models")
                    if idx + 1 < len(path_parts) and path_parts[idx + 1].isdigit():
                        expected_sha256 = await fetch_civitai_sha256(
                            path_parts[idx + 1],
                            getattr(envs, "CIVITAI_TOKEN", None),
                        )
                        if expected_sha256:
                            log.info(f"Fetched CivitAI SHA256: {expected_sha256}")
                        else:
                            log.warning(
                                "Could not fetch SHA256 from CivitAI API, falling back to size check"
                            )

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

                # fetch SHA256 from HF Hub API
                # URL: /resolve/{revision}/{...filepath} → path_parts[3] = 'resolve'
                hf_path_parts = parsed_url[2].strip("/").split("/")
                if len(hf_path_parts) >= 5 and hf_path_parts[2] == "resolve":
                    hf_owner = hf_path_parts[0]
                    hf_repo = hf_path_parts[1]
                    hf_filepath = "/".join(hf_path_parts[4:])
                    expected_sha256 = await fetch_hf_sha256(
                        hf_owner,
                        hf_repo,
                        hf_filepath,
                        getattr(envs, "HUGGINGFACE_TOKEN", None),
                    )
                    if expected_sha256:
                        log.info(f"Fetched HuggingFace SHA256: {expected_sha256}")
                    else:
                        log.warning(
                            "Could not fetch SHA256 from HuggingFace API, falling back to size check"
                        )

            try:
                await _download_http(
                    url,
                    destination,
                    filename=hf_filename,
                    headers=hf_headers,
                    expected_sha256=expected_sha256,
                )
                await downloadHistory.update_status(id, DownloadStatus.COMPLETED)
                res = DownloadMessage(
                    data=DownloadData(
                        id=id,
                        name=name,
                        url=original_url,
                        model_type=type_name,
                        status=DownloadStatus.COMPLETED,
                    )
                )
                await manager.broadcast(res.model_dump_json())
                log.info(f"Download completed: {name}")
                return True
            except Exception as e:
                res = DownloadMessage(
                    data=DownloadData(
                        id=id,
                        name=name,
                        url=original_url,
                        model_type=type_name,
                        status=DownloadStatus.FAILED,
                    )
                )
                await downloadHistory.update_status(id, DownloadStatus.FAILED)
                await manager.broadcast(res.model_dump_json())
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
                limit=1024 * 1024,  # 1MB limit to handle long progress lines
            )
        except Exception as e:
            res = DownloadMessage(
                data=DownloadData(
                    id=id,
                    name=name,
                    url=original_url,
                    model_type=type_name,
                    status=DownloadStatus.FAILED,
                )
            )
            await downloadHistory.update_status(id, DownloadStatus.FAILED)
            await manager.broadcast(res.model_dump_json())
            log.error(f"Download failed: {name} (exit code {e})")
            return False

        try:
            # read lines as they come in
            assert proc.stdout is not None
            while True:
                try:
                    raw_line = await proc.stdout.readline()
                except ValueError:
                    # line exceeded StreamReader limit; read and discard the rest of the chunk
                    await proc.stdout.read(1024 * 1024)
                    continue
                if not raw_line:
                    break
                line = raw_line.decode("utf-8", errors="replace").strip("\n")
                print(line, flush=True)

        except asyncio.CancelledError:
            # if someone cancels the task, kill the subprocess
            proc.kill()
            await proc.wait()
            log.warning(f"Download cancelled: {name}")
            return False

        return_code = await proc.wait()

        res = DownloadMessage(
            data=DownloadData(
                id=id,
                name=name,
                url=original_url,
                model_type=type_name,
                status=DownloadStatus.COMPLETED,
            )
        )

        if return_code == 0:
            await downloadHistory.update_status(id, DownloadStatus.COMPLETED)
            await manager.broadcast(res.model_dump_json())
            log.info(f"Download completed: {name}")
            return True
        else:
            res.data.status = DownloadStatus.FAILED
            await downloadHistory.update_status(id, DownloadStatus.FAILED)
            await manager.broadcast(res.model_dump_json())
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

                if get_data["status"] == DownloadStatus.FAILED:
                    dl_lst.append(
                        download_async(id, i["name"], str(i["url"]), i["type"], True)
                    )

                    retry_msg = DownloadMessage(
                        data=DownloadData(
                            id=id,
                            name=i["name"],
                            url=str(i["url"]),
                            model_type=i["type"],
                            status=DownloadStatus.RETRYING,
                        )
                    )

                    await manager.broadcast(retry_msg.model_dump_json())
                    await downloadHistory.update_status(id, DownloadStatus.RETRYING)
                    log.info(
                        f"retry downloading {i['name']} from {str(i['url'])} again..."
                    )
                    continue

                log.warning(
                    f"download {i['name']} was skipped because it exists in download history"
                )
                continue

            dl_lst.append(download_async(id, i["name"], str(i["url"]), i["type"], True))
            inqueue = DownloadMessage(
                data=DownloadData(
                    id=id,
                    name=i["name"],
                    url=str(i["url"]),
                    model_type=i["type"],
                    status=DownloadStatus.IN_QUEUE,
                )
            )

            await downloadHistory.put(inqueue.data.model_dump())

            await manager.broadcast(inqueue.model_dump_json())

    await asyncio.gather(*dl_lst)
