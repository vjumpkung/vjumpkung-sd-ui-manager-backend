import asyncio
import hashlib
import os
import re
import shutil
import sys
import urllib.parse as urlparse
from typing import Literal
from urllib.parse import urlencode

import httpx
from curl_cffi.requests import AsyncSession
from pydantic import BaseModel, ConfigDict

from config.load_config import RESOURCE_PATH, UI_TYPE
from env_manager import envs
from event_handler import manager
from history_manager import downloadHistory
from log_manager import log
from utils.checksum import compute_sha256, fetch_civitai_sha256, fetch_hf_sha256
from utils.enums import DownloadStatus
from utils.ws_messages import DownloadData, DownloadMessage

PYTHON = sys.executable

semaphore = asyncio.Semaphore(5)
preflight_semaphore = asyncio.Semaphore(5)
active_download_tasks: set[asyncio.Task[bool]] = set()

CIVITAI_HOSTS = frozenset({"civitai.com", "civitai.red"})
HUGGINGFACE_HOSTS = frozenset({"huggingface.co"})

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


class DownloadPreparation(BaseModel):
    model_config = ConfigDict(frozen=True)

    cache_key: str
    expected_sha256: str | None
    filename: str | None
    destination: str
    file_matches_sha256: bool


class QueueDownloadResult(BaseModel):
    model_config = ConfigDict(frozen=True, arbitrary_types_allowed=True)

    action: Literal["queued", "retrying", "duplicate", "already_downloaded"]
    task: asyncio.Task[bool] | None = None


def _get_download_destination(model_type: str) -> tuple[str, str]:
    destination_type = model_type

    if destination_type == "checkpoints" and UI_TYPE != "FORGE":
        destination_type = "ckpts"
    elif UI_TYPE == "FORGE" and destination_type in forge_types_mapping:
        destination_type = forge_types_mapping[destination_type]

    return destination_type, os.path.join(RESOURCE_PATH, destination_type)


def _with_civitai_token(url: str) -> str:
    parsed_url = urlparse.urlparse(url)
    if parsed_url.hostname not in CIVITAI_HOSTS or not getattr(
        envs, "CIVITAI_TOKEN", ""
    ):
        return url

    url_parts = list(parsed_url)
    query = dict(urlparse.parse_qsl(url_parts[4]))
    query.setdefault("token", envs.CIVITAI_TOKEN)
    url_parts[4] = urlencode(query)
    return urlparse.urlunparse(url_parts)


async def _fetch_expected_sha256(url: str) -> str | None:
    parsed_url = urlparse.urlparse(url)
    hostname = parsed_url.hostname or ""

    if hostname in CIVITAI_HOSTS:
        path_parts = parsed_url.path.split("/")
        if "models" not in path_parts:
            return None

        model_version_index = path_parts.index("models") + 1
        if model_version_index >= len(path_parts):
            return None

        model_version_id = path_parts[model_version_index]
        if not model_version_id.isdigit():
            return None

        return await fetch_civitai_sha256(
            model_version_id, getattr(envs, "CIVITAI_TOKEN", None)
        )

    if hostname in HUGGINGFACE_HOSTS:
        path_parts = parsed_url.path.strip("/").split("/")
        if len(path_parts) >= 5 and path_parts[2] == "resolve":
            return await fetch_hf_sha256(
                path_parts[0],
                path_parts[1],
                "/".join(path_parts[4:]),
                getattr(envs, "HUGGINGFACE_TOKEN", None),
            )

    return None


async def _get_http_filename(url: str) -> str | None:
    try:
        async with AsyncSession() as session:
            response = await session.head(url, allow_redirects=True)
            try:
                if response.status_code >= 400:
                    return None
                return _extract_filename_from_cd(
                    response.headers.get("content-disposition", "")
                )
            finally:
                await response.aclose()
    except Exception:
        return None


async def _existing_file_matches_sha256(
    destination: str, filename: str | None, expected_sha256: str | None
) -> bool:
    if not filename or not expected_sha256:
        return False

    filepath = os.path.join(destination, filename)
    if not await asyncio.to_thread(os.path.isfile, filepath):
        return False

    try:
        local_sha256 = await compute_sha256(filepath)
    except OSError as exc:
        log.warning(f"Could not hash existing file {filename}: {exc}")
        return False

    if local_sha256 == expected_sha256:
        log.info(f"Checksum matches, skipping download: {filename}")
        return True

    log.warning(f"Checksum mismatch, removing stale file: {filename}")
    try:
        await asyncio.to_thread(os.remove, filepath)
    except FileNotFoundError:
        pass
    return False


async def prepare_download(
    name: str, url: str, model_type: str, from_model_pack: bool = False
) -> DownloadPreparation:
    """Resolve a source checksum and validate the target file before queueing."""
    envs.get_enviroment_variable()

    destination_type, destination = _get_download_destination(model_type)
    await asyncio.to_thread(os.makedirs, destination, exist_ok=True)

    expected_sha256 = await _fetch_expected_sha256(url)
    if expected_sha256:
        expected_sha256 = expected_sha256.lower()
    cache_key = expected_sha256 or hashlib.sha256(url.encode("utf-8")).hexdigest()

    parsed_url = urlparse.urlparse(url)
    hostname = parsed_url.hostname or ""
    filename = None

    if expected_sha256 and hostname in CIVITAI_HOSTS:
        filename = await _get_http_filename(_with_civitai_token(url))
    elif expected_sha256 and hostname in HUGGINGFACE_HOSTS:
        filename = _get_huggingface_filename(
            url, name, destination_type, cache_key, from_model_pack
        )

    file_matches_sha256 = await _existing_file_matches_sha256(
        destination, filename, expected_sha256
    )

    return DownloadPreparation(
        cache_key=cache_key,
        expected_sha256=expected_sha256,
        filename=filename,
        destination=destination,
        file_matches_sha256=file_matches_sha256,
    )


def _download_message(
    download_id: str,
    name: str,
    url: str,
    model_type: str,
    status: DownloadStatus,
    expected_sha256: str | None,
) -> DownloadMessage:
    return DownloadMessage(
        data=DownloadData(
            id=download_id,
            name=name,
            url=url,
            model_type=model_type,
            status=status,
            sha256=expected_sha256,
        )
    )


def _start_download(
    preparation: DownloadPreparation,
    name: str,
    url: str,
    model_type: str,
    from_model_pack: bool,
) -> asyncio.Task[bool]:
    task = asyncio.create_task(
        download_async(
            preparation.cache_key,
            name,
            url,
            model_type,
            from_model_pack,
            preparation.expected_sha256,
            preparation.filename,
        )
    )
    active_download_tasks.add(task)
    task.add_done_callback(active_download_tasks.discard)
    return task


def _get_huggingface_filename(
    url: str, name: str, model_type: str, download_id: str, from_model_pack: bool
) -> str:
    url_filename = os.path.basename(urlparse.urlparse(url).path)
    extension = os.path.splitext(url_filename)[1]

    if "diffusion_pytorch_model" in url_filename and name == model_type:
        root, _ = os.path.splitext(url_filename)
        return f"{root}-{download_id}{extension}"

    if name != model_type and not from_model_pack:
        if extension in name:
            root, _ = os.path.splitext(name)
            return f"{root}{extension}"
        return f"{name}{extension}"

    return url_filename


def _redact_command(command: list[str]) -> list[str]:
    return [
        "--header=Authorization: Bearer [REDACTED]"
        if arg.startswith("--header=Authorization: Bearer ")
        else arg
        for arg in command
    ]


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
        limit=1024 * 1024 * 100,
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


async def queue_download(
    name: str, url: str, model_type: str, from_model_pack: bool = False
) -> QueueDownloadResult:
    """Validate a download, update history, and start it only when needed."""
    preparation = await prepare_download(name, url, model_type, from_model_pack)
    existing = await downloadHistory.get_by_id(preparation.cache_key)

    if existing:
        status = existing["status"]

        if preparation.file_matches_sha256:
            if status == DownloadStatus.FAILED:
                updated = await downloadHistory.update_status_if_current(
                    preparation.cache_key,
                    DownloadStatus.FAILED,
                    DownloadStatus.COMPLETED,
                )
                if not updated:
                    return QueueDownloadResult(action="duplicate")
                completed = _download_message(
                    preparation.cache_key,
                    name,
                    url,
                    model_type,
                    DownloadStatus.COMPLETED,
                    preparation.expected_sha256,
                )
                await manager.broadcast(completed.model_dump_json())
            return QueueDownloadResult(action="already_downloaded")

        if status not in {DownloadStatus.FAILED, DownloadStatus.COMPLETED}:
            return QueueDownloadResult(action="duplicate")

        # A completed cache entry is only authoritative when its file still matches
        # the expected checksum. A missing or mismatched file must be downloaded again.
        if status == DownloadStatus.COMPLETED and not preparation.expected_sha256:
            return QueueDownloadResult(action="duplicate")

        queue_status = (
            DownloadStatus.RETRYING
            if status == DownloadStatus.FAILED
            else DownloadStatus.IN_QUEUE
        )
        updated = await downloadHistory.update_status_if_current(
            preparation.cache_key, status, queue_status
        )
        if not updated:
            return QueueDownloadResult(action="duplicate")
        message = _download_message(
            preparation.cache_key,
            name,
            url,
            model_type,
            queue_status,
            preparation.expected_sha256,
        )
        await manager.broadcast(message.model_dump_json())
        task = _start_download(preparation, name, url, model_type, from_model_pack)
        return QueueDownloadResult(
            action="retrying" if queue_status == DownloadStatus.RETRYING else "queued",
            task=task,
        )

    if preparation.file_matches_sha256:
        completed = _download_message(
            preparation.cache_key,
            name,
            url,
            model_type,
            DownloadStatus.COMPLETED,
            preparation.expected_sha256,
        )
        inserted = await downloadHistory.put(completed.data.model_dump())
        if not inserted:
            return QueueDownloadResult(action="duplicate")
        await manager.broadcast(completed.model_dump_json())
        return QueueDownloadResult(action="already_downloaded")

    in_queue = _download_message(
        preparation.cache_key,
        name,
        url,
        model_type,
        DownloadStatus.IN_QUEUE,
        preparation.expected_sha256,
    )
    inserted = await downloadHistory.put(in_queue.data.model_dump())
    if not inserted:
        return QueueDownloadResult(action="duplicate")
    await manager.broadcast(in_queue.model_dump_json())
    task = _start_download(preparation, name, url, model_type, from_model_pack)
    return QueueDownloadResult(action="queued", task=task)


async def download_async(
    id: str,
    name: str,
    url: str,
    t: str,
    from_model_pack: bool = False,
    expected_sha256: str | None = None,
    filename: str | None = None,
) -> bool:
    async with semaphore:
        type_name = t
        original_url = str(url)
        start = _download_message(
            id,
            name,
            original_url,
            type_name,
            DownloadStatus.DOWNLOADING,
            expected_sha256,
        )

        await downloadHistory.update_status(id, DownloadStatus.DOWNLOADING)

        await manager.broadcast(start.model_dump_json())

        envs.get_enviroment_variable()
        t, destination = _get_download_destination(t)
        os.makedirs(destination, exist_ok=True)

        log.info(f"model will download into {destination}")
        log.info(f"Starting download: {name}")

        parsed_url = urlparse.urlparse(url)
        hostname = parsed_url.hostname or ""

        if hostname in CIVITAI_HOSTS:
            url = _with_civitai_token(url)

        # CivitAI needs cURL. Hugging Face continues through aria2c below.
        if hostname in CIVITAI_HOSTS:
            try:
                await _download_http(
                    url,
                    destination,
                    filename=filename,
                    expected_sha256=expected_sha256,
                )
                await downloadHistory.update_status(id, DownloadStatus.COMPLETED)
                res = _download_message(
                    id,
                    name,
                    original_url,
                    type_name,
                    DownloadStatus.COMPLETED,
                    expected_sha256,
                )
                await manager.broadcast(res.model_dump_json())
                log.info(f"Download completed: {name}")
                return True
            except Exception as e:
                res = _download_message(
                    id,
                    name,
                    original_url,
                    type_name,
                    DownloadStatus.FAILED,
                    expected_sha256,
                )
                await downloadHistory.update_status(id, DownloadStatus.FAILED)
                await manager.broadcast(res.model_dump_json())
                log.error(f"Download failed: {name} ({e})")
                return False

        # Hugging Face and ordinary HTTP URLs use aria2c.
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

        if hostname in HUGGINGFACE_HOSTS:
            filename = filename or _get_huggingface_filename(
                url, name, t, id, from_model_pack
            )
            aria2_cmd.append(f"--out={filename}")

            if getattr(envs, "HUGGINGFACE_TOKEN", ""):
                aria2_cmd.append(
                    f"--header=Authorization: Bearer {envs.HUGGINGFACE_TOKEN}"
                )

        # if it's a Google Drive link, delegate to your google_drive_download script
        if hostname == "drive.google.com":
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

        log.info(f"executing command: {_redact_command(cmd)}")

        # create subprocess, redirecting stdout/stderr
        try:
            proc = await asyncio.create_subprocess_exec(
                *cmd,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.STDOUT,
                limit=1024 * 1024,  # 1MB limit to handle long progress lines
            )
        except Exception as e:
            res = _download_message(
                id,
                name,
                original_url,
                type_name,
                DownloadStatus.FAILED,
                expected_sha256,
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

        res = _download_message(
            id,
            name,
            original_url,
            type_name,
            DownloadStatus.COMPLETED,
            expected_sha256,
        )

        if return_code == 0:
            if expected_sha256 and filename:
                filepath = os.path.join(destination, filename)
                try:
                    actual_sha256 = await compute_sha256(filepath)
                except OSError as exc:
                    res.data.status = DownloadStatus.FAILED
                    await downloadHistory.update_status(id, DownloadStatus.FAILED)
                    await manager.broadcast(res.model_dump_json())
                    log.error(f"Download failed checksum verification: {name} ({exc})")
                    return False

                if actual_sha256 != expected_sha256:
                    try:
                        await asyncio.to_thread(os.remove, filepath)
                    except FileNotFoundError:
                        pass
                    res.data.status = DownloadStatus.FAILED
                    await downloadHistory.update_status(id, DownloadStatus.FAILED)
                    await manager.broadcast(res.model_dump_json())
                    log.error(
                        f"Download failed checksum verification: {name}. "
                        f"Expected {expected_sha256}, got {actual_sha256}"
                    )
                    return False

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

        models_to_queue = []
        for i in r.json():
            if (UI_TYPE == "INVOKEAI") and (
                i["type"] in ["text_encoders", "clip", "vae"]
            ):
                log.warning(
                    f"download {i['name']} skip because InvokeAI does not support"
                )
                continue

            models_to_queue.append(i)

        async def queue_model(i):
            async with preflight_semaphore:
                result = await queue_download(
                    i["name"], str(i["url"]), i["type"], from_model_pack=True
                )
            return i, result

        queue_results = await asyncio.gather(*(queue_model(i) for i in models_to_queue))

        for i, result in queue_results:
            if result.task:
                dl_lst.append(result.task)
            elif result.action == "duplicate":
                log.warning(
                    f"download {i['name']} was skipped because it exists in download history"
                )

    await asyncio.gather(*dl_lst)
