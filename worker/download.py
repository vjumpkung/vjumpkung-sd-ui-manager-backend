import asyncio
import hashlib
import os
import shutil
import subprocess
import sys
import traceback
import urllib.parse as urlparse
from typing import Literal

import httpx
from pydantic import BaseModel, ConfigDict

from config.load_config import RESOURCE_PATH, UI_TYPE
from env_manager import envs
from event_handler import manager
from history_manager import downloadHistory
from log_manager import log
from utils.checksum import compute_sha256, fetch_hf_sha256
from utils.enums import DownloadStatus
from utils.ws_messages import DownloadData, DownloadMessage

PYTHON = sys.executable
CIVITAI_DOWNLOAD_SCRIPT = os.path.abspath(
    os.path.join(os.path.dirname(__file__), "..", "scripts", "civitai_download.py")
)

semaphore = asyncio.Semaphore(5)
preflight_semaphore = asyncio.Semaphore(5)
active_download_tasks: set[asyncio.Task[bool]] = set()

CIVITAI_HOSTS = frozenset({"civitai.com", "civitai.green", "civitai.red"})
CIVITAI_REDIRECT_STATUSES = frozenset({301, 302, 303, 307, 308})
CIVITAI_DISK_RESERVE_BYTES = 64 * 1024 * 1024
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
    expected_size_bytes: int | None
    civitai_file_id: str | None


class QueueDownloadResult(BaseModel):
    model_config = ConfigDict(frozen=True, arbitrary_types_allowed=True)

    action: Literal["queued", "retrying", "duplicate", "already_downloaded"]
    task: asyncio.Task[bool] | None = None


class HuggingFaceDownloadTarget(BaseModel):
    model_config = ConfigDict(frozen=True)

    repo_id: str
    repo_type: Literal["model", "dataset", "space"]
    revision: str
    filepath: str


class CivitaiDownloadTarget(BaseModel):
    model_config = ConfigDict(frozen=True)

    version_id: str
    file_id: str
    filename: str
    expected_sha256: str
    expected_size_bytes: int


def _get_download_destination(model_type: str) -> tuple[str, str]:
    destination_type = model_type

    if destination_type == "checkpoints" and UI_TYPE != "FORGE":
        destination_type = "ckpts"
    elif UI_TYPE == "FORGE" and destination_type in forge_types_mapping:
        destination_type = forge_types_mapping[destination_type]

    return destination_type, os.path.join(RESOURCE_PATH, destination_type)


def _get_civitai_headers() -> dict[str, str]:
    token = getattr(envs, "CIVITAI_TOKEN", "")
    return {"Authorization": f"Bearer {token}"} if token else {}


def _get_query_value(query: dict[str, list[str]], name: str) -> str | None:
    for key, values in query.items():
        if key.casefold() == name.casefold() and values:
            return values[0]
    return None


def _get_civitai_version_id(url: str) -> str:
    path_parts = urlparse.urlparse(url).path.strip("/").split("/")
    try:
        model_index = path_parts.index("models")
        version_id = path_parts[model_index + 1]
    except (ValueError, IndexError) as exc:
        raise RuntimeError("Invalid CivitAI model download URL") from exc

    if not version_id.isdigit() or int(version_id) <= 0:
        raise RuntimeError("Invalid CivitAI model version ID")
    return version_id


def _validated_download_filename(value: object) -> str:
    filename = os.path.basename(str(value or "").replace("\\", "/")).strip()
    if (
        filename in {"", ".", ".."}
        or filename.endswith(".")
        or any(ord(character) < 32 for character in filename)
        or any(character in filename for character in '<>:"/\\|?*')
        or len(filename.encode("utf-8")) > 240
    ):
        raise RuntimeError("CivitAI returned an unsafe filename")
    return filename


def _select_civitai_file(
    version_id: str, url: str, files: object
) -> CivitaiDownloadTarget:
    if not isinstance(files, list) or not files:
        raise RuntimeError(f"CivitAI version {version_id} has no downloadable files")

    query = urlparse.parse_qs(urlparse.urlparse(url).query)
    requested_file_id = _get_query_value(query, "fileId")
    filters = {
        "type": _get_query_value(query, "type"),
        "format": _get_query_value(query, "format"),
        "size": _get_query_value(query, "size"),
        "fp": _get_query_value(query, "fp"),
    }

    candidates: list[dict] = []
    for file_data in files:
        if not isinstance(file_data, dict):
            continue
        file_id = str(file_data.get("id") or "")
        if requested_file_id and file_id != requested_file_id:
            continue

        metadata = file_data.get("metadata") or {}
        actual_values = {
            "type": file_data.get("type"),
            "format": metadata.get("format"),
            "size": metadata.get("size"),
            "fp": metadata.get("fp"),
        }
        if requested_file_id or all(
            requested is None
            or str(actual_values[key] or "").casefold() == requested.casefold()
            for key, requested in filters.items()
        ):
            candidates.append(file_data)

    if not candidates:
        requested = f"file {requested_file_id}" if requested_file_id else "filters"
        raise RuntimeError(
            f"CivitAI version {version_id} has no file matching the requested {requested}"
        )

    selected = next(
        (file_data for file_data in candidates if file_data.get("primary")),
        candidates[0],
    )
    file_id = str(selected.get("id") or "")
    sha256 = str((selected.get("hashes") or {}).get("SHA256") or "").lower()
    if not file_id.isdigit() or int(file_id) <= 0:
        raise RuntimeError("CivitAI returned an invalid file ID")
    if len(sha256) != 64 or any(
        character not in "0123456789abcdef" for character in sha256
    ):
        raise RuntimeError(f"CivitAI file {file_id} has no valid SHA-256")

    try:
        expected_size_bytes = round(float(selected.get("sizeKB")) * 1024)
    except (TypeError, ValueError) as exc:
        raise RuntimeError(f"CivitAI file {file_id} has no valid size") from exc
    if expected_size_bytes <= 0:
        raise RuntimeError(f"CivitAI file {file_id} has no valid size")

    return CivitaiDownloadTarget(
        version_id=version_id,
        file_id=file_id,
        filename=_validated_download_filename(selected.get("name")),
        expected_sha256=sha256,
        expected_size_bytes=expected_size_bytes,
    )


async def _fetch_civitai_target(url: str) -> CivitaiDownloadTarget:
    token = getattr(envs, "CIVITAI_TOKEN", "")
    if not token:
        raise RuntimeError(
            "CIVITAI_TOKEN is required; configure it before downloading CivitAI models"
        )

    version_id = _get_civitai_version_id(url)
    metadata_url = f"https://civitai.com/api/v1/model-versions/{version_id}"
    try:
        async with httpx.AsyncClient() as client:
            response = await client.get(
                metadata_url,
                headers=_get_civitai_headers(),
                follow_redirects=True,
                timeout=15,
            )
    except httpx.HTTPError as exc:
        raise RuntimeError("Could not fetch CivitAI model metadata") from exc

    if response.status_code in {401, 403}:
        raise RuntimeError("CivitAI API token is invalid or lacks access to this model")
    if response.status_code != 200:
        raise RuntimeError(
            f"CivitAI metadata request failed with HTTP {response.status_code}"
        )
    try:
        payload = response.json()
    except ValueError as exc:
        raise RuntimeError("CivitAI returned invalid model metadata") from exc
    if not isinstance(payload, dict):
        raise TypeError("CivitAI returned invalid model metadata")
    return _select_civitai_file(version_id, url, payload.get("files"))


async def _fetch_expected_sha256(url: str) -> str | None:
    parsed_url = urlparse.urlparse(url)
    hostname = parsed_url.hostname or ""

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
    envs.get_environment_variable()

    destination_type, destination = _get_download_destination(model_type)
    await asyncio.to_thread(os.makedirs, destination, exist_ok=True)

    parsed_url = urlparse.urlparse(url)
    hostname = parsed_url.hostname or ""
    civitai_target = None
    if hostname in CIVITAI_HOSTS:
        civitai_target = await _fetch_civitai_target(url)
        expected_sha256 = civitai_target.expected_sha256
    else:
        expected_sha256 = await _fetch_expected_sha256(url)
    if expected_sha256:
        expected_sha256 = expected_sha256.lower()
    cache_key = expected_sha256 or hashlib.sha256(url.encode("utf-8")).hexdigest()

    filename = civitai_target.filename if civitai_target else None
    if expected_sha256 and hostname in HUGGINGFACE_HOSTS:
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
        expected_size_bytes=(
            civitai_target.expected_size_bytes if civitai_target else None
        ),
        civitai_file_id=civitai_target.file_id if civitai_target else None,
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
            preparation.expected_size_bytes,
            preparation.civitai_file_id,
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


def _parse_huggingface_download_url(
    url: str,
) -> HuggingFaceDownloadTarget | None:
    """Parse a Hub file URL into arguments accepted by ``hf download``."""
    parsed_url = urlparse.urlparse(url)
    if parsed_url.hostname not in HUGGINGFACE_HOSTS:
        return None

    path_parts = parsed_url.path.strip("/").split("/")
    repo_type: Literal["model", "dataset", "space"] = "model"
    repo_start = 0

    if path_parts and path_parts[0] in {"datasets", "spaces"}:
        repo_type = "dataset" if path_parts[0] == "datasets" else "space"
        repo_start = 1

    marker_index = repo_start + 2
    if len(path_parts) <= marker_index + 2 or path_parts[marker_index] not in {
        "resolve",
        "blob",
    }:
        return None

    owner, repo = [
        urlparse.unquote(part) for part in path_parts[repo_start : repo_start + 2]
    ]
    revision = urlparse.unquote(path_parts[marker_index + 1])
    filepath = "/".join(
        urlparse.unquote(part) for part in path_parts[marker_index + 2 :]
    )
    filepath_parts = filepath.split("/")

    if (
        not owner
        or not repo
        or not revision
        or not filepath_parts
        or "/" in owner
        or "\\" in owner
        or "/" in repo
        or "\\" in repo
        or "\\" in filepath
        or any(part in {"", ".", ".."} for part in filepath_parts)
    ):
        return None

    return HuggingFaceDownloadTarget(
        repo_id=f"{owner}/{repo}",
        repo_type=repo_type,
        revision=revision,
        filepath=filepath,
    )


def _build_huggingface_cli_command(
    hf_executable: str,
    target: HuggingFaceDownloadTarget,
    local_dir: str,
) -> list[str]:
    command = [
        hf_executable,
        "download",
        target.repo_id,
        target.filepath,
        "--revision",
        target.revision,
        "--local-dir",
        local_dir,
    ]
    if target.repo_type != "model":
        command.extend(["--repo-type", target.repo_type])
    return command


def _get_huggingface_staging_paths(
    destination: str,
    download_id: str,
    filepath: str,
) -> tuple[str, str]:
    staging_key = hashlib.sha256(download_id.encode("utf-8")).hexdigest()
    staging_dir = os.path.join(destination, ".hf-download", staging_key)
    source_path = os.path.join(staging_dir, *filepath.split("/"))
    return staging_dir, source_path


def _finalize_huggingface_download(
    source_path: str,
    destination: str,
    filename: str,
    staging_dir: str,
) -> None:
    if not os.path.isfile(source_path):
        raise FileNotFoundError(
            f"hf CLI completed but did not create the expected file: {source_path}"
        )

    os.replace(source_path, os.path.join(destination, filename))
    shutil.rmtree(staging_dir)

    staging_parent = os.path.dirname(staging_dir)
    try:
        os.rmdir(staging_parent)
    except OSError:
        pass


def _redact_command(command: list[str]) -> list[str]:
    return [
        "--header=Authorization: Bearer [REDACTED]"
        if arg.startswith("--header=Authorization: Bearer ")
        else arg
        for arg in command
    ]


def _build_civitai_download_url(url: str, file_id: str, token: str) -> str:
    parsed = urlparse.urlparse(url)
    query = [
        (key, value)
        for key, value in urlparse.parse_qsl(parsed.query, keep_blank_values=True)
        if key.casefold() not in {"fileid", "token"}
    ]
    query.extend((("fileId", file_id), ("token", token)))
    return urlparse.urlunparse(parsed._replace(query=urlparse.urlencode(query)))


async def _resolve_civitai_download_url(url: str, file_id: str) -> str:
    token = getattr(envs, "CIVITAI_TOKEN", "")
    if not token:
        raise RuntimeError(
            "CIVITAI_TOKEN is required; configure it before downloading CivitAI models"
        )

    download_url = _build_civitai_download_url(url, file_id, token)
    try:
        async with (
            httpx.AsyncClient() as client,
            client.stream(
                "GET",
                download_url,
                follow_redirects=False,
                timeout=30,
            ) as response,
        ):
            if response.status_code in {401, 403}:
                raise RuntimeError(
                    "CivitAI API token is invalid or lacks access to this model"
                )
            if response.status_code not in CIVITAI_REDIRECT_STATUSES:
                raise RuntimeError(
                    "CivitAI download endpoint returned "
                    f"HTTP {response.status_code}; expected a signed redirect"
                )
            location = response.headers.get("location")
    except httpx.HTTPError as exc:
        raise RuntimeError("Could not resolve the CivitAI download URL") from exc

    if not location:
        raise RuntimeError("CivitAI download redirect did not include a location")
    signed_url = urlparse.urljoin(download_url, location)
    parsed_signed_url = urlparse.urlparse(signed_url)
    if (
        parsed_signed_url.scheme != "https"
        or not parsed_signed_url.hostname
        or any(character in signed_url for character in ("\r", "\n", "\x00"))
    ):
        raise RuntimeError("CivitAI returned an invalid download redirect")
    if any(
        value == token
        for values in urlparse.parse_qs(parsed_signed_url.query).values()
        for value in values
    ):
        raise RuntimeError("CivitAI token was exposed in the download redirect")
    return signed_url


def _build_civitai_helper_command(
    destination: str, staging_filename: str, expected_sha256: str
) -> list[str]:
    return [
        PYTHON,
        CIVITAI_DOWNLOAD_SCRIPT,
        "--destination",
        destination,
        "--output",
        staging_filename,
        "--sha256",
        expected_sha256,
    ]


async def _civitai_file_matches(
    path: str, expected_size_bytes: int, expected_sha256: str
) -> bool:
    if not await asyncio.to_thread(os.path.isfile, path):
        return False
    if await asyncio.to_thread(os.path.getsize, path) != expected_size_bytes:
        return False
    return await compute_sha256(path) == expected_sha256


async def _remove_file_if_present(path: str) -> None:
    try:
        await asyncio.to_thread(os.remove, path)
    except FileNotFoundError:
        pass


async def _download_civitai(
    url: str,
    destination: str,
    filename: str | None = None,
    expected_sha256: str | None = None,
    expected_size_bytes: int | None = None,
    file_id: str | None = None,
) -> str:
    if not all((filename, expected_sha256, expected_size_bytes, file_id)):
        target = await _fetch_civitai_target(url)
        filename = target.filename
        expected_sha256 = target.expected_sha256
        expected_size_bytes = target.expected_size_bytes
        file_id = target.file_id

    assert filename is not None
    assert expected_sha256 is not None
    assert expected_size_bytes is not None
    assert file_id is not None
    filename = _validated_download_filename(filename)

    staging_filename = f".civitai-{file_id}-{expected_sha256[:12]}.part"
    staging_path = os.path.join(destination, staging_filename)
    control_path = f"{staging_path}.aria2"
    control_temp_path = f"{control_path}__temp"
    final_path = os.path.join(destination, filename)

    for path in (staging_path, control_path, control_temp_path):
        if await asyncio.to_thread(os.path.islink, path):
            raise RuntimeError("CivitAI staging path must not be a symbolic link")
    await _remove_file_if_present(control_temp_path)

    if await asyncio.to_thread(
        os.path.isfile, staging_path
    ) and not await asyncio.to_thread(os.path.exists, control_path):
        if await _civitai_file_matches(
            staging_path, expected_size_bytes, expected_sha256
        ):
            await asyncio.to_thread(os.replace, staging_path, final_path)
            await _remove_file_if_present(control_path)
            return filename
        await _remove_file_if_present(staging_path)
    elif await asyncio.to_thread(
        os.path.exists, control_path
    ) and not await asyncio.to_thread(os.path.isfile, staging_path):
        await _remove_file_if_present(control_path)

    current_size = (
        await asyncio.to_thread(os.path.getsize, staging_path)
        if await asyncio.to_thread(os.path.isfile, staging_path)
        else 0
    )
    required_bytes = max(0, expected_size_bytes - current_size)
    free_bytes = await asyncio.to_thread(lambda: shutil.disk_usage(destination).free)
    required_with_reserve = required_bytes + CIVITAI_DISK_RESERVE_BYTES
    if free_bytes < required_with_reserve:
        raise RuntimeError(
            f"Not enough disk space: need {required_with_reserve / 1024**3:.2f} GB, "
            f"free {free_bytes / 1024**3:.2f} GB"
        )

    signed_url = await _resolve_civitai_download_url(url, file_id)
    command = _build_civitai_helper_command(
        destination, staging_filename, expected_sha256
    )
    subprocess_env = os.environ.copy()
    for key in tuple(subprocess_env):
        if key.casefold() == "civitai_token":
            subprocess_env.pop(key)

    log.info(f"executing command: {command}")
    try:
        proc = await asyncio.to_thread(
            subprocess.Popen,
            command,
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            env=subprocess_env,
        )
    except OSError as exc:
        raise RuntimeError("Could not start the CivitAI download helper") from exc

    try:
        output, _ = await asyncio.to_thread(
            proc.communicate,
            input=f"{signed_url}\n".encode(),
        )
    except asyncio.CancelledError:
        await asyncio.to_thread(proc.kill)
        await asyncio.to_thread(proc.wait)
        raise
    except OSError as exc:
        try:
            await asyncio.to_thread(proc.kill)
        except ProcessLookupError:
            pass
        await asyncio.to_thread(proc.wait)
        raise RuntimeError("Could not communicate with the CivitAI helper") from exc

    if proc.returncode != 0:
        diagnostic = output.decode("utf-8", errors="replace").strip()
        detail = f": {diagnostic}" if diagnostic else ""
        raise RuntimeError(
            f"CivitAI download helper exited with code {proc.returncode}{detail}"
        )
    if not await asyncio.to_thread(os.path.isfile, staging_path):
        raise RuntimeError("aria2c finished without creating the expected model file")

    actual_size = await asyncio.to_thread(os.path.getsize, staging_path)
    if actual_size != expected_size_bytes:
        await _remove_file_if_present(staging_path)
        await _remove_file_if_present(control_path)
        await _remove_file_if_present(control_temp_path)
        raise RuntimeError(
            f"Downloaded file size mismatch: expected {expected_size_bytes}, "
            f"got {actual_size}"
        )
    actual_sha256 = await compute_sha256(staging_path)
    if actual_sha256 != expected_sha256:
        await _remove_file_if_present(staging_path)
        await _remove_file_if_present(control_path)
        await _remove_file_if_present(control_temp_path)
        raise RuntimeError(
            "Downloaded file checksum mismatch: "
            f"expected {expected_sha256}, got {actual_sha256}"
        )

    await asyncio.to_thread(os.replace, staging_path, final_path)
    await _remove_file_if_present(control_path)
    await _remove_file_if_present(control_temp_path)
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
    expected_size_bytes: int | None = None,
    civitai_file_id: str | None = None,
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

        envs.get_environment_variable()
        t, destination = _get_download_destination(t)
        os.makedirs(destination, exist_ok=True)

        log.info(f"model will download into {destination}")
        log.info(f"Starting download: {name}")

        parsed_url = urlparse.urlparse(url)
        hostname = parsed_url.hostname or ""

        # Resolve a fresh signed URL immediately before starting CivitAI's aria2 job.
        if hostname in CIVITAI_HOSTS:
            try:
                await _download_civitai(
                    url,
                    destination,
                    filename=filename,
                    expected_sha256=expected_sha256,
                    expected_size_bytes=expected_size_bytes,
                    file_id=civitai_file_id,
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
            except Exception as e:  # noqa: BLE001 - download boundary records failure
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
                traceback.print_exception(e)
                log.error(f"Download failed: {name} ({e})")
                return False

        subprocess_env = None
        hf_staging_dir = None
        hf_source_path = None
        if hostname in HUGGINGFACE_HOSTS:
            filename = filename or _get_huggingface_filename(
                url, name, t, id, from_model_pack
            )

        hf_executable = shutil.which("hf") if hostname in HUGGINGFACE_HOSTS else None
        hf_target = (
            _parse_huggingface_download_url(url) if hf_executable is not None else None
        )

        if hf_executable is not None and hf_target is not None:
            hf_staging_dir, hf_source_path = _get_huggingface_staging_paths(
                destination, id, hf_target.filepath
            )
            os.makedirs(hf_staging_dir, exist_ok=True)
            cmd = _build_huggingface_cli_command(
                hf_executable, hf_target, hf_staging_dir
            )
            subprocess_env = os.environ.copy()
            if getattr(envs, "HUGGINGFACE_TOKEN", ""):
                subprocess_env["HF_TOKEN"] = envs.HUGGINGFACE_TOKEN
            log.info("Using hf CLI for Hugging Face download")
        else:
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
                aria2_cmd.append(f"--out={filename}")

                if getattr(envs, "HUGGINGFACE_TOKEN", ""):
                    aria2_cmd.append(
                        f"--header=Authorization: Bearer {envs.HUGGINGFACE_TOKEN}"
                    )

                fallback_reason = (
                    "hf CLI is not installed"
                    if hf_executable is None
                    else "the Hugging Face URL is not a supported file URL"
                )
                log.warning(f"{fallback_reason}; falling back to aria2c")

            cmd = aria2_cmd

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

        log.info(f"executing command: {_redact_command(cmd)}")

        # create subprocess, redirecting stdout/stderr
        try:
            proc = await asyncio.create_subprocess_exec(
                *cmd,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.STDOUT,
                limit=1024 * 1024,  # 1MB limit to handle long progress lines
                env=subprocess_env,
            )
        except Exception as e:  # noqa: BLE001 - subprocess boundary records failure
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
        failure_reason = f"exit code {return_code}"

        if (
            return_code == 0
            and hf_source_path is not None
            and hf_staging_dir is not None
            and filename is not None
        ):
            try:
                await asyncio.to_thread(
                    _finalize_huggingface_download,
                    hf_source_path,
                    destination,
                    filename,
                    hf_staging_dir,
                )
            except OSError as exc:
                return_code = 1
                failure_reason = str(exc)

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
            log.error(f"Download failed: {name} ({failure_reason})")
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
