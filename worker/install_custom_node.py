import asyncio
import re
import sys
import uuid
from collections import deque
from collections.abc import Awaitable, Callable
from dataclasses import dataclass, field
from pathlib import Path
from urllib.parse import unquote, urlsplit

from pydantic import BaseModel

from config.load_config import COMFYUI_PATH
from event_handler import manager
from log_manager import log
from utils.ws_messages import (
    CustomNodeData,
    CustomNodeJobData,
    CustomNodeJobMessage,
    CustomNodeJobStatus,
    CustomNodeLogData,
    CustomNodeLogMessage,
    CustomNodeMessage,
)
from worker.check_process import UIPort
from worker.restart_program import restart_program

LOG_BUFFER_LINES = 200
MAX_JOBS = 20


class CustomNodeInstallError(Exception):
    """Raised when a custom node cannot be installed."""


class CustomNodeAlreadyInstalledError(CustomNodeInstallError):
    """Raised when the repository destination already exists."""


@dataclass(frozen=True)
class CustomNodeInstallResult:
    repository: str
    dependency_method: str
    restarted: bool = False


@dataclass(frozen=True)
class CustomNodeBatchItemResult:
    url: str
    repository: str | None
    status: str
    dependency_method: str | None
    message: str


@dataclass(frozen=True)
class CustomNodeBatchInstallResult:
    results: tuple[CustomNodeBatchItemResult, ...]
    restarted: bool
    restart_error: str | None = None


@dataclass
class CustomNodeJobItem:
    index: int
    url: str
    repository: str | None
    status: str = "queued"
    dependency_method: str | None = None
    message: str = "Waiting to install."
    logs: deque[str] = field(default_factory=lambda: deque(maxlen=LOG_BUFFER_LINES))


@dataclass
class CustomNodeJob:
    job_id: str
    items: list[CustomNodeJobItem]
    status: str = "queued"
    message: str = ""
    restarted: bool = False

    @property
    def finished(self) -> bool:
        return self.status in {"completed", "partial", "failed"}


_install_lock = asyncio.Lock()
_repository_name_pattern = re.compile(r"^[A-Za-z0-9._-]+$")
_jobs: dict[str, CustomNodeJob] = {}
active_install_tasks: set[asyncio.Task[None]] = set()


async def _is_comfyui_running(
    host: str = "127.0.0.1", port: int = UIPort.COMFY.value
) -> bool:
    try:
        _, writer = await asyncio.open_connection(host, port)
    except (ConnectionRefusedError, OSError):
        return False

    writer.close()
    await writer.wait_closed()
    return True


def _repository_name(repository_url: str) -> str:
    parsed = urlsplit(repository_url)
    if parsed.scheme not in {"http", "https"} or not parsed.netloc:
        raise CustomNodeInstallError("Repository URL must use HTTP or HTTPS.")
    if parsed.username is not None or parsed.password is not None:
        raise CustomNodeInstallError(
            "Repository URLs containing credentials are not supported."
        )

    name = unquote(parsed.path.rstrip("/").rsplit("/", maxsplit=1)[-1])
    if name.lower().endswith(".git"):
        name = name[:-4]

    if not name or name in {".", ".."} or not _repository_name_pattern.fullmatch(name):
        raise CustomNodeInstallError(
            "Repository URL does not contain a valid repository name."
        )

    return name


def _safe_repository_name(repository_url: str) -> str | None:
    try:
        return _repository_name(repository_url)
    except CustomNodeInstallError:
        return None


def _find_repository_file(repository_path: Path, filename: str) -> Path | None:
    root_file = repository_path / filename
    if root_file.is_file():
        return root_file

    matches = [
        path
        for path in repository_path.rglob(filename)
        if ".git" not in path.relative_to(repository_path).parts and path.is_file()
    ]
    if not matches:
        return None

    return min(
        matches,
        key=lambda path: (
            len(path.relative_to(repository_path).parts),
            str(path.relative_to(repository_path)),
        ),
    )


async def _run_command(
    command: list[str],
    cwd: Path,
    on_line: Callable[[str], Awaitable[None]] | None = None,
) -> None:
    try:
        process = await asyncio.create_subprocess_exec(
            *command,
            cwd=str(cwd),
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.STDOUT,
            limit=1024 * 1024,
        )
    except OSError as error:
        raise CustomNodeInstallError(
            f"Could not start {command[0]}: {error}"
        ) from error

    output: deque[str] = deque(maxlen=20)

    try:
        assert process.stdout is not None
        async for raw_line in process.stdout:
            line = raw_line.decode("utf-8", errors="replace").rstrip().split("\r")[-1]
            output.append(line)
            if on_line is not None:
                await on_line(line)
            else:
                log.debug(line)
        return_code = await process.wait()
    except BaseException:
        if process.returncode is None:
            process.kill()
        await process.wait()
        raise

    if return_code != 0:
        details = "\n".join(output).strip()
        message = f"Command failed with exit code {return_code}."
        if details:
            message = f"{message}\n{details}"
        raise CustomNodeInstallError(message)


async def _install_custom_node(
    repository_url: str,
    on_line: Callable[[str], Awaitable[None]] | None = None,
) -> CustomNodeInstallResult:
    repository_name = _repository_name(repository_url)
    custom_nodes_path = (Path(COMFYUI_PATH).expanduser() / "custom_nodes").resolve()
    await asyncio.to_thread(custom_nodes_path.mkdir, parents=True, exist_ok=True)

    repository_path = (custom_nodes_path / repository_name).resolve()
    if repository_path.parent != custom_nodes_path:
        raise CustomNodeInstallError("Invalid custom node destination.")
    if await asyncio.to_thread(repository_path.exists):
        raise CustomNodeAlreadyInstalledError(
            f"Custom node '{repository_name}' is already installed."
        )

    if on_line is not None:
        await on_line(f"Cloning {repository_url}")
    await _run_command(
        [
            "git",
            "clone",
            "--depth=1",
            "--single-branch",
            "--no-tags",
            "--",
            repository_url,
            str(repository_path),
        ],
        custom_nodes_path,
        on_line,
    )

    install_script = await asyncio.to_thread(
        _find_repository_file, repository_path, "install.py"
    )
    dependency_method = "none"

    if install_script is not None:
        if on_line is not None:
            await on_line(f"Running {install_script.relative_to(repository_path)}")
        await _run_command(
            [sys.executable, install_script.name], install_script.parent, on_line
        )
        dependency_method = "install.py"

    requirements_file = await asyncio.to_thread(
        _find_repository_file, repository_path, "requirements.txt"
    )
    if requirements_file is not None:
        if on_line is not None:
            await on_line(
                f"Installing {requirements_file.relative_to(repository_path)} with uv"
            )
        await _run_command(
            ["uv", "pip", "install", "-r", str(requirements_file)],
            requirements_file.parent,
            on_line,
        )
        dependency_method = "requirements.txt"

    return CustomNodeInstallResult(
        repository=repository_name,
        dependency_method=dependency_method,
    )


async def _restart_comfyui_if_running() -> bool:
    if not await _is_comfyui_running():
        log.debug("ComfyUI is not running; skipping restart.")
        return False

    await restart_program()
    return True


async def install_custom_node(repository_url: str) -> CustomNodeInstallResult:
    async with _install_lock:
        result = await _install_custom_node(repository_url)
        restarted = await _restart_comfyui_if_running()

    return CustomNodeInstallResult(
        repository=result.repository,
        dependency_method=result.dependency_method,
        restarted=restarted,
    )


class InstallReporter:
    async def batch_started(self) -> None: ...

    async def item_started(self, index: int) -> None: ...

    async def item_output(self, index: int, line: str) -> None: ...

    async def item_finished(
        self, index: int, result: CustomNodeBatchItemResult
    ) -> None: ...


async def install_custom_nodes(
    repository_urls: list[str],
    reporter: InstallReporter | None = None,
) -> CustomNodeBatchInstallResult:
    reporter = reporter or InstallReporter()
    results: list[CustomNodeBatchItemResult] = []

    async with _install_lock:
        await reporter.batch_started()

        for index, repository_url in enumerate(repository_urls):
            label = _safe_repository_name(repository_url) or repository_url
            await reporter.item_started(index)

            async def on_line(line: str, index: int = index, label: str = label):
                log.info(f"[{label}] {line}")
                await reporter.item_output(index, line)

            try:
                result = await _install_custom_node(repository_url, on_line)
            except CustomNodeAlreadyInstalledError as error:
                item_result = CustomNodeBatchItemResult(
                    url=repository_url,
                    repository=_repository_name(repository_url),
                    status="already_installed",
                    dependency_method=None,
                    message=str(error),
                )
            except CustomNodeInstallError as error:
                item_result = CustomNodeBatchItemResult(
                    url=repository_url,
                    repository=_safe_repository_name(repository_url),
                    status="failed",
                    dependency_method=None,
                    message=str(error),
                )
            except Exception as error:
                log.exception(f"[{label}] Unexpected error while installing.")
                item_result = CustomNodeBatchItemResult(
                    url=repository_url,
                    repository=_safe_repository_name(repository_url),
                    status="failed",
                    dependency_method=None,
                    message=f"Unexpected error: {error}",
                )
            else:
                item_result = CustomNodeBatchItemResult(
                    url=repository_url,
                    repository=result.repository,
                    status="installed",
                    dependency_method=result.dependency_method,
                    message=f"Installed {result.repository}.",
                )

            results.append(item_result)
            log.info(f"[{label}] {item_result.status}")
            await reporter.item_finished(index, item_result)

        installed_any = any(result.status == "installed" for result in results)
        restarted = False
        restart_error: str | None = None
        if installed_any:
            try:
                restarted = await _restart_comfyui_if_running()
            except Exception as error:
                log.exception("Could not restart ComfyUI after installing nodes.")
                restart_error = str(error) or type(error).__name__

    return CustomNodeBatchInstallResult(
        results=tuple(results),
        restarted=restarted,
        restart_error=restart_error,
    )


def summarize_install(
    result: CustomNodeBatchInstallResult,
) -> tuple[CustomNodeJobStatus, str]:
    installed_count = sum(item.status == "installed" for item in result.results)
    already_installed_count = sum(
        item.status == "already_installed" for item in result.results
    )
    failed_count = sum(item.status == "failed" for item in result.results)

    if failed_count == len(result.results):
        status = "failed"
    elif failed_count:
        status = "partial"
    else:
        status = "completed"

    message = (
        f"Processed {len(result.results)} custom node(s): "
        f"{installed_count} installed, "
        f"{already_installed_count} already installed, "
        f"{failed_count} failed."
    )
    if result.restarted:
        message = f"{message} ComfyUI was restarted."
    if result.restart_error:
        message = f"{message} ComfyUI restart failed: {result.restart_error}"

    return status, message


async def _broadcast(message: BaseModel) -> None:
    try:
        await manager.broadcast(message.model_dump_json())
    except Exception:
        log.debug("Could not broadcast custom node message.", exc_info=True)


async def _broadcast_item(job: CustomNodeJob, item: CustomNodeJobItem) -> None:
    await _broadcast(
        CustomNodeMessage(
            data=CustomNodeData(
                job_id=job.job_id,
                index=item.index,
                url=item.url,
                repository=item.repository,
                status=item.status,
                dependency_method=item.dependency_method,
                message=item.message,
            )
        )
    )


async def _broadcast_job(job: CustomNodeJob) -> None:
    await _broadcast(
        CustomNodeJobMessage(
            data=CustomNodeJobData(
                job_id=job.job_id,
                status=job.status,
                message=job.message,
                restarted=job.restarted,
            )
        )
    )


class _JobReporter(InstallReporter):
    def __init__(self, job: CustomNodeJob):
        self.job = job

    async def batch_started(self) -> None:
        self.job.status = "running"
        self.job.message = f"Installing {len(self.job.items)} custom node(s)."
        await _broadcast_job(self.job)

    async def item_started(self, index: int) -> None:
        item = self.job.items[index]
        item.status = "installing"
        item.message = "Installing..."
        await _broadcast_item(self.job, item)

    async def item_output(self, index: int, line: str) -> None:
        item = self.job.items[index]
        item.logs.append(line)
        await _broadcast(
            CustomNodeLogMessage(
                data=CustomNodeLogData(
                    job_id=self.job.job_id,
                    index=item.index,
                    url=item.url,
                    repository=item.repository,
                    line=line,
                )
            )
        )

    async def item_finished(
        self, index: int, result: CustomNodeBatchItemResult
    ) -> None:
        item = self.job.items[index]
        item.status = result.status
        item.repository = result.repository
        item.dependency_method = result.dependency_method
        item.message = result.message
        await _broadcast_item(self.job, item)


def _prune_jobs() -> None:
    for job_id in list(_jobs):
        if len(_jobs) <= MAX_JOBS:
            break
        if _jobs[job_id].finished:
            del _jobs[job_id]


def create_install_job(repository_urls: list[str]) -> CustomNodeJob:
    job = CustomNodeJob(
        job_id=uuid.uuid4().hex,
        items=[
            CustomNodeJobItem(
                index=index,
                url=repository_url,
                repository=_safe_repository_name(repository_url),
            )
            for index, repository_url in enumerate(repository_urls)
        ],
        message=f"Queued {len(repository_urls)} custom node(s) for installation.",
    )
    _jobs[job.job_id] = job
    _prune_jobs()
    return job


def _fail_unfinished_job(job: CustomNodeJob, message: str) -> list[CustomNodeJobItem]:
    unfinished = [item for item in job.items if item.status in {"queued", "installing"}]
    for item in unfinished:
        item.status = "failed"
        item.message = "Not installed because the job stopped unexpectedly."
    job.status = "failed"
    job.message = message
    return unfinished


async def run_install_job(job: CustomNodeJob) -> None:
    await _broadcast_job(job)
    for item in job.items:
        await _broadcast_item(job, item)

    try:
        result = await install_custom_nodes(
            [item.url for item in job.items], reporter=_JobReporter(job)
        )
    except asyncio.CancelledError:
        _fail_unfinished_job(job, "Custom node installation was cancelled.")
        raise
    except Exception:
        log.exception(f"Custom node installation job {job.job_id} failed.")
        unfinished = _fail_unfinished_job(
            job, "Custom node installation stopped unexpectedly."
        )
        for item in unfinished:
            await _broadcast_item(job, item)
        await _broadcast_job(job)
        return

    job.restarted = result.restarted
    job.status, job.message = summarize_install(result)
    log.info(f"Custom node installation job {job.job_id}: {job.message}")
    await _broadcast_job(job)


def start_install_job(repository_urls: list[str]) -> CustomNodeJob:
    job = create_install_job(repository_urls)
    task = asyncio.create_task(run_install_job(job))
    active_install_tasks.add(task)
    task.add_done_callback(active_install_tasks.discard)
    return job


def get_install_job(job_id: str) -> CustomNodeJob | None:
    return _jobs.get(job_id)
