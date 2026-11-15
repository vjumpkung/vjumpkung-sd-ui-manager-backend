import asyncio
import json
import re
import sys
import tempfile
import time
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
PROTECTED_PACKAGES = frozenset(
    {"numpy", "torch", "torchaudio", "torchvision", "transformers"}
)


class CustomNodeInstallError(Exception):
    """Raised when a custom node cannot be installed."""


class CustomNodeAlreadyInstalledError(CustomNodeInstallError):
    """Raised when the repository destination already exists."""


@dataclass(frozen=True)
class CustomNodeInstallResult:
    repository: str
    dependency_method: str
    restarted: bool = False
    failed_requirements: tuple[str, ...] = ()


@dataclass(frozen=True)
class CustomNodeBatchItemResult:
    url: str
    repository: str | None
    status: str
    dependency_method: str | None
    message: str
    failed_requirements: tuple[str, ...] = ()


@dataclass(frozen=True)
class RequirementEntry:
    line: str
    args: tuple[str, ...]
    name: str | None
    cwd: Path


@dataclass
class RequirementsPlan:
    entries: list[RequirementEntry]
    options: list[str]
    warnings: list[str]


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
_name_separator_pattern = re.compile(r"[-_.]+")
_comment_pattern = re.compile(r"(^|\s+)#.*$")
_egg_pattern = re.compile(r"#egg=([A-Za-z0-9][A-Za-z0-9._-]*)")
_wheel_filename_pattern = re.compile(r"([^/\\\s]+)\.whl(?:[?#]|$)", re.IGNORECASE)
_direct_reference_pattern = re.compile(
    r"^[A-Za-z0-9][A-Za-z0-9._-]*\s*(?:\[[^\]]*\])?\s*@"
)
_package_name_pattern = re.compile(
    r"^([A-Za-z0-9][A-Za-z0-9._-]*)(?=$|[\s\[(<>=!~;@,])"
)
_option_pattern = re.compile(r"^(--?[A-Za-z][A-Za-z0-9-]*)(?:\s*=\s*|\s+)?(.*)$")
_per_requirement_option_pattern = re.compile(r"\s--")
_value_options = {
    "-i": "--index-url",
    "--index-url": "--index-url",
    "--extra-index-url": "--extra-index-url",
    "-f": "--find-links",
    "--find-links": "--find-links",
    "-c": "--constraint",
    "--constraint": "--constraint",
}
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


def _normalize_package_name(name: str) -> str:
    return _name_separator_pattern.sub("-", name).lower()


def _requirement_lines(text: str) -> list[str]:
    lines: list[str] = []
    pending = ""

    # The trailing sentinel flushes a file that ends with a continuation.
    for raw_line in [*text.removeprefix("﻿").splitlines(), ""]:
        line = raw_line.rstrip()
        if line.endswith("\\") and not line.lstrip().startswith("#"):
            pending += line[:-1]
            continue

        cleaned = _comment_pattern.sub("", pending + line).strip()
        pending = ""
        if cleaned:
            lines.append(cleaned)

    return lines


def _requirement_name(spec: str) -> str | None:
    egg = _egg_pattern.search(spec)
    if egg:
        return _normalize_package_name(egg.group(1))

    if not _direct_reference_pattern.match(spec):
        wheel = _wheel_filename_pattern.search(spec)
        if wheel:
            filename = unquote(wheel.group(1))
            return _normalize_package_name(filename.split("-", maxsplit=1)[0])

    name = _package_name_pattern.match(spec)
    if name:
        return _normalize_package_name(name.group(1))

    return None


def _add_option(options: list[str], *group: str) -> None:
    if any(tuple(options[i : i + len(group)]) == group for i in range(len(options))):
        return
    options.extend(group)


def _merge_requirements_plan(
    target: RequirementsPlan, source: RequirementsPlan
) -> None:
    target.entries.extend(source.entries)
    target.warnings.extend(source.warnings)

    source_options = iter(source.options)
    for option in source_options:
        if option in _value_options.values():
            _add_option(target.options, option, next(source_options))
        else:
            _add_option(target.options, option)


def _load_requirements(path: Path, seen: set[Path] | None = None) -> RequirementsPlan:
    plan = RequirementsPlan(entries=[], options=[], warnings=[])
    seen = set() if seen is None else seen
    resolved = path.resolve()
    if resolved in seen:
        return plan
    seen.add(resolved)

    try:
        text = path.read_text(encoding="utf-8", errors="replace")
    except OSError as error:
        plan.warnings.append(f"Could not read {path.name}: {error}")
        return plan

    for line in _requirement_lines(text):
        if not line.startswith("-"):
            spec, *extra = _per_requirement_option_pattern.split(line, maxsplit=1)
            if extra:
                plan.warnings.append(f"Ignoring per-requirement options in: {line}")
            spec = spec.strip()
            plan.entries.append(
                RequirementEntry(
                    line=spec,
                    args=(spec,),
                    name=_requirement_name(spec),
                    cwd=path.parent,
                )
            )
            continue

        match = _option_pattern.match(line)
        flag, value = (match.group(1), match.group(2).strip()) if match else (line, "")

        if flag in {"-r", "--requirement"} and value:
            included = path.parent / value
            if included.is_file():
                _merge_requirements_plan(plan, _load_requirements(included, seen))
            else:
                plan.warnings.append(f"Included requirements file not found: {value}")
        elif flag in _value_options and value:
            option = _value_options[flag]
            if option == "--constraint" and not value.startswith(
                ("http://", "https://")
            ):
                value = str((path.parent / value).resolve())
            _add_option(plan.options, option, value)
        elif flag == "--no-index" and not value:
            _add_option(plan.options, "--no-index")
        elif flag == "--pre" and not value:
            _add_option(plan.options, "--prerelease=allow")
        elif flag in {"-e", "--editable"} and value:
            plan.entries.append(
                RequirementEntry(
                    line=f"-e {value}",
                    args=("-e", value),
                    name=_requirement_name(value),
                    cwd=path.parent,
                )
            )
        else:
            plan.warnings.append(f"Ignoring unsupported requirements option: {line}")

    return plan


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


async def _emit(on_line: Callable[[str], Awaitable[None]] | None, line: str) -> None:
    if on_line is not None:
        await on_line(line)
    else:
        log.debug(line)


def _elapsed(started: float) -> str:
    return f"{time.monotonic() - started:.1f}s"


def _summarize_requirements(requirements: tuple[str, ...]) -> str:
    summary = ", ".join(requirements[:5])
    if len(requirements) > 5:
        summary = f"{summary} and {len(requirements) - 5} more"
    return summary


async def _installed_protected_versions(
    cwd: Path,
    on_line: Callable[[str], Awaitable[None]] | None = None,
) -> dict[str, str]:
    output: list[str] = []

    async def collect(line: str) -> None:
        output.append(line)

    failure: CustomNodeInstallError | None = None
    try:
        await _run_command(["uv", "pip", "list", "--format", "json"], cwd, collect)
    except CustomNodeInstallError as error:
        failure = error

    json_index = next(
        (i for i in reversed(range(len(output))) if output[i].startswith("[")), None
    )
    for index, line in enumerate(output):
        if index != json_index:
            await _emit(on_line, line)

    prefix = "Could not read installed package versions"
    if failure is not None:
        raise CustomNodeInstallError(f"{prefix}: {failure}") from failure
    if json_index is None:
        raise CustomNodeInstallError(f"{prefix}: uv pip list returned no JSON.")

    try:
        versions: dict[str, str] = {}
        for package in json.loads(output[json_index]):
            name = _normalize_package_name(package["name"])
            if name in PROTECTED_PACKAGES:
                versions[name] = str(package["version"])
    except (ValueError, KeyError, TypeError) as error:
        raise CustomNodeInstallError(f"{prefix}: {error!r}") from error

    return versions


async def _install_requirements(
    requirements_file: Path,
    repository_path: Path,
    on_line: Callable[[str], Awaitable[None]] | None = None,
) -> tuple[str, ...]:
    plan = await asyncio.to_thread(_load_requirements, requirements_file)
    entries = plan.entries
    total = len(entries)
    relative = requirements_file.relative_to(repository_path).as_posix()
    installable = [entry for entry in entries if entry.name not in PROTECTED_PACKAGES]

    await _emit(
        on_line, f"Found requirements.txt at {relative} ({total} requirement(s))"
    )
    for warning in plan.warnings:
        await _emit(on_line, warning)
    if plan.options:
        await _emit(
            on_line,
            f"Requirements options applied to every install: {' '.join(plan.options)}",
        )
    await _emit(
        on_line,
        "Protected packages (never installed or changed): "
        f"{', '.join(sorted(PROTECTED_PACKAGES))}",
    )

    versions: dict[str, str] | None = None
    pins: list[str] = []
    if installable:
        await _emit(on_line, "Reading installed versions: uv pip list --format json")
        versions = await _installed_protected_versions(
            requirements_file.parent, on_line
        )
        pins = [f"{name}=={versions[name]}" for name in sorted(versions)]
        pinned = ", ".join(pins) if pins else "none installed"
        await _emit(on_line, f"Pinned protected versions: {pinned}")
        missing = sorted(PROTECTED_PACKAGES - versions.keys())
        if missing:
            await _emit(
                on_line, f"Protected packages not installed: {', '.join(missing)}"
            )
    else:
        await _emit(on_line, "Nothing to install from requirements.txt.")

    started = time.monotonic()
    installed_count = 0
    skipped_count = 0
    failed: list[str] = []

    with tempfile.TemporaryDirectory() as temp_dir:
        constraint_args: list[str] = []
        if pins:
            constraints_file = Path(temp_dir) / "constraints.txt"
            await asyncio.to_thread(
                constraints_file.write_text, "\n".join(pins) + "\n", encoding="utf-8"
            )
            constraint_args = ["-c", str(constraints_file)]

        for number, entry in enumerate(entries, start=1):
            prefix = f"[{number}/{total}]"

            if entry.name in PROTECTED_PACKAGES:
                skipped_count += 1
                if versions is None:
                    reason = "protected"
                elif entry.name in versions:
                    version = versions[entry.name]
                    reason = f"protected, installed {entry.name}=={version}"
                else:
                    reason = "protected, not installed"
                await _emit(on_line, f"{prefix} SKIPPED {entry.line} ({reason})")
                continue

            await _emit(on_line, f"{prefix} Installing {entry.line}")
            entry_started = time.monotonic()
            try:
                await _run_command(
                    ["uv", "pip", "install", *constraint_args, *plan.options]
                    + list(entry.args),
                    entry.cwd,
                    on_line,
                )
            except CustomNodeInstallError as error:
                failed.append(entry.line)
                reason = str(error).split("\n", maxsplit=1)[0]
                elapsed = _elapsed(entry_started)
                await _emit(
                    on_line, f"{prefix} FAILED {entry.line}: {reason} ({elapsed})"
                )
            else:
                installed_count += 1
                elapsed = _elapsed(entry_started)
                await _emit(on_line, f"{prefix} OK {entry.line} ({elapsed})")

    await _emit(
        on_line,
        f"Dependencies finished in {_elapsed(started)}: {installed_count} installed, "
        f"{skipped_count} skipped (protected), {len(failed)} failed",
    )
    for line in failed:
        await _emit(on_line, f"Failed requirement: {line}")

    if installable and not installed_count:
        raise CustomNodeInstallError(
            f"Could not install any requirements from {relative}: "
            f"{_summarize_requirements(tuple(failed))}"
        )

    return tuple(failed)


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
    failed_requirements: tuple[str, ...] = ()

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
        failed_requirements = await _install_requirements(
            requirements_file, repository_path, on_line
        )
        dependency_method = "requirements.txt"

    return CustomNodeInstallResult(
        repository=repository_name,
        dependency_method=dependency_method,
        failed_requirements=failed_requirements,
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
        failed_requirements=result.failed_requirements,
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
                message = f"Installed {result.repository}."
                if result.failed_requirements:
                    message = (
                        f"Installed {result.repository}, but "
                        f"{len(result.failed_requirements)} requirement(s) failed: "
                        f"{_summarize_requirements(result.failed_requirements)}"
                    )
                item_result = CustomNodeBatchItemResult(
                    url=repository_url,
                    repository=result.repository,
                    status="installed",
                    dependency_method=result.dependency_method,
                    message=message,
                    failed_requirements=result.failed_requirements,
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
    failed_requirements_count = sum(
        item.status == "installed" and bool(item.failed_requirements)
        for item in result.results
    )

    if failed_count == len(result.results):
        status = "failed"
    elif failed_count or failed_requirements_count:
        status = "partial"
    else:
        status = "completed"

    message = (
        f"Processed {len(result.results)} custom node(s): "
        f"{installed_count} installed, "
        f"{already_installed_count} already installed, "
        f"{failed_count} failed."
    )
    if failed_requirements_count:
        message = (
            f"{message} {failed_requirements_count} installed node(s) "
            "have failed requirements."
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
