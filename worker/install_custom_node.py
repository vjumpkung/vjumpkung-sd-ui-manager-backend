import asyncio
import re
import sys
from collections import deque
from dataclasses import dataclass
from pathlib import Path
from urllib.parse import unquote, urlsplit

from config.load_config import COMFYUI_PATH
from log_manager import log
from worker.check_process import UIPort
from worker.restart_program import restart_program


class CustomNodeInstallError(Exception):
    """Raised when a custom node cannot be installed."""


class CustomNodeAlreadyInstalledError(CustomNodeInstallError):
    """Raised when the repository destination already exists."""


@dataclass(frozen=True)
class CustomNodeInstallResult:
    repository: str
    dependency_method: str


_install_lock = asyncio.Lock()
_repository_name_pattern = re.compile(r"^[A-Za-z0-9._-]+$")


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


async def _run_command(command: list[str], cwd: Path) -> None:
    try:
        process = await asyncio.create_subprocess_exec(
            *command,
            cwd=str(cwd),
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.STDOUT,
        )
    except OSError as error:
        raise CustomNodeInstallError(
            f"Could not start {command[0]}: {error}"
        ) from error

    output: deque[str] = deque(maxlen=20)

    try:
        assert process.stdout is not None
        async for raw_line in process.stdout:
            line = raw_line.decode("utf-8", errors="replace").rstrip()
            output.append(line)
            log.debug(line)
        return_code = await process.wait()
    except asyncio.CancelledError:
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


async def install_custom_node(repository_url: str) -> CustomNodeInstallResult:
    repository_name = _repository_name(repository_url)

    async with _install_lock:
        custom_nodes_path = (Path(COMFYUI_PATH).expanduser() / "custom_nodes").resolve()
        await asyncio.to_thread(custom_nodes_path.mkdir, parents=True, exist_ok=True)

        repository_path = (custom_nodes_path / repository_name).resolve()
        if repository_path.parent != custom_nodes_path:
            raise CustomNodeInstallError("Invalid custom node destination.")
        if await asyncio.to_thread(repository_path.exists):
            raise CustomNodeAlreadyInstalledError(
                f"Custom node '{repository_name}' is already installed."
            )

        await _run_command(
            ["git", "clone", "--", repository_url, str(repository_path)],
            custom_nodes_path,
        )

        install_script = await asyncio.to_thread(
            _find_repository_file, repository_path, "install.py"
        )
        dependency_method = "none"

        if install_script is not None:
            await _run_command(
                [sys.executable, install_script.name], install_script.parent
            )
            dependency_method = "install.py"

        # always install with requirements.txt
        requirements_file = await asyncio.to_thread(
            _find_repository_file, repository_path, "requirements.txt"
        )
        if requirements_file is not None:
            await _run_command(
                ["uv", "pip", "install", "-r", str(requirements_file)],
                requirements_file.parent,
            )
            dependency_method = "requirements.txt"

        if await _is_comfyui_running():
            await restart_program()
        else:
            log.debug("ComfyUI is not running; skipping restart.")

        return CustomNodeInstallResult(
            repository=repository_name,
            dependency_method=dependency_method,
        )
