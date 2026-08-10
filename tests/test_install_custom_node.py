import asyncio
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import AsyncMock, patch

from worker import install_custom_node


class FakeProcess:
    def __init__(self, output: str = "", return_code: int = 0):
        self.stdout = asyncio.StreamReader()
        if output:
            self.stdout.feed_data(output.encode())
        self.stdout.feed_eof()
        self.return_code = return_code

    async def wait(self) -> int:
        return self.return_code

    def kill(self) -> None:
        self.return_code = -1


class RepositoryNameTests(unittest.TestCase):
    def test_extracts_repository_name(self) -> None:
        self.assertEqual(
            install_custom_node._repository_name(
                "https://github.com/owner/example-node.git"
            ),
            "example-node",
        )

    def test_rejects_credentials(self) -> None:
        with self.assertRaises(install_custom_node.CustomNodeInstallError):
            install_custom_node._repository_name(
                "https://token@github.com/owner/example-node.git"
            )


class InstallCustomNodeTests(unittest.IsolatedAsyncioTestCase):
    async def test_runs_install_script_and_restarts(self) -> None:
        commands: list[tuple[tuple[str, ...], str]] = []

        async def create_process(*command, **kwargs):
            commands.append((command, kwargs["cwd"]))
            if command[:2] == ("git", "clone"):
                repository_path = Path(command[-1])
                repository_path.mkdir()
                repository_path.joinpath("install.py").write_text("", encoding="utf-8")
                repository_path.joinpath("requirements.txt").write_text(
                    "ignored", encoding="utf-8"
                )
            return FakeProcess()

        with tempfile.TemporaryDirectory() as temp_dir:
            with (
                patch.object(install_custom_node, "COMFYUI_PATH", temp_dir),
                patch.object(
                    install_custom_node.asyncio,
                    "create_subprocess_exec",
                    side_effect=create_process,
                ),
                patch.object(
                    install_custom_node,
                    "restart_program",
                    new=AsyncMock(),
                ) as restart,
            ):
                result = await install_custom_node.install_custom_node(
                    "https://github.com/owner/example-node.git"
                )

        self.assertEqual(result.repository, "example-node")
        self.assertEqual(result.dependency_method, "install.py")
        self.assertEqual(commands[1][0], (sys.executable, "install.py"))
        self.assertEqual(len(commands), 2)
        restart.assert_awaited_once()

    async def test_uses_uv_requirements_when_install_script_is_absent(self) -> None:
        commands: list[tuple[str, ...]] = []

        async def create_process(*command, **kwargs):
            commands.append(command)
            if command[:2] == ("git", "clone"):
                repository_path = Path(command[-1])
                repository_path.mkdir()
                repository_path.joinpath("requirements.txt").write_text(
                    "package", encoding="utf-8"
                )
            return FakeProcess()

        with tempfile.TemporaryDirectory() as temp_dir:
            with (
                patch.object(install_custom_node, "COMFYUI_PATH", temp_dir),
                patch.object(
                    install_custom_node.asyncio,
                    "create_subprocess_exec",
                    side_effect=create_process,
                ),
                patch.object(
                    install_custom_node,
                    "restart_program",
                    new=AsyncMock(),
                ) as restart,
            ):
                result = await install_custom_node.install_custom_node(
                    "https://github.com/owner/requirements-node"
                )

        self.assertEqual(result.dependency_method, "requirements.txt")
        self.assertEqual(commands[1][:4], ("uv", "pip", "install", "-r"))
        restart.assert_awaited_once()

    async def test_does_not_restart_after_failed_install(self) -> None:
        process_count = 0

        async def create_process(*command, **kwargs):
            nonlocal process_count
            process_count += 1
            if process_count == 1:
                repository_path = Path(command[-1])
                repository_path.mkdir()
                repository_path.joinpath("install.py").write_text("", encoding="utf-8")
                return FakeProcess()
            return FakeProcess("installation failed", return_code=1)

        with tempfile.TemporaryDirectory() as temp_dir:
            with (
                patch.object(install_custom_node, "COMFYUI_PATH", temp_dir),
                patch.object(
                    install_custom_node.asyncio,
                    "create_subprocess_exec",
                    side_effect=create_process,
                ),
                patch.object(
                    install_custom_node,
                    "restart_program",
                    new=AsyncMock(),
                ) as restart,
            ):
                with self.assertRaises(install_custom_node.CustomNodeInstallError):
                    await install_custom_node.install_custom_node(
                        "https://github.com/owner/failing-node"
                    )

        restart.assert_not_awaited()


if __name__ == "__main__":
    unittest.main()
