import asyncio
import json
import sys
import tempfile
import unittest
from collections.abc import Iterable, Iterator
from contextlib import contextmanager
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import AsyncMock, patch

from worker import install_custom_node

REPOSITORY_URL = "https://github.com/owner/example-node.git"
INSTALL = ("uv", "pip", "install")


class FakeProcess:
    returncode = None

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


class ProcessRecorder:
    """Fake subprocess factory that clones a repository with a requirements file."""

    def __init__(
        self,
        requirements: str | None,
        *,
        requirements_path: str = "requirements.txt",
        list_output: str = "[]\n",
        list_return_code: int = 0,
        failing: Iterable[str] = (),
    ):
        self.requirements = requirements
        self.requirements_path = requirements_path
        self.list_output = list_output
        self.list_return_code = list_return_code
        self.failing = set(failing)
        self.commands: list[tuple[str, ...]] = []
        self.cwds: list[str] = []
        self.constraints: list[str | None] = []
        self.constraint_files: list[Path] = []

    @property
    def install_commands(self) -> list[tuple[str, ...]]:
        return [command for command in self.commands if command[:3] == INSTALL]

    async def create_process(self, *command, **kwargs):
        self.commands.append(command)
        self.cwds.append(kwargs["cwd"])

        if command[:2] == ("git", "clone"):
            repository_path = Path(command[-1])
            repository_path.mkdir()
            if self.requirements is not None:
                requirements_file = repository_path / self.requirements_path
                requirements_file.parent.mkdir(parents=True, exist_ok=True)
                requirements_file.write_text(self.requirements, encoding="utf-8")
        elif command[:3] == ("uv", "pip", "list"):
            return FakeProcess(self.list_output, self.list_return_code)
        elif command[:3] == INSTALL:
            if "-c" in command:
                constraints_file = Path(command[command.index("-c") + 1])
                self.constraint_files.append(constraints_file)
                self.constraints.append(constraints_file.read_text(encoding="utf-8"))
            else:
                self.constraints.append(None)
            if command[-1] in self.failing:
                return FakeProcess("ERROR: no matching distribution\n", 1)
            return FakeProcess("resolved and installed\n")

        return FakeProcess()


@contextmanager
def patched_environment(
    recorder: ProcessRecorder, *, comfyui_running: bool = False
) -> Iterator[str]:
    with (
        tempfile.TemporaryDirectory() as temp_dir,
        patch.object(install_custom_node, "COMFYUI_PATH", temp_dir),
        patch.object(
            install_custom_node.asyncio,
            "create_subprocess_exec",
            side_effect=recorder.create_process,
        ),
        patch.object(install_custom_node, "restart_program", new=AsyncMock()),
        patch.object(
            install_custom_node,
            "_is_comfyui_running",
            new=AsyncMock(return_value=comfyui_running),
        ),
    ):
        yield temp_dir


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


class RequirementParsingTests(unittest.TestCase):
    def test_requirement_lines_clean_comments_and_continuations(self) -> None:
        text = (
            "﻿# header comment\r\n"
            "numpy>=1.26  # inline comment\r\n"
            "\r\n"
            "pkg>=1,\\\r\n"
            "<2\r\n"
            "git+https://example.com/y.git#egg=bar\r\n"
            "baz # trailing\r\n"
            "   \r\n"
            "# comment with continuation \\\r\n"
            "last"
        )

        self.assertEqual(
            install_custom_node._requirement_lines(text),
            [
                "numpy>=1.26",
                "pkg>=1,<2",
                "git+https://example.com/y.git#egg=bar",
                "baz",
                "last",
            ],
        )

    def test_requirement_name_extracts_normalized_package_names(self) -> None:
        wheel_url = (
            "https://download.pytorch.org/whl/cu121/"
            "torch-2.1.0%2Bcu121-cp311-cp311-linux_x86_64.whl"
        )
        cases = {
            "NumPy>=1.26": "numpy",
            "torch==2.1.0+cu121": "torch",
            "Transformers[torch]>=4": "transformers",
            "torchvision ; sys_platform != 'darwin'": "torchvision",
            "torchaudio @ https://example.com/a.whl": "torchaudio",
            wheel_url: "torch",
            "git+https://example.com/a/b.git#egg=numpy": "numpy",
            "git+https://github.com/a/b.git": None,
            "torchsde": "torchsde",
            "numpy_quaternion": "numpy-quaternion",
            "./pkg": None,
        }

        for spec, expected in cases.items():
            with self.subTest(spec=spec):
                self.assertEqual(install_custom_node._requirement_name(spec), expected)

    def test_load_requirements_follows_includes_and_collects_options(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir).resolve()
            nested = root / "sub"
            nested.mkdir()
            root.joinpath("requirements.txt").write_text(
                "--extra-index-url https://example.com/simple\n"
                "--extra-index-url https://example.com/simple\n"
                "-r sub/more.txt\n"
                "-r sub/more.txt\n"
                "-e .\n"
                "--trusted-host example.com\n"
                "--pre\n"
                "-c constraints.txt\n"
                "foo==1.0 --hash=sha256:abc\n"
                "-r missing.txt\n",
                encoding="utf-8",
            )
            nested.joinpath("more.txt").write_text(
                "-r ../requirements.txt\nbar\n--index-url=https://other.example/simple\n",
                encoding="utf-8",
            )

            plan = install_custom_node._load_requirements(root / "requirements.txt")

        self.assertEqual(
            [entry.line for entry in plan.entries], ["bar", "-e .", "foo==1.0"]
        )
        self.assertEqual(plan.entries[0].cwd, nested)
        self.assertEqual(plan.entries[1].cwd, root)
        self.assertEqual(plan.entries[1].args, ("-e", "."))
        self.assertIsNone(plan.entries[1].name)
        self.assertEqual(plan.entries[2].args, ("foo==1.0",))
        self.assertEqual(
            plan.options,
            [
                "--extra-index-url",
                "https://example.com/simple",
                "--index-url",
                "https://other.example/simple",
                "--prerelease=allow",
                "--constraint",
                str(root / "constraints.txt"),
            ],
        )
        self.assertEqual(
            plan.warnings,
            [
                "Ignoring unsupported requirements option: --trusted-host example.com",
                "Ignoring per-requirement options in: foo==1.0 --hash=sha256:abc",
                "Included requirements file not found: missing.txt",
            ],
        )


class InstallRequirementsTests(unittest.IsolatedAsyncioTestCase):
    async def install(
        self, recorder: ProcessRecorder, logs: list[str]
    ) -> install_custom_node.CustomNodeInstallResult:
        async def on_line(line: str) -> None:
            logs.append(line)

        with patched_environment(recorder):
            return await install_custom_node._install_custom_node(
                REPOSITORY_URL, on_line
            )

    async def test_mixed_file_skips_protected_packages_and_pins_versions(self) -> None:
        installed_packages = (
            '[{"name": "torch", "version": "2.7.0+cu128"},'
            ' {"name": "Pillow", "version": "11.0.0"},'
            ' {"name": "numpy", "version": "2.3.1"}]'
        )
        recorder = ProcessRecorder(
            "torch>=2.0\n"
            "opencv-python>=4.7\n"
            "NumPy>=1.26  # keep numpy pinned\n"
            "transformers[torch]>=4\n"
            "torchaudio\n"
            "torchsde\n"
            "Pillow\n",
            requirements_path="nested/requirements.txt",
            list_output=f"warning: cache is cold\n{installed_packages}\n",
        )
        logs: list[str] = []

        result = await self.install(recorder, logs)

        self.assertEqual(result.dependency_method, "requirements.txt")
        self.assertEqual(result.failed_requirements, ())

        self.assertEqual(
            [command[-1] for command in recorder.install_commands],
            ["opencv-python>=4.7", "torchsde", "Pillow"],
        )
        for command in recorder.install_commands:
            self.assertEqual(command[3], "-c")
            self.assertEqual(len(command), 6)
        self.assertEqual(
            recorder.constraints, ["numpy==2.3.1\ntorch==2.7.0+cu128\n"] * 3
        )

        requirements_dir = str(Path(recorder.commands[0][-1]) / "nested")
        list_index = recorder.commands.index(("uv", "pip", "list", "--format", "json"))
        self.assertEqual(recorder.cwds[list_index], requirements_dir)
        install_cwds = [
            cwd
            for command, cwd in zip(recorder.commands, recorder.cwds)
            if command[:3] == INSTALL
        ]
        self.assertEqual(install_cwds, [requirements_dir] * 3)

        self.assertEqual(logs[0], f"Cloning {REPOSITORY_URL}")
        self.assertEqual(
            logs[1],
            "Found requirements.txt at nested/requirements.txt (7 requirement(s))",
        )
        self.assertIn(
            "Protected packages (never installed or changed): "
            "numpy, torch, torchaudio, torchvision, transformers",
            logs,
        )
        self.assertIn("warning: cache is cold", logs)
        self.assertIn(
            "Pinned protected versions: numpy==2.3.1, torch==2.7.0+cu128", logs
        )
        self.assertIn(
            "Protected packages not installed: torchaudio, torchvision, transformers",
            logs,
        )
        self.assertIn(
            "[1/7] SKIPPED torch>=2.0 (protected, installed torch==2.7.0+cu128)", logs
        )
        self.assertIn(
            "[3/7] SKIPPED NumPy>=1.26 (protected, installed numpy==2.3.1)", logs
        )
        self.assertIn(
            "[4/7] SKIPPED transformers[torch]>=4 (protected, not installed)", logs
        )
        self.assertIn("[2/7] Installing opencv-python>=4.7", logs)
        self.assertIn("resolved and installed", logs)
        self.assertRegex(
            "\n".join(logs), r"\[2/7\] OK opencv-python>=4\.7 \(\d+\.\ds\)"
        )
        self.assertRegex(
            logs[-1],
            r"Dependencies finished in \d+\.\ds: "
            r"3 installed, 4 skipped \(protected\), 0 failed",
        )
        self.assertEqual(len(recorder.constraint_files), 3)
        self.assertFalse(recorder.constraint_files[0].exists())

    async def test_failed_line_does_not_stop_later_lines(self) -> None:
        recorder = ProcessRecorder("bad-pkg\ngood-pkg\n", failing=["bad-pkg"])
        logs: list[str] = []

        result = await self.install(recorder, logs)

        self.assertEqual(result.failed_requirements, ("bad-pkg",))
        self.assertEqual(result.dependency_method, "requirements.txt")
        self.assertEqual(
            [command[-1] for command in recorder.install_commands],
            ["bad-pkg", "good-pkg"],
        )
        self.assertRegex(
            "\n".join(logs),
            r"\[1/2\] FAILED bad-pkg: Command failed with exit code 1\. \(\d+\.\ds\)",
        )
        self.assertRegex("\n".join(logs), r"\[2/2\] OK good-pkg \(\d+\.\ds\)")
        self.assertEqual(logs[-1], "Failed requirement: bad-pkg")
        self.assertRegex(logs[-2], r"1 installed, 0 skipped \(protected\), 1 failed$")

    async def test_node_fails_when_every_installable_line_fails(self) -> None:
        recorder = ProcessRecorder(
            "torch\nbad-one\nbad-two\n", failing=["bad-one", "bad-two"]
        )

        with self.assertRaises(install_custom_node.CustomNodeInstallError) as context:
            await self.install(recorder, [])

        self.assertIn("bad-one, bad-two", str(context.exception))

    async def test_only_protected_lines_run_no_uv_commands(self) -> None:
        recorder = ProcessRecorder("torch>=2\nNumPy\ntransformers\n")
        logs: list[str] = []

        result = await self.install(recorder, logs)

        self.assertEqual(len(recorder.commands), 1)
        self.assertEqual(recorder.commands[0][:2], ("git", "clone"))
        self.assertEqual(result.dependency_method, "requirements.txt")
        self.assertEqual(result.failed_requirements, ())
        self.assertIn("Nothing to install from requirements.txt.", logs)
        self.assertIn("[1/3] SKIPPED torch>=2 (protected)", logs)

    async def test_node_fails_closed_when_installed_versions_are_unreadable(
        self,
    ) -> None:
        cases = {
            "uv exits with an error": ("uv: command failed\n", 1),
            "uv prints no JSON": ("warning: nothing to see\n", 0),
            "uv prints invalid JSON": ("[not json\n", 0),
        }

        for name, (output, return_code) in cases.items():
            with self.subTest(name):
                recorder = ProcessRecorder(
                    "opencv-python\n",
                    list_output=output,
                    list_return_code=return_code,
                )
                with self.assertRaises(
                    install_custom_node.CustomNodeInstallError
                ) as context:
                    await self.install(recorder, [])

                self.assertIn(
                    "Could not read installed package versions", str(context.exception)
                )
                self.assertEqual(recorder.install_commands, [])

    async def test_no_constraint_argument_when_nothing_is_pinned(self) -> None:
        recorder = ProcessRecorder("opencv-python\ntorch\n", list_output="[]\n")
        logs: list[str] = []

        await self.install(recorder, logs)

        self.assertEqual(
            recorder.install_commands, [("uv", "pip", "install", "opencv-python")]
        )
        self.assertEqual(recorder.constraints, [None])
        self.assertIn("Pinned protected versions: none installed", logs)
        self.assertIn("[2/2] SKIPPED torch (protected, not installed)", logs)

    async def test_requirement_options_are_appended_to_every_install(self) -> None:
        recorder = ProcessRecorder(
            "--extra-index-url https://example.com/simple\n--pre\npkg-a\npkg-b\n"
        )
        logs: list[str] = []

        await self.install(recorder, logs)

        options = (
            "--extra-index-url",
            "https://example.com/simple",
            "--prerelease=allow",
        )
        self.assertEqual(
            recorder.install_commands,
            [
                ("uv", "pip", "install", *options, "pkg-a"),
                ("uv", "pip", "install", *options, "pkg-b"),
            ],
        )
        self.assertIn(
            "Requirements options applied to every install: "
            "--extra-index-url https://example.com/simple --prerelease=allow",
            logs,
        )


class SummarizeInstallTests(unittest.TestCase):
    def test_installed_node_with_failed_requirements_makes_job_partial(self) -> None:
        result = install_custom_node.CustomNodeBatchInstallResult(
            results=(
                install_custom_node.CustomNodeBatchItemResult(
                    url="https://github.com/owner/first-node",
                    repository="first-node",
                    status="installed",
                    dependency_method="requirements.txt",
                    message="Installed first-node, but 1 requirement(s) failed: bad",
                    failed_requirements=("bad",),
                ),
                install_custom_node.CustomNodeBatchItemResult(
                    url="https://github.com/owner/second-node",
                    repository="second-node",
                    status="installed",
                    dependency_method="none",
                    message="Installed second-node.",
                ),
            ),
            restarted=False,
        )

        status, message = install_custom_node.summarize_install(result)

        self.assertEqual(status, "partial")
        self.assertIn("2 installed", message)
        self.assertIn("1 installed node(s) have failed requirements.", message)

    def test_job_without_failed_requirements_stays_completed(self) -> None:
        result = install_custom_node.CustomNodeBatchInstallResult(
            results=(
                install_custom_node.CustomNodeBatchItemResult(
                    url="https://github.com/owner/first-node",
                    repository="first-node",
                    status="installed",
                    dependency_method="none",
                    message="Installed first-node.",
                ),
            ),
            restarted=False,
        )

        status, message = install_custom_node.summarize_install(result)

        self.assertEqual(status, "completed")
        self.assertNotIn("failed requirements", message)

    def test_failed_requirements_message_lists_first_five(self) -> None:
        failed = tuple(f"pkg-{number}" for number in range(7))

        self.assertEqual(
            install_custom_node._summarize_requirements(failed),
            "pkg-0, pkg-1, pkg-2, pkg-3, pkg-4 and 2 more",
        )


class InstallCustomNodeTests(unittest.IsolatedAsyncioTestCase):
    async def test_runs_install_script_requirements_and_restarts(self) -> None:
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
            if command[:3] == ("uv", "pip", "list"):
                return FakeProcess("[]\n")
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
                patch.object(
                    install_custom_node,
                    "_is_comfyui_running",
                    new=AsyncMock(return_value=True),
                ),
            ):
                result = await install_custom_node.install_custom_node(
                    "https://github.com/owner/example-node.git"
                )

        self.assertEqual(result.repository, "example-node")
        self.assertEqual(result.dependency_method, "requirements.txt")
        self.assertEqual(commands[1][0], (sys.executable, "install.py"))
        self.assertEqual(commands[2][0], ("uv", "pip", "list", "--format", "json"))
        self.assertEqual(commands[3][0], ("uv", "pip", "install", "ignored"))
        self.assertEqual(len(commands), 4)
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
            if command[:3] == ("uv", "pip", "list"):
                return FakeProcess("[]\n")
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
                patch.object(
                    install_custom_node,
                    "_is_comfyui_running",
                    new=AsyncMock(return_value=True),
                ),
            ):
                result = await install_custom_node.install_custom_node(
                    "https://github.com/owner/requirements-node"
                )

        self.assertEqual(result.dependency_method, "requirements.txt")
        self.assertEqual(commands[1][:3], ("uv", "pip", "list"))
        self.assertEqual(commands[2][-1], "package")
        self.assertEqual(len(commands), 3)
        restart.assert_awaited_once()

    async def test_skips_restart_when_comfyui_is_not_running(self) -> None:
        async def create_process(*command, **kwargs):
            if command[:2] == ("git", "clone"):
                Path(command[-1]).mkdir()
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
                patch.object(
                    install_custom_node,
                    "_is_comfyui_running",
                    new=AsyncMock(return_value=False),
                ) as is_running,
            ):
                result = await install_custom_node.install_custom_node(
                    "https://github.com/owner/stopped-comfy-node"
                )

        self.assertEqual(result.repository, "stopped-comfy-node")
        is_running.assert_awaited_once_with()
        restart.assert_not_awaited()

    async def test_batch_continues_after_skipped_and_failed_nodes(self) -> None:
        async def create_process(*command, **kwargs):
            if command[:2] == ("git", "clone"):
                repository_path = Path(command[-1])
                if repository_path.name == "failing-node":
                    return FakeProcess("clone failed", return_code=1)
                repository_path.mkdir()
            return FakeProcess()

        with tempfile.TemporaryDirectory() as temp_dir:
            custom_nodes_path = Path(temp_dir) / "custom_nodes"
            custom_nodes_path.mkdir()
            custom_nodes_path.joinpath("existing-node").mkdir()

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
                patch.object(
                    install_custom_node,
                    "_is_comfyui_running",
                    new=AsyncMock(return_value=True),
                ) as is_running,
            ):
                result = await install_custom_node.install_custom_nodes(
                    [
                        "https://github.com/owner/first-node.git",
                        "https://github.com/owner/existing-node.git",
                        "https://github.com/owner/failing-node.git",
                        "https://github.com/owner/last-node.git",
                    ]
                )

        self.assertEqual(
            [item.status for item in result.results],
            ["installed", "already_installed", "failed", "installed"],
        )
        self.assertTrue(result.restarted)
        is_running.assert_awaited_once_with()
        restart.assert_awaited_once_with()

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


class CustomNodeJobTests(unittest.IsolatedAsyncioTestCase):
    def setUp(self) -> None:
        install_custom_node._jobs.clear()
        self.addCleanup(install_custom_node._jobs.clear)

        self.broadcast = AsyncMock()
        manager_patch = patch.object(
            install_custom_node,
            "manager",
            new=SimpleNamespace(broadcast=self.broadcast),
        )
        manager_patch.start()
        self.addCleanup(manager_patch.stop)

    def broadcast_messages(self, message_type: str) -> list[dict]:
        messages = [json.loads(call.args[0]) for call in self.broadcast.await_args_list]
        return [message for message in messages if message["type"] == message_type]

    async def test_job_records_command_output_and_completes(self) -> None:
        async def create_process(*command, **kwargs):
            if command[:2] == ("git", "clone"):
                Path(command[-1]).mkdir()
                return FakeProcess("line one\nprogress 10%\rprogress 100%\n")
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
                    "_is_comfyui_running",
                    new=AsyncMock(return_value=False),
                ),
            ):
                job = install_custom_node.create_install_job(
                    ["https://github.com/owner/logged-node.git"]
                )
                self.assertEqual(job.status, "queued")
                self.assertEqual(job.items[0].status, "queued")
                self.assertEqual(job.items[0].repository, "logged-node")

                await install_custom_node.run_install_job(job)

        item = job.items[0]
        self.assertEqual(job.status, "completed")
        self.assertFalse(job.restarted)
        self.assertEqual(item.status, "installed")
        self.assertEqual(item.dependency_method, "none")
        self.assertIn("line one", item.logs)
        self.assertIn("progress 100%", item.logs)
        self.assertNotIn("progress 10%", item.logs)

        log_messages = self.broadcast_messages("custom_node_log")
        self.assertIn("line one", [message["data"]["line"] for message in log_messages])
        self.assertEqual(log_messages[0]["data"]["job_id"], job.job_id)
        self.assertEqual(log_messages[0]["data"]["index"], 0)
        self.assertEqual(
            [m["data"]["status"] for m in self.broadcast_messages("custom_node")],
            ["queued", "installing", "installed"],
        )
        self.assertEqual(
            [m["data"]["status"] for m in self.broadcast_messages("custom_node_job")],
            ["queued", "running", "completed"],
        )

    async def test_unexpected_error_fails_only_that_item(self) -> None:
        with (
            patch.object(
                install_custom_node,
                "_install_custom_node",
                new=AsyncMock(
                    side_effect=[
                        RuntimeError("boom"),
                        install_custom_node.CustomNodeInstallResult(
                            repository="second-node", dependency_method="none"
                        ),
                    ]
                ),
            ),
            patch.object(
                install_custom_node,
                "_is_comfyui_running",
                new=AsyncMock(return_value=False),
            ),
        ):
            job = install_custom_node.create_install_job(
                [
                    "https://github.com/owner/first-node.git",
                    "https://github.com/owner/second-node.git",
                ]
            )
            await install_custom_node.run_install_job(job)

        self.assertEqual(job.items[0].status, "failed")
        self.assertEqual(job.items[0].message, "Unexpected error: boom")
        self.assertEqual(job.items[1].status, "installed")
        self.assertEqual(job.status, "partial")

    async def test_restart_failure_does_not_fail_the_job(self) -> None:
        async def create_process(*command, **kwargs):
            if command[:2] == ("git", "clone"):
                Path(command[-1]).mkdir()
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
                    "_is_comfyui_running",
                    new=AsyncMock(return_value=True),
                ),
                patch.object(
                    install_custom_node,
                    "restart_program",
                    new=AsyncMock(side_effect=RuntimeError("restart exploded")),
                ),
            ):
                job = install_custom_node.create_install_job(
                    ["https://github.com/owner/restart-node.git"]
                )
                await install_custom_node.run_install_job(job)

        self.assertEqual(job.status, "completed")
        self.assertFalse(job.restarted)
        self.assertEqual(job.items[0].status, "installed")
        self.assertIn("restart failed", job.message)
        self.assertIn("restart exploded", job.message)

    async def test_broadcast_failure_does_not_stop_the_job(self) -> None:
        self.broadcast.side_effect = RuntimeError("socket closed")

        async def create_process(*command, **kwargs):
            if command[:2] == ("git", "clone"):
                Path(command[-1]).mkdir()
            return FakeProcess("some output\n")

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
                    "_is_comfyui_running",
                    new=AsyncMock(return_value=False),
                ),
            ):
                job = install_custom_node.create_install_job(
                    ["https://github.com/owner/quiet-node.git"]
                )
                await install_custom_node.run_install_job(job)

        self.assertEqual(job.status, "completed")
        self.assertEqual(job.items[0].status, "installed")
        self.assertIn("some output", job.items[0].logs)

    async def test_start_install_job_runs_in_background(self) -> None:
        with (
            patch.object(
                install_custom_node,
                "_install_custom_node",
                new=AsyncMock(
                    return_value=install_custom_node.CustomNodeInstallResult(
                        repository="background-node", dependency_method="none"
                    )
                ),
            ),
            patch.object(
                install_custom_node,
                "_is_comfyui_running",
                new=AsyncMock(return_value=False),
            ),
        ):
            job = install_custom_node.start_install_job(
                ["https://github.com/owner/background-node.git"]
            )
            self.assertIs(install_custom_node.get_install_job(job.job_id), job)
            self.assertEqual(len(install_custom_node.active_install_tasks), 1)

            await asyncio.gather(*install_custom_node.active_install_tasks)

        self.assertTrue(job.finished)
        self.assertEqual(job.status, "completed")
        self.assertEqual(len(install_custom_node.active_install_tasks), 0)
        self.assertIsNone(install_custom_node.get_install_job("missing"))

    async def test_job_is_partial_when_a_requirement_failed(self) -> None:
        recorder = ProcessRecorder("bad-pkg\ngood-pkg\n", failing=["bad-pkg"])

        with patched_environment(recorder):
            job = install_custom_node.create_install_job([REPOSITORY_URL])
            await install_custom_node.run_install_job(job)

        item = job.items[0]
        self.assertEqual(item.status, "installed")
        self.assertEqual(
            item.message,
            "Installed example-node, but 1 requirement(s) failed: bad-pkg",
        )
        self.assertIn("Failed requirement: bad-pkg", item.logs)
        self.assertEqual(job.status, "partial")
        self.assertIn("1 installed node(s) have failed requirements.", job.message)

    async def test_prunes_oldest_finished_jobs_only(self) -> None:
        url = "https://github.com/owner/pruned-node.git"
        max_jobs = install_custom_node.MAX_JOBS

        first_job = install_custom_node.create_install_job([url])
        first_job.status = "completed"
        for _ in range(max_jobs + 5):
            job = install_custom_node.create_install_job([url])
            job.status = "completed"

        self.assertLessEqual(len(install_custom_node._jobs), max_jobs)
        self.assertNotIn(first_job.job_id, install_custom_node._jobs)
        self.assertIn(job.job_id, install_custom_node._jobs)

        install_custom_node._jobs.clear()
        unfinished = [
            install_custom_node.create_install_job([url]) for _ in range(max_jobs + 3)
        ]
        self.assertEqual(len(install_custom_node._jobs), max_jobs + 3)
        self.assertTrue(
            all(item.job_id in install_custom_node._jobs for item in unfinished)
        )


if __name__ == "__main__":
    unittest.main()
