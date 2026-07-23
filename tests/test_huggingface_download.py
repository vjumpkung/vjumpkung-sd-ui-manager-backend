import asyncio
import os
import tempfile
import unittest
from pathlib import Path
from unittest.mock import AsyncMock, patch

from env_manager import Envs
from worker import download


class FakeProcess:
    def __init__(self, return_code: int = 0):
        self.return_code = return_code
        self.stdout = asyncio.StreamReader()
        self.stdout.feed_eof()

    async def wait(self) -> int:
        return self.return_code

    def kill(self) -> None:
        self.return_code = -1


class EnvsTests(unittest.TestCase):
    def test_huggingface_token_is_mapped_to_hf_token(self) -> None:
        with patch.dict(os.environ, {"HUGGINGFACE_TOKEN": "legacy-token"}, clear=True):
            test_envs = Envs()

            test_envs.get_environment_variable()

            self.assertEqual(test_envs.HUGGINGFACE_TOKEN, "legacy-token")
            self.assertEqual(os.environ["HF_TOKEN"], "legacy-token")

    def test_hf_token_is_also_accepted_by_backend(self) -> None:
        with patch.dict(os.environ, {"HF_TOKEN": "standard-token"}, clear=True):
            test_envs = Envs()

            test_envs.get_environment_variable()

            self.assertEqual(test_envs.HUGGINGFACE_TOKEN, "standard-token")
            self.assertEqual(os.environ["HUGGINGFACE_TOKEN"], "standard-token")


class HuggingFaceUrlTests(unittest.TestCase):
    def test_parse_model_file_url(self) -> None:
        target = download._parse_huggingface_download_url(
            "https://huggingface.co/owner/repo/resolve/main/subdir/model.safetensors"
        )

        self.assertIsNotNone(target)
        assert target is not None
        self.assertEqual(target.repo_id, "owner/repo")
        self.assertEqual(target.repo_type, "model")
        self.assertEqual(target.revision, "main")
        self.assertEqual(target.filepath, "subdir/model.safetensors")

    def test_parse_dataset_url_and_encoded_revision(self) -> None:
        target = download._parse_huggingface_download_url(
            "https://huggingface.co/datasets/owner/repo/resolve/"
            "refs%2Fpr%2F1/data/train.parquet?download=true"
        )

        self.assertIsNotNone(target)
        assert target is not None
        self.assertEqual(target.repo_type, "dataset")
        self.assertEqual(target.revision, "refs/pr/1")
        self.assertEqual(target.filepath, "data/train.parquet")

        command = download._build_huggingface_cli_command("hf", target, "/tmp/download")
        self.assertEqual(
            command[0:4], ["hf", "download", "owner/repo", target.filepath]
        )
        self.assertIn("--repo-type", command)
        self.assertEqual(command[-1], "dataset")

    def test_reject_non_file_url(self) -> None:
        target = download._parse_huggingface_download_url(
            "https://huggingface.co/owner/repo"
        )

        self.assertIsNone(target)

    def test_reject_path_traversal(self) -> None:
        target = download._parse_huggingface_download_url(
            "https://huggingface.co/owner/repo/resolve/main/"
            "models%2F..%2Fsecret.safetensors"
        )

        self.assertIsNone(target)


class HuggingFaceDownloadTests(unittest.IsolatedAsyncioTestCase):
    async def test_hf_cli_is_preferred_and_receives_mapped_token(self) -> None:
        captured_command = None
        captured_env = None

        async def create_process(*command, **kwargs):
            nonlocal captured_command, captured_env
            captured_command = command
            captured_env = kwargs["env"]
            local_dir = command[command.index("--local-dir") + 1]
            remote_file = command[3]
            source = Path(local_dir).joinpath(*remote_file.split("/"))
            source.parent.mkdir(parents=True, exist_ok=True)
            source.write_bytes(b"model")
            return FakeProcess()

        with tempfile.TemporaryDirectory() as temp_dir:
            with (
                patch.object(download, "RESOURCE_PATH", temp_dir),
                patch.object(download.shutil, "which", return_value="/usr/bin/hf"),
                patch.object(
                    download.asyncio,
                    "create_subprocess_exec",
                    side_effect=create_process,
                ),
                patch.object(
                    download.downloadHistory,
                    "update_status",
                    new=AsyncMock(),
                ),
                patch.object(download.manager, "broadcast", new=AsyncMock()),
                patch.object(download.envs, "get_environment_variable"),
                patch.object(download.envs, "HUGGINGFACE_TOKEN", "secret-token"),
            ):
                result = await download.download_async(
                    "download-id",
                    "renamed-model",
                    "https://huggingface.co/owner/repo/resolve/main/"
                    "weights/model.safetensors",
                    "vae",
                )

            self.assertTrue(result)
            self.assertIsNotNone(captured_command)
            assert captured_command is not None
            self.assertEqual(captured_command[0], "/usr/bin/hf")
            self.assertEqual(captured_command[1], "download")
            self.assertNotIn("aria2c", captured_command)
            self.assertIsNotNone(captured_env)
            assert captured_env is not None
            self.assertEqual(captured_env["HF_TOKEN"], "secret-token")
            self.assertEqual(
                Path(temp_dir, "vae", "renamed-model.safetensors").read_bytes(),
                b"model",
            )
            self.assertFalse(Path(temp_dir, "vae", ".hf-download").exists())

    async def test_aria2c_is_used_when_hf_cli_is_missing(self) -> None:
        captured_command = None

        async def create_process(*command, **kwargs):
            nonlocal captured_command
            captured_command = command
            return FakeProcess()

        with tempfile.TemporaryDirectory() as temp_dir:
            with (
                patch.object(download, "RESOURCE_PATH", temp_dir),
                patch.object(download.shutil, "which", return_value=None),
                patch.object(
                    download.asyncio,
                    "create_subprocess_exec",
                    side_effect=create_process,
                ),
                patch.object(
                    download.downloadHistory,
                    "update_status",
                    new=AsyncMock(),
                ),
                patch.object(download.manager, "broadcast", new=AsyncMock()),
                patch.object(download.envs, "get_environment_variable"),
                patch.object(download.envs, "HUGGINGFACE_TOKEN", "secret-token"),
            ):
                result = await download.download_async(
                    "download-id",
                    "renamed-model",
                    "https://huggingface.co/owner/repo/resolve/main/model.safetensors",
                    "vae",
                )

        self.assertTrue(result)
        self.assertIsNotNone(captured_command)
        assert captured_command is not None
        self.assertEqual(captured_command[0], "aria2c")
        self.assertIn("--out=renamed-model.safetensors", captured_command)
        self.assertIn("--header=Authorization: Bearer secret-token", captured_command)


if __name__ == "__main__":
    unittest.main()
