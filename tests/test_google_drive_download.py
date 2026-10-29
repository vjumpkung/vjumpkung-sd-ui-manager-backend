import io
import subprocess
import tempfile
import unittest
from unittest.mock import AsyncMock, Mock, call, patch

from utils.enums import DownloadStatus
from worker import download


class FakeProcess:
    def __init__(self, return_code: int = 0):
        self.return_code = return_code
        self.stdout = io.BytesIO()

    def wait(self) -> int:
        return self.return_code

    def kill(self) -> None:
        self.return_code = -1


class GoogleDriveDownloadTests(unittest.IsolatedAsyncioTestCase):
    async def test_google_drive_download_completes(self) -> None:
        url = "https://drive.google.com/file/d/abc123/view?usp=sharing"
        captured_command = None
        captured_kwargs = None

        def create_process(command, **kwargs):
            nonlocal captured_command, captured_kwargs
            captured_command = command
            captured_kwargs = kwargs
            return FakeProcess(0)

        with tempfile.TemporaryDirectory() as temp_dir:
            with (
                patch.object(download, "RESOURCE_PATH", temp_dir),
                patch.object(download.subprocess, "Popen", side_effect=create_process),
                patch.object(
                    download.asyncio,
                    "create_subprocess_exec",
                    new=Mock(side_effect=NotImplementedError()),
                ) as asyncio_create_process,
                patch.object(
                    download.downloadHistory, "update_status", new=AsyncMock()
                ) as update_status,
                patch.object(download.manager, "broadcast", new=AsyncMock()),
                patch.object(download.envs, "get_environment_variable"),
            ):
                expected_destination = download._get_download_destination("vae")[1]
                result = await download.download_async(
                    "download-id", "gdrive-model", url, "vae"
                )

        self.assertTrue(result)
        self.assertEqual(
            captured_command,
            [
                download.PYTHON,
                "./scripts/google_drive_download.py",
                "--path",
                expected_destination,
                "--url",
                url,
            ],
        )
        self.assertIsNotNone(captured_kwargs)
        assert captured_kwargs is not None
        self.assertIs(captured_kwargs["stdout"], subprocess.PIPE)
        self.assertIs(captured_kwargs["stderr"], subprocess.STDOUT)
        self.assertEqual(
            update_status.await_args_list[-1],
            call("download-id", DownloadStatus.COMPLETED),
        )
        asyncio_create_process.assert_not_called()

    async def test_google_drive_nonzero_exit_fails(self) -> None:
        url = "https://drive.google.com/file/d/abc123/view?usp=sharing"

        with tempfile.TemporaryDirectory() as temp_dir:
            with (
                patch.object(download, "RESOURCE_PATH", temp_dir),
                patch.object(
                    download.subprocess, "Popen", return_value=FakeProcess(1)
                ),
                patch.object(
                    download.asyncio,
                    "create_subprocess_exec",
                    new=Mock(side_effect=NotImplementedError()),
                ) as asyncio_create_process,
                patch.object(
                    download.downloadHistory, "update_status", new=AsyncMock()
                ) as update_status,
                patch.object(download.manager, "broadcast", new=AsyncMock()),
                patch.object(download.envs, "get_environment_variable"),
            ):
                result = await download.download_async(
                    "download-id", "gdrive-model", url, "vae"
                )

        self.assertFalse(result)
        self.assertEqual(
            update_status.await_args_list[-1],
            call("download-id", DownloadStatus.FAILED),
        )
        asyncio_create_process.assert_not_called()

    async def test_google_drive_launch_failure_fails(self) -> None:
        url = "https://drive.google.com/file/d/abc123/view?usp=sharing"

        with tempfile.TemporaryDirectory() as temp_dir:
            with (
                patch.object(download, "RESOURCE_PATH", temp_dir),
                patch.object(
                    download.subprocess, "Popen", side_effect=OSError("boom")
                ),
                patch.object(
                    download.asyncio,
                    "create_subprocess_exec",
                    new=Mock(side_effect=NotImplementedError()),
                ) as asyncio_create_process,
                patch.object(
                    download.downloadHistory, "update_status", new=AsyncMock()
                ) as update_status,
                patch.object(download.manager, "broadcast", new=AsyncMock()),
                patch.object(download.envs, "get_environment_variable"),
            ):
                result = await download.download_async(
                    "download-id", "gdrive-model", url, "vae"
                )

        self.assertFalse(result)
        self.assertEqual(
            update_status.await_args_list[-1],
            call("download-id", DownloadStatus.FAILED),
        )
        asyncio_create_process.assert_not_called()


if __name__ == "__main__":
    unittest.main()
