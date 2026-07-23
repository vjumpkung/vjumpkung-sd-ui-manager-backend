import asyncio
import tempfile
import unittest
from pathlib import Path
from unittest.mock import AsyncMock, patch

from worker import download


class FakeCurlProcess:
    def __init__(self, return_code: int = 0):
        self.return_code = return_code
        self.stdout = asyncio.StreamReader()
        self.stdout.feed_data(b"200\n")
        self.stdout.feed_eof()

    async def wait(self) -> int:
        return self.return_code


class CivitaiAuthTests(unittest.IsolatedAsyncioTestCase):
    def test_civitai_token_is_returned_as_bearer_header(self) -> None:
        with patch.object(download.envs, "CIVITAI_TOKEN", "secret-token"):
            headers = download._get_civitai_headers()

        self.assertEqual(
            headers,
            {"Authorization": "Bearer secret-token"},
        )

    def test_empty_civitai_token_does_not_add_authorization_header(self) -> None:
        with patch.object(download.envs, "CIVITAI_TOKEN", ""):
            headers = download._get_civitai_headers()

        self.assertEqual(headers, {})

    async def test_civitai_filename_probe_uses_bearer_header(self) -> None:
        url = "https://civitai.com/api/download/models/123?type=Model"
        get_http_filename = AsyncMock(return_value="model.safetensors")

        with tempfile.TemporaryDirectory() as temp_dir:
            with (
                patch.object(download, "RESOURCE_PATH", temp_dir),
                patch.object(
                    download,
                    "_fetch_expected_sha256",
                    new=AsyncMock(return_value="a" * 64),
                ),
                patch.object(
                    download,
                    "_get_http_filename",
                    new=get_http_filename,
                ),
                patch.object(
                    download,
                    "_existing_file_matches_sha256",
                    new=AsyncMock(return_value=False),
                ),
                patch.object(download.envs, "get_environment_variable"),
                patch.object(download.envs, "CIVITAI_TOKEN", "secret-token"),
            ):
                await download.prepare_download("model", url, "checkpoints")

        get_http_filename.assert_awaited_once_with(
            url,
            {"Authorization": "Bearer secret-token"},
        )

    async def test_download_uses_get_content_disposition_filename(self) -> None:
        captured_command = None
        response_headers = (
            "HTTP/1.1 302 Found\r\n"
            'Content-Disposition: attachment; filename="redirect-name.bin"\r\n'
            "\r\n"
            "HTTP/1.1 200 OK\r\n"
            "Content-Disposition: attachment; "
            "filename*=UTF-8''final%20model.safetensors\r\n"
            "\r\n"
        )

        async def create_process(*command, **kwargs):
            nonlocal captured_command
            captured_command = command
            headers_path = command[command.index("-D") + 1]
            body_path = command[command.index("-o") + 1]
            Path(headers_path).write_text(response_headers, encoding="latin-1")
            Path(body_path).write_bytes(b"model")
            return FakeCurlProcess()

        url = "https://civitai.com/api/download/models/123?type=Model"
        with tempfile.TemporaryDirectory() as temp_dir:
            with (
                patch.object(
                    download,
                    "AsyncSession",
                    side_effect=OSError("HEAD unavailable"),
                ),
                patch.object(
                    download.asyncio,
                    "create_subprocess_exec",
                    side_effect=create_process,
                ),
            ):
                filename = await download._download_http(
                    url,
                    temp_dir,
                    filename="head-name.safetensors",
                    headers={"Authorization": "Bearer secret-token"},
                )

            self.assertEqual(filename, "final model.safetensors")
            self.assertEqual(
                Path(temp_dir, filename).read_bytes(),
                b"model",
            )
            self.assertFalse(Path(temp_dir, "head-name.safetensors").exists())
            self.assertEqual(
                sorted(path.name for path in Path(temp_dir).iterdir()),
                ["final model.safetensors"],
            )

        self.assertIsNotNone(captured_command)
        assert captured_command is not None
        self.assertIn("-D", captured_command)
        self.assertIn(
            "Authorization: Bearer secret-token",
            captured_command,
        )
        self.assertEqual(captured_command[-1], url)

    async def test_civitai_download_uses_bearer_header_and_unchanged_url(self) -> None:
        for hostname in ("civitai.com", "civitai.red"):
            with (
                self.subTest(hostname=hostname),
                tempfile.TemporaryDirectory() as temp_dir,
            ):
                url = f"https://{hostname}/api/download/models/123?type=Model"
                download_http = AsyncMock(return_value="model.safetensors")
                destination = download.os.path.join(temp_dir, "ckpts")

                with (
                    patch.object(download, "RESOURCE_PATH", temp_dir),
                    patch.object(download, "_download_http", new=download_http),
                    patch.object(
                        download.downloadHistory,
                        "update_status",
                        new=AsyncMock(),
                    ),
                    patch.object(download.manager, "broadcast", new=AsyncMock()),
                    patch.object(download.envs, "get_environment_variable"),
                    patch.object(download.envs, "CIVITAI_TOKEN", "secret-token"),
                ):
                    result = await download.download_async(
                        "download-id",
                        "model",
                        url,
                        "checkpoints",
                    )

                self.assertTrue(result)
                download_http.assert_awaited_once_with(
                    url,
                    destination,
                    filename=None,
                    headers={"Authorization": "Bearer secret-token"},
                    expected_sha256=None,
                )


if __name__ == "__main__":
    unittest.main()
