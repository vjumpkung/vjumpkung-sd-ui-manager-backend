import hashlib
import os
import tempfile
import unittest
import urllib.parse
from pathlib import Path
from unittest.mock import AsyncMock, patch

from worker import download


class FakeHelperProcess:
    def __init__(self, return_code: int = 0) -> None:
        self.return_code = return_code
        self.input: bytes | None = None

    @property
    def returncode(self) -> int:
        return self.return_code

    def communicate(self, input: bytes | None = None):
        self.input = input
        return b"", None

    def wait(self) -> int:
        return self.return_code

    def kill(self) -> None:
        self.return_code = -1


class FakeStreamResponse:
    def __init__(self, status_code: int, headers: dict[str, str]) -> None:
        self.status_code = status_code
        self.headers = headers

    async def __aenter__(self):
        return self

    async def __aexit__(self, exc_type, exc, traceback) -> None:
        pass


class FakeHttpClient:
    def __init__(self, response: FakeStreamResponse) -> None:
        self.response = response
        self.request: tuple | None = None

    async def __aenter__(self):
        return self

    async def __aexit__(self, exc_type, exc, traceback) -> None:
        pass

    def stream(self, method, url, **kwargs):
        self.request = (method, url, kwargs)
        return self.response


class CivitaiMetadataTests(unittest.IsolatedAsyncioTestCase):
    def test_civitai_token_is_returned_as_bearer_header(self) -> None:
        with patch.object(download.envs, "CIVITAI_TOKEN", "secret-token"):
            headers = download._get_civitai_headers()

        self.assertEqual(headers, {"Authorization": "Bearer secret-token"})

    def test_selects_file_matching_download_url_filters(self) -> None:
        target = download._select_civitai_file(
            "123",
            "https://civitai.com/api/download/models/123?type=Model&format=SafeTensor",
            [
                {
                    "id": 10,
                    "name": "unsafe.ckpt",
                    "primary": True,
                    "type": "Model",
                    "sizeKB": 1,
                    "metadata": {"format": "PickleTensor"},
                    "hashes": {"SHA256": "b" * 64},
                },
                {
                    "id": 11,
                    "name": "model.safetensors",
                    "primary": False,
                    "type": "Model",
                    "sizeKB": 2.5,
                    "metadata": {"format": "SafeTensor"},
                    "hashes": {"SHA256": "A" * 64},
                },
            ],
        )

        self.assertEqual(target.file_id, "11")
        self.assertEqual(target.filename, "model.safetensors")
        self.assertEqual(target.expected_size_bytes, 2560)
        self.assertEqual(target.expected_sha256, "a" * 64)

    def test_explicit_file_id_takes_priority_over_other_filters(self) -> None:
        target = download._select_civitai_file(
            "123",
            "https://civitai.com/api/download/models/123?fileId=10&format=SafeTensor",
            [
                {
                    "id": 10,
                    "name": "chosen.ckpt",
                    "primary": False,
                    "type": "Model",
                    "sizeKB": 1,
                    "metadata": {"format": "PickleTensor"},
                    "hashes": {"SHA256": "b" * 64},
                }
            ],
        )

        self.assertEqual(target.file_id, "10")
        self.assertEqual(target.filename, "chosen.ckpt")

    async def test_prepare_download_uses_exact_civitai_metadata(self) -> None:
        target = download.CivitaiDownloadTarget(
            version_id="123",
            file_id="11",
            filename="model.safetensors",
            expected_sha256="a" * 64,
            expected_size_bytes=2560,
        )
        url = "https://civitai.com/api/download/models/123?type=Model&format=SafeTensor"

        with (
            tempfile.TemporaryDirectory() as temp_dir,
            patch.object(download, "RESOURCE_PATH", temp_dir),
            patch.object(
                download,
                "_fetch_civitai_target",
                new=AsyncMock(return_value=target),
            ),
            patch.object(
                download,
                "_existing_file_matches_sha256",
                new=AsyncMock(return_value=False),
            ),
            patch.object(download.envs, "get_environment_variable"),
        ):
            preparation = await download.prepare_download("model", url, "checkpoints")

        self.assertEqual(preparation.cache_key, "a" * 64)
        self.assertEqual(preparation.filename, "model.safetensors")
        self.assertEqual(preparation.expected_size_bytes, 2560)
        self.assertEqual(preparation.civitai_file_id, "11")

    async def test_missing_token_is_rejected_before_queueing(self) -> None:
        with (
            patch.object(download.envs, "CIVITAI_TOKEN", ""),
            self.assertRaisesRegex(RuntimeError, "CIVITAI_TOKEN is required"),
        ):
            await download._fetch_civitai_target(
                "https://civitai.com/api/download/models/123"
            )


class CivitaiRedirectTests(unittest.IsolatedAsyncioTestCase):
    async def test_resolves_signed_url_with_get_without_following_redirect(
        self,
    ) -> None:
        signed_url = "https://storage.example/model?X-Amz-Signature=signed"
        client = FakeHttpClient(FakeStreamResponse(307, {"location": signed_url}))

        with (
            patch.object(download.envs, "CIVITAI_TOKEN", "secret-token"),
            patch.object(download.httpx, "AsyncClient", return_value=client),
        ):
            result = await download._resolve_civitai_download_url(
                "https://civitai.com/api/download/models/123?type=Model",
                "11",
            )

        self.assertEqual(result, signed_url)
        self.assertIsNotNone(client.request)
        assert client.request is not None
        method, requested_url, kwargs = client.request
        self.assertEqual(method, "GET")
        self.assertFalse(kwargs["follow_redirects"])
        query = urllib.parse.parse_qs(urllib.parse.urlparse(requested_url).query)
        self.assertEqual(query["fileId"], ["11"])
        self.assertEqual(query["token"], ["secret-token"])

    async def test_rejects_redirect_that_leaks_civitai_token(self) -> None:
        client = FakeHttpClient(
            FakeStreamResponse(
                302,
                {"location": "https://storage.example/model?token=secret-token"},
            )
        )

        with (
            patch.object(download.envs, "CIVITAI_TOKEN", "secret-token"),
            patch.object(download.httpx, "AsyncClient", return_value=client),
            self.assertRaisesRegex(RuntimeError, "token was exposed"),
        ):
            await download._resolve_civitai_download_url(
                "https://civitai.com/api/download/models/123", "11"
            )


class CivitaiAriaTests(unittest.IsolatedAsyncioTestCase):
    async def test_completed_staging_file_is_recovered_without_network(self) -> None:
        content = b"complete model"
        expected_sha256 = hashlib.sha256(content).hexdigest()

        with tempfile.TemporaryDirectory() as temp_dir:
            staging = Path(temp_dir, f".civitai-11-{expected_sha256[:12]}.part")
            staging.write_bytes(content)
            resolve_url = AsyncMock()

            with patch.object(
                download, "_resolve_civitai_download_url", new=resolve_url
            ):
                filename = await download._download_civitai(
                    "https://civitai.com/api/download/models/123",
                    temp_dir,
                    filename="model.safetensors",
                    expected_sha256=expected_sha256,
                    expected_size_bytes=len(content),
                    file_id="11",
                )

            self.assertEqual(filename, "model.safetensors")
            self.assertEqual(Path(temp_dir, filename).read_bytes(), content)
            self.assertFalse(staging.exists())
            resolve_url.assert_not_awaited()

    async def test_helper_receives_signed_url_only_through_stdin(self) -> None:
        content = b"model"
        expected_sha256 = hashlib.sha256(content).hexdigest()
        signed_url = "https://storage.example/model?X-Amz-Signature=secret"
        captured_command = None
        captured_env = None
        process = None

        def create_process(command, **kwargs):
            nonlocal captured_command, captured_env, process
            captured_command = command
            captured_env = kwargs["env"]
            destination = command[command.index("--destination") + 1]
            staging_filename = command[command.index("--output") + 1]
            Path(destination, staging_filename).write_bytes(content)
            process = FakeHelperProcess()
            return process

        with tempfile.TemporaryDirectory() as temp_dir:
            with (
                patch.object(
                    download,
                    "_resolve_civitai_download_url",
                    new=AsyncMock(return_value=signed_url),
                ),
                patch.object(
                    download.subprocess,
                    "Popen",
                    side_effect=create_process,
                ),
                patch.object(
                    download.asyncio,
                    "create_subprocess_exec",
                    side_effect=AssertionError("async subprocess must not be used"),
                ),
                patch.dict(
                    os.environ,
                    {"CIVITAI_TOKEN": "secret-token", "KEEP_ME": "yes"},
                    clear=True,
                ),
            ):
                filename = await download._download_civitai(
                    "https://civitai.com/api/download/models/123",
                    temp_dir,
                    filename="model.safetensors",
                    expected_sha256=expected_sha256,
                    expected_size_bytes=len(content),
                    file_id="11",
                )

            self.assertEqual(filename, "model.safetensors")
            self.assertEqual(
                Path(temp_dir, filename).read_bytes(),
                content,
            )

        self.assertIsNotNone(captured_command)
        assert captured_command is not None
        self.assertEqual(captured_command[0], download.PYTHON)
        self.assertEqual(captured_command[1], download.CIVITAI_DOWNLOAD_SCRIPT)
        self.assertNotIn(signed_url, captured_command)
        self.assertIsNotNone(process)
        assert process is not None
        self.assertEqual(process.input, f"{signed_url}\n".encode())
        self.assertIsNotNone(captured_env)
        assert captured_env is not None
        self.assertNotIn("CIVITAI_TOKEN", captured_env)
        self.assertEqual(captured_env["KEEP_ME"], "yes")

    async def test_download_async_routes_all_civitai_hosts_to_aria(self) -> None:
        for hostname in ("civitai.com", "civitai.green", "civitai.red"):
            with self.subTest(hostname=hostname):
                url = f"https://{hostname}/api/download/models/123"
                download_civitai = AsyncMock(return_value="model.safetensors")

                with tempfile.TemporaryDirectory() as temp_dir:
                    destination = os.path.join(temp_dir, "ckpts")
                    with (
                        patch.object(download, "RESOURCE_PATH", temp_dir),
                        patch.object(
                            download, "_download_civitai", new=download_civitai
                        ),
                        patch.object(
                            download.downloadHistory,
                            "update_status",
                            new=AsyncMock(),
                        ),
                        patch.object(download.manager, "broadcast", new=AsyncMock()),
                        patch.object(download.envs, "get_environment_variable"),
                    ):
                        result = await download.download_async(
                            "download-id", "model", url, "checkpoints"
                        )

                self.assertTrue(result)
                download_civitai.assert_awaited_once_with(
                    url,
                    destination,
                    filename=None,
                    expected_sha256=None,
                    expected_size_bytes=None,
                    file_id=None,
                )


if __name__ == "__main__":
    unittest.main()
