import asyncio
import io
import json
import os
import unittest
from unittest.mock import AsyncMock, Mock, call, patch

import httpx
from fastapi import HTTPException

import api
from utils.enums import DownloadStatus
from worker import download


class FakeResponse:
    def __init__(self, status_code: int, headers=None, payload=None) -> None:
        self.status_code = status_code
        self.headers = headers or {}
        self.payload = payload

    async def __aenter__(self):
        return self

    async def __aexit__(self, exc_type, exc, traceback) -> None:
        pass

    def json(self):
        return self.payload


class FakeHttpClient:
    def __init__(self, *responses) -> None:
        self.get = AsyncMock(side_effect=responses)
        self.head = AsyncMock(side_effect=responses)
        self.stream = Mock(return_value=responses[0] if responses else None)

    async def __aenter__(self):
        return self

    async def __aexit__(self, exc_type, exc, traceback) -> None:
        pass


class FakeProcess:
    def __init__(self, return_code: int = 1) -> None:
        self.return_code = return_code
        self.stdout = io.BytesIO()

    def wait(self) -> int:
        return self.return_code


class CivitaiTokenRequiredTests(unittest.IsolatedAsyncioTestCase):
    async def test_missing_metadata_token_keeps_existing_message(self) -> None:
        with (
            patch.object(download.envs, "CIVITAI_TOKEN", ""),
            patch.object(download.httpx, "AsyncClient") as client,
            self.assertRaises(download.TokenRequiredError) as caught,
        ):
            await download._fetch_civitai_target(
                "https://civitai.com/api/download/models/123"
            )

        self.assertIsInstance(caught.exception, RuntimeError)
        self.assertEqual(caught.exception.provider, "civitai")
        self.assertEqual(caught.exception.reason, "missing")
        self.assertEqual(
            str(caught.exception),
            "CIVITAI_TOKEN is required; configure it before downloading CivitAI models",
        )
        client.assert_not_called()

    async def test_metadata_token_rejection_keeps_existing_message(self) -> None:
        for status_code in (401, 403):
            with self.subTest(status_code=status_code):
                client = FakeHttpClient(FakeResponse(status_code))
                with (
                    patch.object(download.envs, "CIVITAI_TOKEN", "secret-token"),
                    patch.object(download.httpx, "AsyncClient", return_value=client),
                    self.assertRaises(download.TokenRequiredError) as caught,
                ):
                    await download._fetch_civitai_target(
                        "https://civitai.com/api/download/models/123"
                    )

                self.assertEqual(caught.exception.provider, "civitai")
                self.assertEqual(caught.exception.reason, "rejected")
                self.assertEqual(
                    str(caught.exception),
                    "CivitAI API token is invalid or lacks access to this model",
                )

    async def test_missing_download_token_keeps_existing_message(self) -> None:
        with (
            patch.object(download.envs, "CIVITAI_TOKEN", ""),
            self.assertRaises(download.TokenRequiredError) as caught,
        ):
            await download._resolve_civitai_download_url(
                "https://civitai.com/api/download/models/123", "11"
            )

        self.assertEqual(caught.exception.provider, "civitai")
        self.assertEqual(caught.exception.reason, "missing")
        self.assertEqual(
            str(caught.exception),
            "CIVITAI_TOKEN is required; configure it before downloading CivitAI models",
        )

    async def test_signed_url_token_rejection_keeps_existing_message(self) -> None:
        for status_code in (401, 403):
            with self.subTest(status_code=status_code):
                client = FakeHttpClient(FakeResponse(status_code))
                with (
                    patch.object(download.envs, "CIVITAI_TOKEN", "secret-token"),
                    patch.object(download.httpx, "AsyncClient", return_value=client),
                    self.assertRaises(download.TokenRequiredError) as caught,
                ):
                    await download._resolve_civitai_download_url(
                        "https://civitai.com/api/download/models/123", "11"
                    )

                self.assertEqual(caught.exception.provider, "civitai")
                self.assertEqual(caught.exception.reason, "rejected")
                self.assertEqual(
                    str(caught.exception),
                    "CivitAI API token is invalid or lacks access to this model",
                )


class HuggingFaceAccessTests(unittest.IsolatedAsyncioTestCase):
    async def test_401_without_token_requires_missing_token(self) -> None:
        url = "https://huggingface.co/owner/repo/resolve/main/model.safetensors"
        client = FakeHttpClient(FakeResponse(401))
        with (
            patch.object(download.envs, "HUGGINGFACE_TOKEN", ""),
            patch.object(download.httpx, "AsyncClient", return_value=client),
            self.assertRaises(download.TokenRequiredError) as caught,
        ):
            await download._check_huggingface_access(url)

        self.assertEqual(caught.exception.provider, "huggingface")
        self.assertEqual(caught.exception.reason, "missing")
        self.assertEqual(
            str(caught.exception),
            "Hugging Face access token may be required; the model may be gated, "
            "private, or the URL may be wrong",
        )
        client.head.assert_awaited_once_with(
            url, headers={}, follow_redirects=False, timeout=15
        )

    async def test_403_with_token_sends_bearer_header_and_requires_token(self) -> None:
        url = "https://huggingface.co/owner/repo/resolve/main/model.safetensors"
        client = FakeHttpClient(FakeResponse(403))
        with (
            patch.object(download.envs, "HUGGINGFACE_TOKEN", "secret-token"),
            patch.object(download.httpx, "AsyncClient", return_value=client),
            self.assertRaises(download.TokenRequiredError) as caught,
        ):
            await download._check_huggingface_access(url)

        self.assertEqual(caught.exception.provider, "huggingface")
        self.assertEqual(caught.exception.reason, "rejected")
        self.assertEqual(
            str(caught.exception),
            "Hugging Face denied access with the current token; the token may be invalid "
            "or access to the gated model has not been granted",
        )
        client.head.assert_awaited_once_with(
            url,
            headers={"Authorization": "Bearer secret-token"},
            follow_redirects=False,
            timeout=15,
        )

    async def test_external_or_insecure_redirect_is_not_followed(self) -> None:
        url = "https://huggingface.co/owner/repo/resolve/main/model.safetensors"
        for location in (
            "https://cdn.example/model.safetensors",
            "http://huggingface.co/owner/repo/resolve/main/model.safetensors",
            "https://huggingface.co.example/model.safetensors",
        ):
            with self.subTest(location=location):
                client = FakeHttpClient(FakeResponse(302, {"location": location}))
                with (
                    patch.object(download.envs, "HUGGINGFACE_TOKEN", "secret-token"),
                    patch.object(download.httpx, "AsyncClient", return_value=client),
                ):
                    await download._check_huggingface_access(url)

                client.head.assert_awaited_once_with(
                    url,
                    headers={"Authorization": "Bearer secret-token"},
                    follow_redirects=False,
                    timeout=15,
                )

    async def test_on_hub_redirect_then_401_requires_token(self) -> None:
        url = "https://huggingface.co/owner/repo/resolve/main/model.safetensors"
        location = "/owner/repo/resolve/revision/model.safetensors"
        client = FakeHttpClient(
            FakeResponse(307, {"location": location}), FakeResponse(401)
        )
        with (
            patch.object(download.envs, "HUGGINGFACE_TOKEN", ""),
            patch.object(download.httpx, "AsyncClient", return_value=client),
            self.assertRaises(download.TokenRequiredError) as caught,
        ):
            await download._check_huggingface_access(url)

        self.assertEqual(caught.exception.reason, "missing")
        self.assertEqual(
            client.head.await_args_list,
            [
                call(url, headers={}, follow_redirects=False, timeout=15),
                call(
                    f"https://huggingface.co{location}",
                    headers={},
                    follow_redirects=False,
                    timeout=15,
                ),
            ],
        )

    async def test_follows_at_most_three_redirects(self) -> None:
        url = "https://huggingface.co/owner/repo/resolve/main/model.safetensors"
        client = FakeHttpClient(
            *(FakeResponse(307, {"location": url}) for _ in range(4))
        )
        with (
            patch.object(download.envs, "HUGGINGFACE_TOKEN", ""),
            patch.object(download.httpx, "AsyncClient", return_value=client),
        ):
            await download._check_huggingface_access(url)

        self.assertEqual(client.head.await_count, 4)

    async def test_network_error_does_not_require_token_or_log_secrets(self) -> None:
        client = FakeHttpClient(httpx.ConnectError("contains secret-token"))
        with (
            patch.object(download.envs, "HUGGINGFACE_TOKEN", "secret-token"),
            patch.object(download.httpx, "AsyncClient", return_value=client),
            patch.object(download.log, "warning") as warning,
        ):
            await download._check_huggingface_access(
                "https://huggingface.co/owner/repo/resolve/main/model.safetensors"
            )

        warning.assert_called_once()
        self.assertNotIn("secret-token", str(warning.call_args))

    async def test_builds_resolve_url_for_blob_dataset_and_space_urls(self) -> None:
        for path, resolve_path in (
            (
                "owner/repo/blob/refs%2Fpr%2F1/weights/model%20name.safetensors?token=secret",
                "owner/repo/resolve/refs%2Fpr%2F1/weights/model%20name.safetensors",
            ),
            (
                "datasets/owner/repo/blob/main/data/train.parquet",
                "datasets/owner/repo/resolve/main/data/train.parquet",
            ),
            (
                "spaces/owner/repo/resolve/main/model.safetensors",
                "spaces/owner/repo/resolve/main/model.safetensors",
            ),
        ):
            with self.subTest(path=path):
                client = FakeHttpClient(FakeResponse(200))
                with (
                    patch.object(download.envs, "HUGGINGFACE_TOKEN", ""),
                    patch.object(download.httpx, "AsyncClient", return_value=client),
                ):
                    await download._check_huggingface_access(
                        f"https://huggingface.co/{path}"
                    )

                client.head.assert_awaited_once_with(
                    f"https://huggingface.co/{resolve_path}",
                    headers={},
                    follow_redirects=False,
                    timeout=15,
                )

    async def test_unparsable_url_makes_no_request(self) -> None:
        with patch.object(download.httpx, "AsyncClient") as client:
            await download._check_huggingface_access(
                "https://huggingface.co/owner/repo"
            )

        client.assert_not_called()

    async def test_other_statuses_do_not_require_token(self) -> None:
        for status_code in (200, 404, 500):
            with self.subTest(status_code=status_code):
                client = FakeHttpClient(FakeResponse(status_code))
                with (
                    patch.object(download.envs, "HUGGINGFACE_TOKEN", ""),
                    patch.object(download.httpx, "AsyncClient", return_value=client),
                ):
                    await download._check_huggingface_access(
                        "https://huggingface.co/owner/repo/resolve/main/model.safetensors"
                    )


class TokenDownloadTests(unittest.IsolatedAsyncioTestCase):
    async def test_prepare_download_preserves_cached_cli_login(self) -> None:
        url = "https://huggingface.co/owner/repo/resolve/main/model.safetensors"
        for token, executable, should_probe in (
            ("", "/usr/bin/hf", False),
            ("secret-token", "/usr/bin/hf", True),
            ("", None, True),
        ):
            with self.subTest(token=bool(token), executable=executable):
                with (
                    patch.object(download.os, "makedirs"),
                    patch.object(download.envs, "get_environment_variable"),
                    patch.object(download.envs, "HUGGINGFACE_TOKEN", token),
                    patch.object(download.shutil, "which", return_value=executable),
                    patch.object(
                        download, "_check_huggingface_access", new=AsyncMock()
                    ) as check,
                    patch.object(
                        download,
                        "_fetch_expected_sha256",
                        new=AsyncMock(return_value=None),
                    ),
                    patch.object(
                        download,
                        "_existing_file_matches_sha256",
                        new=AsyncMock(return_value=False),
                    ),
                ):
                    await download.prepare_download("model", url, "vae")

                if should_probe:
                    check.assert_awaited_once_with(url)
                else:
                    check.assert_not_awaited()

    async def test_hf_nonzero_exit_probes_and_only_broadcasts_auth_denial(self) -> None:
        url = "https://huggingface.co/owner/repo/resolve/main/model.safetensors"
        for error in (
            None,
            download.TokenRequiredError("huggingface", "rejected", "Access denied"),
        ):
            with self.subTest(auth_denied=error is not None):
                with (
                    patch.object(download.os, "makedirs"),
                    patch.object(download.envs, "get_environment_variable"),
                    patch.object(download.envs, "HUGGINGFACE_TOKEN", ""),
                    patch.object(download.shutil, "which", return_value=None),
                    patch.object(
                        download.subprocess,
                        "Popen",
                        new=Mock(return_value=FakeProcess()),
                    ) as create_process,
                    patch.object(
                        download.downloadHistory, "update_status", new=AsyncMock()
                    ) as update_status,
                    patch.object(
                        download.manager, "broadcast", new=AsyncMock()
                    ) as broadcast,
                    patch.object(
                        download,
                        "_check_huggingface_access",
                        new=AsyncMock(side_effect=error),
                    ) as check,
                ):
                    result = await download.download_async(
                        "download-id", "model", url, "vae"
                    )

                self.assertFalse(result)
                create_process.assert_called_once()
                check.assert_awaited_once_with(url)
                update_status.assert_has_awaits(
                    [
                        call("download-id", DownloadStatus.DOWNLOADING),
                        call("download-id", DownloadStatus.FAILED),
                    ]
                )
                messages = [
                    json.loads(args.args[0]) for args in broadcast.await_args_list
                ]
                self.assertEqual(messages[1]["data"]["status"], "FAILED")
                self.assertEqual(
                    [message["type"] for message in messages],
                    ["download", "download", "token_required"]
                    if error
                    else ["download", "download"],
                )
                if error:
                    self.assertEqual(messages[2]["data"]["provider"], "huggingface")
                    self.assertEqual(messages[2]["data"]["reason"], "rejected")

    async def test_late_civitai_token_rejection_broadcasts_after_failure(self) -> None:
        url = "https://civitai.com/api/download/models/123"
        error = download.TokenRequiredError("civitai", "rejected", "Access denied")
        with (
            patch.object(download.os, "makedirs"),
            patch.object(download.envs, "get_environment_variable"),
            patch.object(
                download, "_download_civitai", new=AsyncMock(side_effect=error)
            ),
            patch.object(download.downloadHistory, "update_status", new=AsyncMock()),
            patch.object(download.manager, "broadcast", new=AsyncMock()) as broadcast,
            patch.object(download.traceback, "print_exception"),
        ):
            result = await download.download_async("download-id", "model", url, "vae")

        self.assertFalse(result)
        messages = [json.loads(args.args[0]) for args in broadcast.await_args_list]
        self.assertEqual(messages[1]["data"]["status"], "FAILED")
        self.assertEqual(messages[2]["type"], "token_required")
        self.assertEqual(messages[2]["data"]["provider"], "civitai")

    async def test_model_pack_continues_after_token_required(self) -> None:
        models = [
            {
                "name": "restricted",
                "url": "https://civitai.com/api/download/models/123",
                "type": "vae",
            },
            {
                "name": "public",
                "url": "https://example.com/model.safetensors",
                "type": "vae",
            },
        ]
        client = FakeHttpClient(FakeResponse(200, payload=models))
        task = asyncio.create_task(AsyncMock(return_value=True)())
        with (
            patch.object(download.httpx, "AsyncClient", return_value=client),
            patch.object(
                download,
                "queue_download",
                new=AsyncMock(
                    side_effect=[
                        download.TokenRequiredError(
                            "civitai", "missing", "Token required"
                        ),
                        download.QueueDownloadResult(action="queued", task=task),
                    ]
                ),
            ) as queue,
            patch.object(download.manager, "broadcast", new=AsyncMock()) as broadcast,
        ):
            await download.download_multiple(
                [{"name": "pack", "url": "https://example.com/models.json"}]
            )

        self.assertEqual(
            queue.await_args_list,
            [
                call(model["name"], model["url"], model["type"], from_model_pack=True)
                for model in models
            ],
        )
        self.assertTrue(task.done())
        broadcast.assert_awaited_once()
        message = json.loads(broadcast.await_args.args[0])
        self.assertEqual(message["type"], "token_required")
        self.assertEqual(message["data"]["name"], "restricted")


class TokenBroadcastTests(unittest.IsolatedAsyncioTestCase):
    async def test_broadcast_redacts_userinfo_and_token_query_parameters(self) -> None:
        url = (
            "https://user:password@huggingface.co:443/owner/repo/resolve/main/model"
            "?download=true&token=secret&ToKeN=other&%74oken=encoded&empty=&keep=a%20b#file"
        )
        with patch.object(download.manager, "broadcast", new=AsyncMock()) as broadcast:
            await download.broadcast_token_required(
                download.TokenRequiredError("huggingface", "missing", "Token required"),
                "model",
                url,
                "vae",
            )

        broadcast.assert_awaited_once()
        self.assertEqual(
            json.loads(broadcast.await_args.args[0]),
            {
                "type": "token_required",
                "data": {
                    "provider": "huggingface",
                    "reason": "missing",
                    "name": "model",
                    "url": "https://huggingface.co:443/owner/repo/resolve/main/model"
                    "?download=true&empty=&keep=a%20b#file",
                    "model_type": "vae",
                },
            },
        )

    async def test_broken_websocket_does_not_propagate_or_log_secrets(self) -> None:
        with (
            patch.object(
                download.manager,
                "broadcast",
                new=AsyncMock(side_effect=RuntimeError("contains secret-token")),
            ),
            patch.object(download.log, "warning") as warning,
        ):
            await download.broadcast_token_required(
                download.TokenRequiredError("civitai", "missing", "Token required"),
                "model",
                "https://civitai.com/api/download/models/123?token=secret-token",
                "vae",
            )

        warning.assert_called_once()
        self.assertNotIn("secret-token", str(warning.call_args))


class TokenRequiredApiTests(unittest.IsolatedAsyncioTestCase):
    async def test_custom_download_returns_structured_422_for_token_errors(
        self,
    ) -> None:
        for provider, url in (
            ("civitai", "https://civitai.com/api/download/models/123"),
            ("huggingface", "https://huggingface.co/owner/repo/resolve/main/model"),
        ):
            for reason in ("missing", "rejected"):
                for name in ("model", "", None):
                    with self.subTest(provider=provider, reason=reason, name=name):
                        request = api.ModelDownloadRequest(
                            name=name,
                            url=url.replace("https://", "https://user:password@")
                            + "?token=secret&download=true",
                            model_type="vae",
                        )
                        error = download.TokenRequiredError(
                            provider, reason, "Token required"
                        )
                        with (
                            patch.object(
                                api, "queue_download", new=AsyncMock(side_effect=error)
                            ),
                            self.assertRaises(HTTPException) as caught,
                        ):
                            await api.download_custom_model(request)

                        self.assertEqual(caught.exception.status_code, 422)
                        self.assertEqual(
                            caught.exception.detail,
                            {
                                "code": "TOKEN_REQUIRED",
                                "provider": provider,
                                "reason": reason,
                                "message": "Token required",
                                "name": name or "vae",
                                "url": url + "?download=true",
                                "model_type": "vae",
                            },
                        )

    async def test_import_continues_after_token_required(self) -> None:
        models = [
            api.ImportModel(
                name="restricted",
                url="https://civitai.com/api/download/models/123",
                type="vae",
            ),
            api.ImportModel(
                name="public", url="https://example.com/model.safetensors", type="vae"
            ),
        ]
        error = download.TokenRequiredError("civitai", "missing", "Token required")
        with (
            patch.object(
                api,
                "queue_download",
                new=AsyncMock(
                    side_effect=[error, download.QueueDownloadResult(action="queued")]
                ),
            ) as queue,
            patch.object(api, "broadcast_token_required", new=AsyncMock()) as broadcast,
        ):
            response = await api.import_models(models)

        self.assertEqual(response.status_code, 200)
        self.assertEqual(
            json.loads(response.body),
            {"status": "received", "message": "Import Models request received."},
        )
        self.assertEqual(
            queue.await_args_list,
            [call(model.name, str(model.url), model.type) for model in models],
        )
        broadcast.assert_awaited_once_with(
            error, "restricted", str(models[0].url), "vae"
        )

    async def test_update_tokens_from_json_body(self) -> None:
        with (
            patch.object(api.envs, "CIVITAI_TOKEN", ""),
            patch.object(api.envs, "HUGGINGFACE_TOKEN", ""),
            patch.dict(os.environ, {}, clear=True),
        ):
            result = await api.update_api_key(
                "civitai", api.UpdateEnvRequest(value="civitai-secret")
            )
            self.assertIsNone(result)
            self.assertEqual(api.envs.CIVITAI_TOKEN, "civitai-secret")
            result = await api.update_api_key(
                "huggingface", api.UpdateEnvRequest(value="hf-secret")
            )
            self.assertIsNone(result)
            self.assertEqual(api.envs.HUGGINGFACE_TOKEN, "hf-secret")
            self.assertEqual(os.environ["HUGGINGFACE_TOKEN"], "hf-secret")
            self.assertEqual(os.environ["HF_TOKEN"], "hf-secret")


if __name__ == "__main__":
    unittest.main()
