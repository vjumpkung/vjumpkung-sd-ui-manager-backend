import unittest
from unittest.mock import AsyncMock, patch

import api
from worker.install_custom_node import (
    CustomNodeBatchInstallResult,
    CustomNodeBatchItemResult,
)


class InstallCustomNodeApiTests(unittest.IsolatedAsyncioTestCase):
    async def test_returns_results_for_every_custom_node(self) -> None:
        batch_result = CustomNodeBatchInstallResult(
            results=(
                CustomNodeBatchItemResult(
                    url="https://github.com/owner/installed-node.git",
                    repository="installed-node",
                    status="installed",
                    dependency_method="requirements.txt",
                    message="Installed installed-node.",
                ),
                CustomNodeBatchItemResult(
                    url="https://github.com/owner/failed-node.git",
                    repository="failed-node",
                    status="failed",
                    dependency_method=None,
                    message="Clone failed.",
                ),
            ),
            restarted=True,
        )

        with (
            patch.object(api, "UI_TYPE", "COMFY"),
            patch.object(
                api,
                "install_custom_nodes",
                new=AsyncMock(return_value=batch_result),
            ) as install,
        ):
            response = await api.install_comfyui_custom_node(
                api.CustomNodeInstallRequest(
                    urls=[
                        "https://github.com/owner/installed-node.git",
                        "https://github.com/owner/failed-node.git",
                    ]
                )
            )

        self.assertEqual(response.status, "partial")
        self.assertTrue(response.restarted)
        self.assertEqual(
            [result.status for result in response.results],
            ["installed", "failed"],
        )
        install.assert_awaited_once_with(
            [
                "https://github.com/owner/installed-node.git",
                "https://github.com/owner/failed-node.git",
            ]
        )

    async def test_accepts_legacy_single_url_request(self) -> None:
        batch_result = CustomNodeBatchInstallResult(results=(), restarted=False)

        with (
            patch.object(api, "UI_TYPE", "COMFY"),
            patch.object(
                api,
                "install_custom_nodes",
                new=AsyncMock(return_value=batch_result),
            ) as install,
        ):
            await api.install_comfyui_custom_node(
                api.CustomNodeInstallRequest(
                    url="https://github.com/owner/custom-node.git"
                )
            )

        install.assert_awaited_once_with(["https://github.com/owner/custom-node.git"])


if __name__ == "__main__":
    unittest.main()
