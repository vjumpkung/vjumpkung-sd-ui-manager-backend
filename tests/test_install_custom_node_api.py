import unittest
from unittest.mock import patch

from fastapi import FastAPI, HTTPException
from fastapi.testclient import TestClient

import api
from worker import install_custom_node

INSTALLED_URL = "https://github.com/owner/installed-node.git"
FAILED_URL = "https://github.com/owner/failed-node.git"


class InstallCustomNodeApiTests(unittest.IsolatedAsyncioTestCase):
    def setUp(self) -> None:
        install_custom_node._jobs.clear()
        self.addCleanup(install_custom_node._jobs.clear)

    async def test_queues_job_for_every_custom_node(self) -> None:
        job = install_custom_node.create_install_job([INSTALLED_URL, FAILED_URL])

        with (
            patch.object(api, "UI_TYPE", "COMFY"),
            patch.object(api, "start_install_job", return_value=job) as start,
        ):
            response = await api.install_comfyui_custom_node(
                api.CustomNodeInstallRequest(urls=[INSTALLED_URL, FAILED_URL])
            )

        self.assertEqual(response.job_id, job.job_id)
        self.assertEqual(response.status, "queued")
        self.assertFalse(response.restarted)
        self.assertEqual([item.index for item in response.results], [0, 1])
        self.assertEqual(
            [item.status for item in response.results], ["queued", "queued"]
        )
        self.assertEqual(
            [item.url for item in response.results], [INSTALLED_URL, FAILED_URL]
        )
        start.assert_called_once_with([INSTALLED_URL, FAILED_URL])

    async def test_accepts_legacy_single_url_request(self) -> None:
        url = "https://github.com/owner/custom-node.git"
        job = install_custom_node.create_install_job([url])

        with (
            patch.object(api, "UI_TYPE", "COMFY"),
            patch.object(api, "start_install_job", return_value=job) as start,
        ):
            response = await api.install_comfyui_custom_node(
                api.CustomNodeInstallRequest(url=url)
            )

        self.assertEqual(response.job_id, job.job_id)
        start.assert_called_once_with([url])

    async def test_rejects_install_when_ui_is_not_comfy(self) -> None:
        with (
            patch.object(api, "UI_TYPE", "FORGE"),
            patch.object(api, "start_install_job") as start,
        ):
            with self.assertRaises(HTTPException) as context:
                await api.install_comfyui_custom_node(
                    api.CustomNodeInstallRequest(url=INSTALLED_URL)
                )

        self.assertEqual(context.exception.status_code, 400)
        start.assert_not_called()

    async def test_returns_job_snapshot_with_logs(self) -> None:
        job = install_custom_node.create_install_job([INSTALLED_URL])
        job.status = "running"
        job.items[0].status = "installing"
        job.items[0].logs.append("Cloning into example")

        response = await api.get_custom_node_install_job(job.job_id)

        self.assertEqual(response.job_id, job.job_id)
        self.assertEqual(response.status, "running")
        self.assertEqual(response.results[0].status, "installing")
        self.assertEqual(response.results[0].logs, ["Cloning into example"])

    async def test_unknown_job_returns_404(self) -> None:
        with self.assertRaises(HTTPException) as context:
            await api.get_custom_node_install_job("missing")

        self.assertEqual(context.exception.status_code, 404)

    async def test_lists_job_summaries_newest_first_without_logs(self) -> None:
        first_job = install_custom_node.create_install_job([INSTALLED_URL])
        second_job = install_custom_node.create_install_job([INSTALLED_URL, FAILED_URL])
        first_job.status = "completed"
        second_job.items[0].logs.append("Cloning into example")

        results = await api.list_custom_node_install_jobs()

        self.assertEqual(
            [job.job_id for job in results], [second_job.job_id, first_job.job_id]
        )
        self.assertEqual(
            [job.created_at for job in results],
            [second_job.created_at, first_job.created_at],
        )
        self.assertEqual(results[1].status, "completed")
        self.assertEqual(len(results[0].results), 2)
        self.assertNotIn("logs", results[0].results[0].model_dump())

    async def test_lists_no_jobs_when_none_exist(self) -> None:
        self.assertEqual(await api.list_custom_node_install_jobs(), [])


class InstallCustomNodeHttpTests(unittest.TestCase):
    def setUp(self) -> None:
        install_custom_node._jobs.clear()
        self.addCleanup(install_custom_node._jobs.clear)

        app = FastAPI()
        app.include_router(api.router)
        self.client = TestClient(app)

    def test_post_returns_202_with_job_id(self) -> None:
        job = install_custom_node.create_install_job([INSTALLED_URL])

        with (
            patch.object(api, "UI_TYPE", "COMFY"),
            patch.object(api, "start_install_job", return_value=job),
        ):
            response = self.client.post(
                "/api/install_custom_node", json={"urls": [INSTALLED_URL]}
            )

        self.assertEqual(response.status_code, 202)
        body = response.json()
        self.assertEqual(body["job_id"], job.job_id)
        self.assertEqual(body["status"], "queued")
        self.assertEqual(body["results"][0]["index"], 0)
        self.assertEqual(body["results"][0]["logs"], [])

    def test_get_returns_job_and_404_for_unknown_job(self) -> None:
        job = install_custom_node.create_install_job([INSTALLED_URL])

        found = self.client.get(f"/api/install_custom_node/{job.job_id}")
        missing = self.client.get("/api/install_custom_node/missing")

        self.assertEqual(found.status_code, 200)
        self.assertEqual(found.json()["job_id"], job.job_id)
        self.assertEqual(found.json()["created_at"], job.created_at)
        self.assertEqual(missing.status_code, 404)

    def test_get_list_returns_job_summaries_without_logs(self) -> None:
        job = install_custom_node.create_install_job([INSTALLED_URL])
        job.items[0].logs.append("Cloning into example")

        response = self.client.get("/api/install_custom_node")

        self.assertEqual(response.status_code, 200)
        body = response.json()
        self.assertEqual(len(body), 1)
        self.assertEqual(body[0]["job_id"], job.job_id)
        self.assertEqual(body[0]["created_at"], job.created_at)
        self.assertEqual(body[0]["results"][0]["url"], INSTALLED_URL)
        self.assertNotIn("logs", body[0]["results"][0])


if __name__ == "__main__":
    unittest.main()
