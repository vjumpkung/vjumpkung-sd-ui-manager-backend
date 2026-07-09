import asyncio
from datetime import datetime
from typing import Any

from utils.enums import DownloadStatus


def get_mili_timestamp():
    now = datetime.now()

    unix_timestamp_seconds = now.timestamp()

    unix_timestamp_milliseconds = int(unix_timestamp_seconds * 1000)

    return unix_timestamp_milliseconds


class DownloadHistory:
    def __init__(self):
        self._download_list: dict[str, dict[str, Any]] = {}
        self._lock = asyncio.Lock()

    async def put(self, downloadDto: dict[str, Any]) -> bool:
        async with self._lock:
            cache_key = downloadDto.get("sha256") or downloadDto["id"]
            if cache_key in self._download_list:
                return False

            self._download_list[cache_key] = {
                "name": downloadDto["name"],
                "url": downloadDto["url"],
                "model_type": downloadDto["model_type"],
                "status": downloadDto["status"],
                "sha256": downloadDto.get("sha256"),
                "createdAt": get_mili_timestamp(),
            }
            return True

    async def update_status(self, cache_key: str, status: DownloadStatus) -> None:
        async with self._lock:
            if cache_key in self._download_list:
                self._download_list[cache_key]["status"] = status

    async def update_status_if_current(
        self,
        cache_key: str,
        expected_status: DownloadStatus,
        status: DownloadStatus,
    ) -> bool:
        async with self._lock:
            download = self._download_list.get(cache_key)
            if not download or download["status"] != expected_status:
                return False

            download["status"] = status
            return True

    async def get(self) -> dict[str, dict[str, Any]]:
        async with self._lock:
            return dict(
                sorted(self._download_list.items(), key=lambda x: x[1]["createdAt"])
            )

    async def is_exists(self, cache_key: str) -> bool:
        async with self._lock:
            return cache_key in self._download_list

    async def get_by_id(self, cache_key: str) -> dict[str, Any] | None:
        async with self._lock:
            return self._download_list.get(cache_key)

    async def delete(self, downloadDto: dict[str, Any]) -> None:
        async with self._lock:
            cache_key = downloadDto.get("sha256") or downloadDto["id"]
            if cache_key in self._download_list:
                del self._download_list[cache_key]


downloadHistory = DownloadHistory()
