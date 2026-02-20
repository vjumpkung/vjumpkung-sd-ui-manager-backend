import asyncio
from datetime import datetime
from typing import Any


def get_mili_timestamp():
    now = datetime.now()

    unix_timestamp_seconds = now.timestamp()

    unix_timestamp_milliseconds = int(unix_timestamp_seconds * 1000)

    return unix_timestamp_milliseconds


class DownloadHistory:
    def __init__(self):
        self._download_list: dict[str, dict[str, Any]] = {}
        self._lock = asyncio.Lock()

    async def put(self, downloadDto: dict[str, Any]) -> None:
        async with self._lock:
            self._download_list.setdefault(
                downloadDto["id"],
                {
                    "name": downloadDto["name"],
                    "url": downloadDto["url"],
                    "model_type": downloadDto["model_type"],
                    "status": downloadDto["status"],
                    "createdAt": get_mili_timestamp(),
                },
            )

    async def update_status(self, id: str, status: str) -> None:
        async with self._lock:
            if id in self._download_list:
                self._download_list[id]["status"] = status

    async def get(self) -> dict[str, dict[str, Any]]:
        async with self._lock:
            return dict(
                sorted(self._download_list.items(), key=lambda x: x[1]["createdAt"])
            )

    async def is_exists(self, id: str) -> bool:
        async with self._lock:
            return id in self._download_list

    async def get_by_id(self, id: str) -> dict[str, Any] | None:
        async with self._lock:
            return self._download_list.get(id)

    async def delete(self, downloadDto: dict[str, Any]) -> None:
        async with self._lock:
            if downloadDto["id"] in self._download_list:
                del self._download_list[downloadDto["id"]]


downloadHistory = DownloadHistory()
