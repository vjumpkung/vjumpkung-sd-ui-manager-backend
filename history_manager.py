from datetime import datetime


def get_mili_timestamp():
    now = datetime.now()

    unix_timestamp_seconds = now.timestamp()

    unix_timestamp_milliseconds = int(unix_timestamp_seconds * 1000)

    return unix_timestamp_milliseconds


class DownloadHistory:
    _download_list = {}

    def put(self, downloadDto):
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

    def update_status(self, id, status):
        if id in self._download_list:
            self._download_list[id]["status"] = status

    def get(self):
        return dict(
            sorted(self._download_list.items(), key=lambda x: x[1]["createdAt"])
        )

    def is_exists(self, id):
        return id in self._download_list.keys()

    def get_by_id(self, id):
        return self._download_list[id]

    def delete(self, downloadDto):
        del self._download_list[downloadDto["id"]]


downloadHistory = DownloadHistory()
