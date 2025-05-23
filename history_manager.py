class DownloadHistory:
    _download_list = {}

    def put(self, downloadDto):
        self._download_list[downloadDto["id"]] = {
            "name": downloadDto["name"],
            "url": downloadDto["url"],
            "model_type": downloadDto["model_type"],
            "status": downloadDto["status"],
        }

    def update_status(self, id, status):
        if id in self._download_list:
            self._download_list[id]["status"] = status

    def get(self):
        return self._download_list

    def is_exists(self, id):
        return id in self._download_list.keys()

    def get_by_id(self, id):
        return self._download_list[id]

    def delete(self, downloadDto):
        del self._download_list[downloadDto["id"]]


downloadHistory = DownloadHistory()
