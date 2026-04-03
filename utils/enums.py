from enum import Enum


class DownloadStatus(str, Enum):
    IN_QUEUE = "IN_QUEUE"
    DOWNLOADING = "DOWNLOADING"
    COMPLETED = "COMPLETED"
    FAILED = "FAILED"
    RETRYING = "RETRYING"
