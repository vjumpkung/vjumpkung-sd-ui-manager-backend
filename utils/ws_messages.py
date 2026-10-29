from typing import Literal

from pydantic import BaseModel

from utils.enums import DownloadStatus


class DownloadData(BaseModel):
    model_config = {"protected_namespaces": ()}

    id: str
    name: str
    url: str
    model_type: str
    status: DownloadStatus
    sha256: str | None = None


class DownloadMessage(BaseModel):
    type: Literal["download"] = "download"
    data: DownloadData


class TokenRequiredData(BaseModel):
    model_config = {"protected_namespaces": ()}

    provider: Literal["civitai", "huggingface"]
    reason: Literal["missing", "rejected"]
    name: str
    url: str
    model_type: str


class TokenRequiredMessage(BaseModel):
    type: Literal["token_required"] = "token_required"
    data: TokenRequiredData


class MonitorData(BaseModel):
    status: str


class MonitorMessage(BaseModel):
    type: Literal["monitor"] = "monitor"
    data: MonitorData


class LogData(BaseModel):
    m: str


class LogMessage(BaseModel):
    key: str
    type: Literal["logs"] = "logs"
    data: LogData
