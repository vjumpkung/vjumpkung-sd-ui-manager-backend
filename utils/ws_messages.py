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


CustomNodeItemStatus = Literal[
    "queued", "installing", "installed", "already_installed", "failed"
]
CustomNodeJobStatus = Literal["queued", "running", "completed", "partial", "failed"]
CustomNodeDependencyMethod = Literal["install.py", "requirements.txt", "none"]


class CustomNodeData(BaseModel):
    job_id: str
    index: int
    url: str
    repository: str | None
    status: CustomNodeItemStatus
    dependency_method: CustomNodeDependencyMethod | None = None
    message: str


class CustomNodeMessage(BaseModel):
    type: Literal["custom_node"] = "custom_node"
    data: CustomNodeData


class CustomNodeLogData(BaseModel):
    job_id: str
    index: int
    url: str
    repository: str | None
    line: str


class CustomNodeLogMessage(BaseModel):
    type: Literal["custom_node_log"] = "custom_node_log"
    data: CustomNodeLogData


class CustomNodeJobData(BaseModel):
    job_id: str
    status: CustomNodeJobStatus
    message: str
    restarted: bool


class CustomNodeJobMessage(BaseModel):
    type: Literal["custom_node_job"] = "custom_node_job"
    data: CustomNodeJobData


class LogData(BaseModel):
    m: str


class LogMessage(BaseModel):
    key: str
    type: Literal["logs"] = "logs"
    data: LogData
