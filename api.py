import asyncio
import hashlib
import json
import os
import tempfile
from datetime import datetime
from pathlib import Path
from typing import Annotated, List, Literal, Optional

import aiofiles
from fastapi import APIRouter, BackgroundTasks, HTTPException
from fastapi.responses import JSONResponse, StreamingResponse
from pydantic import BaseModel, Field, HttpUrl, model_validator

from config.load_config import OUTPUT_PATH, RUNPOD_POD_ID, UI_TYPE
from env_manager import envs
from history_manager import downloadHistory
from utils.ws_messages import CustomNodeItemStatus, CustomNodeJobStatus
from worker.check_process import programStatus
from worker.download import (
    TokenRequiredError,
    _redact_url,
    broadcast_token_required,
    download_multiple,
    queue_download,
)
from worker.export_zip import _create_zip_file
from worker.install_custom_node import (
    CustomNodeJob,
    get_install_job,
    list_install_jobs,
    start_install_job,
)
from worker.program_logs import programLog
from worker.restart_program import restart_program

if UI_TYPE == "ZIMAGE":
    pass
else:
    import torch

    print("import torch completed")


class ModelDownloadRequest(BaseModel):
    model_config = {"protected_namespaces": ()}

    name: Optional[str]
    url: HttpUrl
    model_type: str


class UpdateEnvRequest(BaseModel):
    value: str


class DownloadSelectedDto(BaseModel):
    name: str
    url: HttpUrl


class ImportModel(BaseModel):
    name: str
    url: HttpUrl
    type: str


class CustomNodeInstallRequest(BaseModel):
    urls: Annotated[list[HttpUrl] | None, Field(min_length=1, max_length=100)] = None
    url: HttpUrl | None = None

    @model_validator(mode="after")
    def validate_repository_urls(self) -> "CustomNodeInstallRequest":
        if self.urls is None and self.url is None:
            raise ValueError("Provide at least one custom node URL.")
        if self.urls is not None and self.url is not None:
            raise ValueError("Provide either 'urls' or 'url', not both.")
        return self

    def repository_urls(self) -> list[str]:
        urls = self.urls if self.urls is not None else [self.url]
        return [str(url) for url in urls if url is not None]


class CustomNodeInstallItemSummaryResponse(BaseModel):
    index: int
    url: str
    repository: str | None
    status: CustomNodeItemStatus
    dependency_method: Literal["install.py", "requirements.txt", "none"] | None
    message: str


class CustomNodeInstallItemResponse(CustomNodeInstallItemSummaryResponse):
    logs: list[str] = Field(default_factory=list)


class CustomNodeInstallJobResponse(BaseModel):
    job_id: str
    status: CustomNodeJobStatus
    message: str
    restarted: bool
    created_at: int
    results: list[CustomNodeInstallItemResponse]


class CustomNodeInstallJobSummaryResponse(BaseModel):
    job_id: str
    status: CustomNodeJobStatus
    message: str
    restarted: bool
    created_at: int
    results: list[CustomNodeInstallItemSummaryResponse]


def _job_summary_response(job: CustomNodeJob) -> CustomNodeInstallJobSummaryResponse:
    return CustomNodeInstallJobSummaryResponse(
        job_id=job.job_id,
        status=job.status,
        message=job.message,
        restarted=job.restarted,
        created_at=job.created_at,
        results=[
            CustomNodeInstallItemSummaryResponse(
                index=item.index,
                url=item.url,
                repository=item.repository,
                status=item.status,
                dependency_method=item.dependency_method,
                message=item.message,
            )
            for item in job.items
        ],
    )


def _job_response(job: CustomNodeJob) -> CustomNodeInstallJobResponse:
    return CustomNodeInstallJobResponse(
        job_id=job.job_id,
        status=job.status,
        message=job.message,
        restarted=job.restarted,
        created_at=job.created_at,
        results=[
            CustomNodeInstallItemResponse(
                index=item.index,
                url=item.url,
                repository=item.repository,
                status=item.status,
                dependency_method=item.dependency_method,
                message=item.message,
                logs=list(item.logs),
            )
            for item in job.items
        ],
    )


router = APIRouter(prefix="/api")


@router.get("/checkcuda")
async def checkcuda():
    if UI_TYPE == "ZIMAGE":
        return JSONResponse(
            {
                "cuda": "skipped",
                "gpu_name": "skipped",
                "pytorch_version": "skipped",
                "runpod_id": RUNPOD_POD_ID,
                "status": programStatus.get_status(),
                "ui": UI_TYPE,
            }
        )

    is_cuda_available = torch.cuda.is_available()
    if not is_cuda_available:
        return JSONResponse(
            {
                "cuda": is_cuda_available,
                "gpu_name": "",
                "pytorch_version": torch.__version__,
                "runpod_id": RUNPOD_POD_ID,
                "status": "NOT_RUNNING",
                "ui": UI_TYPE,
            }
        )
    current_device = torch.cuda.current_device()
    gpu_name = torch.cuda.get_device_name(current_device)
    return JSONResponse(
        {
            "cuda": is_cuda_available,
            "gpu_name": gpu_name,
            "pytorch_version": torch.__version__,
            "runpod_id": RUNPOD_POD_ID,
            "status": programStatus.get_status(),
            "ui": UI_TYPE,
        }
    )


@router.get("/download_history")
async def getDownloadHistory():
    return await downloadHistory.get()


@router.get("/get_model_packs")
async def getModelPacks():
    if UI_TYPE == "ZIMAGE":
        return JSONResponse([])

    target = f"./resources/{UI_TYPE.lower()}_model_packs.json"

    async with aiofiles.open(target) as fp:
        model_packs = json.loads(await fp.read())

    return JSONResponse(model_packs)


@router.put("/update_env/{api_key_type}", status_code=204)
async def update_api_key(
    api_key_type: Literal["civitai", "huggingface"], request: UpdateEnvRequest
):
    if api_key_type == "civitai":
        envs.CIVITAI_TOKEN = request.value
    elif api_key_type == "huggingface":
        envs.set_huggingface_token(request.value)


@router.post("/download_selected")
async def download_selected(
    request: List[DownloadSelectedDto], background_tasks: BackgroundTasks
):
    try:
        task = asyncio.create_task(
            download_multiple(list(map(lambda x: dict(x), request)))
        )
        background_tasks.add_task(lambda: task)
        return JSONResponse(
            {
                "status": "received",
                "message": "Download request received.",
            }
        )
    except Exception as e:
        raise HTTPException(
            status_code=500, detail=f"Error processing request: {str(e)}"
        )


@router.post("/import_models")
async def import_models(request: List[ImportModel]):
    try:
        for t in request:
            try:
                await queue_download(t.name, str(t.url), t.type)
            except TokenRequiredError as e:
                await broadcast_token_required(e, t.name, str(t.url), t.type)

        return JSONResponse(
            {
                "status": "received",
                "message": "Import Models request received.",
            }
        )

    except Exception as e:
        raise HTTPException(
            status_code=500, detail=f"Error processing request: {str(e)}"
        )


@router.post("/download_custom_model")
async def download_custom_model(
    request: ModelDownloadRequest,
):
    model_name = request.model_type if request.name in ("", None) else request.name
    try:
        result = await queue_download(model_name, str(request.url), request.model_type)

        if result.action == "retrying":
            return JSONResponse(
                {
                    "status": "retrying",
                    "message": "Download request is retrying.",
                }
            )

        if result.action == "duplicate":
            return JSONResponse(
                {
                    "status": "duplicated",
                    "message": "An equivalent download is already queued or complete.",
                }
            )

        if result.action == "already_downloaded":
            return JSONResponse(
                {
                    "status": "already_downloaded",
                    "message": "A local file already matches the expected SHA256.",
                }
            )

        return JSONResponse(
            {
                "status": "received",
                "message": "Download request received.",
            }
        )

    except TokenRequiredError as e:
        raise HTTPException(
            status_code=422,
            detail={
                "code": "TOKEN_REQUIRED",
                "provider": e.provider,
                "reason": e.reason,
                "message": str(e),
                "name": model_name,
                "url": _redact_url(str(request.url)),
                "model_type": request.model_type,
            },
        )
    except Exception as e:
        raise HTTPException(
            status_code=500, detail=f"Error processing request: {str(e)}"
        )


@router.get("/logs")
def get_program_log():
    return programLog.get()


@router.post("/restart", status_code=204)
async def restart():
    await restart_program()


@router.post(
    "/install_custom_node",
    status_code=202,
    response_model=CustomNodeInstallJobResponse,
)
async def install_comfyui_custom_node(
    request: CustomNodeInstallRequest,
) -> CustomNodeInstallJobResponse:
    if UI_TYPE != "COMFY":
        raise HTTPException(
            status_code=400,
            detail="Custom nodes can only be installed when UI_TYPE is COMFY.",
        )

    return _job_response(start_install_job(request.repository_urls()))


@router.get(
    "/install_custom_node",
    response_model=list[CustomNodeInstallJobSummaryResponse],
)
async def list_custom_node_install_jobs() -> list[CustomNodeInstallJobSummaryResponse]:
    return [_job_summary_response(job) for job in list_install_jobs()]


@router.get(
    "/install_custom_node/{job_id}",
    response_model=CustomNodeInstallJobResponse,
)
async def get_custom_node_install_job(job_id: str) -> CustomNodeInstallJobResponse:
    job = get_install_job(job_id)
    if job is None:
        raise HTTPException(
            status_code=404, detail="Custom node installation job not found."
        )

    return _job_response(job)


@router.get("/download-images")
async def download_images_zip():
    """
    Creates a zip file of all images in ./output_images folder (recursive)
    and returns it as a streaming download with Content-Disposition header.
    """
    output_dir = Path(OUTPUT_PATH)
    # Check if the directory exists (using async path operations)
    if not await asyncio.to_thread(output_dir.exists) or not await asyncio.to_thread(
        output_dir.is_dir
    ):
        raise HTTPException(status_code=404, detail="Output images directory not found")

    # Create a temporary file for the zip
    temp_zip = tempfile.NamedTemporaryFile(delete=False, suffix=".zip")
    temp_zip_path = temp_zip.name
    temp_zip.close()

    try:
        # Create zip file asynchronously
        await _create_zip_file(output_dir, temp_zip_path)

        # Check if zip file was created and has content (non-blocking)
        file_exists = await asyncio.to_thread(os.path.exists, temp_zip_path)
        file_size = (
            await asyncio.to_thread(os.path.getsize, temp_zip_path)
            if file_exists
            else 0
        )

        if not file_exists or file_size == 0:
            raise HTTPException(status_code=404, detail="No files found to zip")

        # Define async generator to stream file content
        async def file_streamer():
            try:
                async with aiofiles.open(temp_zip_path, "rb") as file:
                    while chunk := await file.read(8192):  # 8KB chunks
                        yield chunk
            finally:
                # Clean up temp file after streaming
                try:
                    if await asyncio.to_thread(os.path.exists, temp_zip_path):
                        await asyncio.to_thread(os.unlink, temp_zip_path)
                except:
                    pass  # Ignore cleanup errors

        # Generate filename with YYYY-MM-DD and short hash
        date_str = datetime.now().strftime("%Y-%m-%d")
        short_hash = hashlib.sha256(
            str(datetime.now().timestamp()).encode()
        ).hexdigest()[:8]
        filename = f"output_images_{date_str}_{short_hash}.zip"

        # Return streaming response with Content-Disposition header
        return StreamingResponse(
            file_streamer(),
            media_type="application/zip",
            headers={
                "Content-Disposition": f"attachment; filename={filename}",
                "Content-Length": str(file_size),
            },
        )

    except Exception as e:
        # Clean up temp file if something goes wrong (non-blocking)
        try:
            if await asyncio.to_thread(os.path.exists, temp_zip_path):
                await asyncio.to_thread(os.unlink, temp_zip_path)
        except:
            pass  # Ignore cleanup errors
        raise HTTPException(
            status_code=500, detail=f"Error creating zip file: {str(e)}"
        )
