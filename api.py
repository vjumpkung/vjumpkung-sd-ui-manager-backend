import asyncio
import json
import os
import tempfile
from pathlib import Path
from typing import List, Literal

import aiofiles
import torch
from fastapi import APIRouter, BackgroundTasks, HTTPException, Request
from fastapi.responses import FileResponse, JSONResponse
from pydantic import BaseModel, HttpUrl

from config.load_config import RUNPOD_POD_ID, UI_TYPE, OUTPUT_PATH
from env_manager import envs
from event_handler import manager
from history_manager import downloadHistory
from worker.check_process import programStatus
from worker.download import download_async, download_multiple
from worker.export_zip import _create_zip_file
from worker.program_logs import programLog
from worker.restart_program import restart_program


class ModelDownloadRequest(BaseModel):
    url: HttpUrl
    model_type: Literal[
        "checkpoints",
        "clip",
        "clip_vision",
        "controlnet",
        "diffusion_models",
        "embeddings",
        "esrgan",
        "gfpgan",
        "gligen",
        "hypernetwork",
        "hypernetworks",
        "ipadapter",
        "loras",
        "text-encoder",
        "text_encoders",
        "unet",
        "upscale_models",
        "vae",
    ]


class DownloadSelectedDto(BaseModel):
    name: str
    url: HttpUrl


router = APIRouter(prefix="/api")


@router.get("/checkcuda")
async def checkcuda():
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
    return downloadHistory.get()


@router.get("/get_model_packs")
async def getModelPacks():

    target = f"./resources/{UI_TYPE.lower()}_model_packs.json"

    async with aiofiles.open(target) as fp:
        model_packs = json.loads(await fp.read())

    return JSONResponse(model_packs)


@router.put("/update_env/{api_key_type}", status_code=204)
async def update_api_key(
    request: Request, api_key_type: Literal["civitai", "huggingface"], value: str
):
    if api_key_type == "civitai":
        envs.CIVITAI_TOKEN = value
    elif api_key_type == "huggingface":
        envs.HUGGINGFACE_TOKEN = value


@router.post("/download_selected")
async def download_selected(
    request: List[DownloadSelectedDto], background_tasks: BackgroundTasks
):
    try:
        task = asyncio.create_task(
            download_multiple(list(map(lambda x: dict(x), request)))
        )
        background_tasks.add_task(lambda: task)
        return {
            "status": "received",
            "message": "Download request received.",
        }
    except Exception as e:
        raise HTTPException(
            status_code=500, detail=f"Error processing request: {str(e)}"
        )


@router.post("/download_custom_model")
async def download_custom_model(
    request: ModelDownloadRequest, background_tasks: BackgroundTasks
):
    try:
        id = hex(hash(str(request.url)))

        exists = downloadHistory.is_exists(id)

        if exists:
            return {
                "status": "duplicated",
                "message": "Download request received but skip",
            }

        res = {
            "type": "download",
            "data": {
                "id": id,
                "name": request.model_type,
                "url": str(request.url),
                "model_type": request.model_type,
                "status": "IN_QUEUE",
            },
        }

        await manager.broadcast(json.dumps(res))

        downloadHistory.put(res["data"])

        task = asyncio.create_task(
            download_async(id, request.model_type, str(request.url), request.model_type)
        )
        background_tasks.add_task(lambda: task)  # schedule it after response

        return {
            "status": "received",
            "message": "Download request received.",
        }

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

@router.get("/download-images")
async def download_images_zip():
    """
    Creates a zip file of all images in ./output_images folder (recursive)
    and returns it as a file download using aiofiles for non-blocking I/O.
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

        # Return the zip file as a download
        return FileResponse(
            path=temp_zip_path,
            filename="output.zip",
            media_type="application/zip",
            background=None,  # File will be deleted after response
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
