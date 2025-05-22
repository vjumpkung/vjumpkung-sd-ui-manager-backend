from fastapi import APIRouter, HTTPException, Body, BackgroundTasks, Request, Response
from fastapi.responses import JSONResponse
import torch
import aiofiles
import json
from pydantic import BaseModel, HttpUrl
from typing import Literal
from typing import List
import time
from worker.program_logs import programLog
from random import randint
from event_handler import manager
from env_manager import envs
from config.load_config import RUNPOD_POD_ID, UI_TYPE
from worker.download import download_async, download_multiple
import asyncio
from uuid import uuid4
from history_manager import downloadHistory
from worker.check_process import programStatus


class ModelDownloadRequest(BaseModel):
    url: HttpUrl
    model_type: Literal[
        "checkpoints",
        "clip_vision",
        "controlnet",
        "diffusion_models",
        "embeddings",
        "hypernetworks",
        "ipadapter",
        "loras",
        "text_encoders",
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
                "gpu_name": None,
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
    async with aiofiles.open("model_packs.json") as fp:
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
        id = str(uuid4())

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
