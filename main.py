import asyncio
import json
import os
import threading

import uvicorn
from fastapi import FastAPI, Request, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles

import config.load_config as CONFIG
from event_handler import manager
from api import router
from worker.check_process import programStatus
from worker.program_logs import programLog

app = FastAPI(docs_url=None, redoc_url=None)

app.include_router(router)

origins = ["*"]

app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Mount the static files from the NextJS export
# app.mount("/static", StaticFiles(directory="web/static"), name="static")
app.mount("/_next", StaticFiles(directory="web/_next"), name="next_assets")


# For other static files in the root directory
@app.get("/favicon.ico")
async def favicon():
    return FileResponse("web/favicon.ico")


# Route all other requests to the NextJS index.html
@app.get("/{full_path:path}")
async def serve_nextjs(full_path: str, request: Request):
    # Try to serve the exact path
    path = f"web/{full_path}"

    # Check if the path exists and is a file
    if os.path.exists(path) and os.path.isfile(path):
        return FileResponse(path)

    # If path is a directory, look for index.html
    if os.path.exists(path) and os.path.isdir(path):
        index_path = os.path.join(path, "index.html")
        if os.path.exists(index_path):
            return FileResponse(index_path)

    # Fall back to the main index.html for client-side routing
    return FileResponse("web/index.html")


@app.websocket("/ws/{client_id}")
async def websocket_endpoint(websocket: WebSocket, client_id: str):
    await manager.connect(websocket)
    await manager.send_message(json.dumps({"message": "ws connect"}), websocket)
    try:
        while True:
            data = await websocket.receive_text()
    except WebSocketDisconnect:
        manager.disconnect(websocket)


if __name__ == "__main__":
    _thread = threading.Thread(
        target=asyncio.run, args=(programLog.monitor_log(),), daemon=True
    )
    _thread2 = threading.Thread(
        target=asyncio.run,
        args=(
            programStatus.ping_check(
                "127.0.0.1", programStatus.MAP_PORT[CONFIG.UI_TYPE]
            ),
        ),
        daemon=True,
    )
    _thread.start()
    _thread2.start()

    uvicorn.run(
        "main:app",
        port=int(CONFIG.PORT),
        reload=CONFIG.RELOAD,
        host=CONFIG.HOST,
        proxy_headers=True,
        forwarded_allow_ips="*",
    )
