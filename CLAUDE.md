# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## What This Project Does

FastAPI backend for managing Stable Diffusion UI processes (ComfyUI, Forge, InvokeAI, ZImage). Handles model downloads (CivitAI, Hugging Face, Google Drive via aria2c), real-time WebSocket broadcasting, and process monitoring. Serves a Next.js frontend from the `web/` directory.

## Commands

```bash
# Install dependencies
pip install -r requirements.txt

# Run the server
python main.py

# Configure environment (copy and edit)
cp .env.example .env
```

## Environment Configuration

Key `.env` variables:
- `UI_TYPE`: `COMFY` | `FORGE` | `INVOKEAI` | `ZIMAGE`
- `PORT` / `HOST`: Server bind address
- `RESOURCE_PATH`: Where model files are stored
- `OUTPUT_PATH`: Where generated images are stored
- `LOG_PATH`: Path to UI process log file
- `PROGRAM_LOG`: Path to the SD UI process log file tailed by the log worker
- `CIVITAI_TOKEN` / `HF_TOKEN`: API tokens for downloads
- `RUNPOD_POD_ID`: RunPod pod ID exposed via `/api/checkcuda`
- `DEBUG`: Enables uvicorn logging when `True`
- `RELOAD`: Enables uvicorn hot-reload

## Architecture

**Entry point:** `main.py` — configures FastAPI app, mounts routes, starts two background threads on startup:
1. `worker/program_logs.py` — tails `PROGRAM_LOG` file, broadcasts new lines via WebSocket
2. `worker/check_process.py` — pings UI port (8188/7860/9090) every 5s, broadcasts RUNNING/NOT_RUNNING status

**Static file serving:** `/_next` is mounted as a `StaticFiles` directory. All other non-`/api` and non-`/ws` paths fall through to `serve_nextjs()`, which tries to serve from `web/`, falls back to `web/index.html` for client-side routing. Paths starting with `api/` or `ws/` are explicitly rejected with 404 to avoid shadowing those routers.

**Download flow:** `api.py` → `worker/download.py` → spawns `aria2c` subprocess (or `gdown` for Google Drive) → streams stdout to history/WebSocket. UUID5 is generated from URL to deduplicate downloads.

**WebSocket:** `event_handler.py` manages connections. All status changes, log lines, and download progress broadcast to all connected clients at `/ws/{client_id}`.

**Model packs:** `resources/{comfy,forge,invokeai}_model_packs.json` define categories with URLs pointing to external JSON lists of models.

**UI process control:** `worker/restart_program.py` calls shell scripts (e.g., `/notebooks/stop_process.sh`) that are external to this repo.

**Download history:** `history_manager.py` — in-memory async dict tracking download status (IN_QUEUE, RETRYING, FAILED, etc.), keyed by UUID5 of the URL.

**Env tokens:** `env_manager.py` — runtime-mutable `CIVITAI_TOKEN` / `HUGGINGFACE_TOKEN` updated via `PUT /api/update_env/{civitai|huggingface}`.

**Export:** `worker/export_zip.py` — zips `OUTPUT_PATH` directory and streams it as a download via `GET /api/download-images`.

**Logging:** `log_manager.py` — Rich-based console logger (no file output); `worker/create_log_file.py` — ensures log files exist on startup.

## API Endpoints

| Method | Path | Description |
|--------|------|-------------|
| GET | `/api/checkcuda` | CUDA/GPU info + process status |
| GET | `/api/download_history` | In-memory download queue |
| GET | `/api/get_model_packs` | Load model packs JSON for current UI type |
| GET | `/api/logs` | Current program log buffer |
| GET | `/api/download-images` | Stream zip of output images |
| PUT | `/api/update_env/{type}` | Update CivitAI or HuggingFace token at runtime |
| POST | `/api/download_custom_model` | Queue a single model download |
| POST | `/api/import_models` | Queue multiple model downloads |
| POST | `/api/download_selected` | Queue selected pack models |
| POST | `/api/restart` | Trigger UI process restart |
| WS | `/ws/{client_id}` | WebSocket for real-time events |

## Key External Dependencies

- `aria2c` must be installed on the system (used for all model downloads)
- Shell scripts at `/notebooks/` or `/invokeai/` for start/stop operations (external, deployment-specific)
- PyTorch (optional) — imported conditionally for CUDA/GPU detection; `ZIMAGE` UI type skips this
