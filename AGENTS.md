# AGENTS.md

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
- `COMFYUI_PATH`: ComfyUI installation directory (defaults to `/notebooks/ComfyUI`)
- `PORT` / `HOST`: Server bind address
- `RESOURCE_PATH`: Where model files are stored
- `OUTPUT_PATH`: Where generated images are stored
- `LOG_PATH`: Path to UI process log file
- `PROGRAM_LOG`: Path to the SD UI process log file tailed by the log worker
- `CIVITAI_TOKEN` / `HUGGINGFACE_TOKEN` / `HF_TOKEN`: API tokens for downloads. `HUGGINGFACE_TOKEN` and `HF_TOKEN` are synchronized for Hugging Face CLI compatibility.
- `RUNPOD_POD_ID`: RunPod pod ID exposed via `/api/checkcuda`
- `DEBUG`: Enables uvicorn logging when `True`
- `RELOAD`: Enables uvicorn hot-reload

## Architecture

**Entry point:** `main.py` — configures FastAPI app, mounts routes, and uses a `lifespan` context manager to start two background async tasks on startup (cancelled on shutdown):
1. `worker/program_logs.py` — tails `PROGRAM_LOG` file, broadcasts new lines via WebSocket
2. `worker/check_process.py` — pings UI port (8188/7860/9090) every 5s, broadcasts RUNNING/NOT_RUNNING status

**Static file serving:** `/_next` is mounted as a `StaticFiles` directory. All other non-`/api` and non-`/ws` paths fall through to `serve_nextjs()`, which tries to serve from `web/`, falls back to `web/index.html` for client-side routing. Paths starting with `api/` or `ws/` are explicitly rejected with 404 to avoid shadowing those routers.

**Download flow:** `api.py` → `worker/download.py` → resolves CivitAI metadata and a short-lived signed URL, then calls `scripts/civitai_download.py` in a worker thread so Windows event loops do not need asyncio subprocess support; the helper runs `aria2c`. Hugging Face file URLs prefer the `hf` CLI when available and otherwise use `aria2c`; Google Drive uses `gdown`. Status is streamed to history/WebSocket, and SHA256 or a URL hash is used to deduplicate downloads.

**WebSocket:** `event_handler.py` manages connections. All status changes, log lines, and download progress broadcast to all connected clients at `/ws/{client_id}`.

**Model packs:** `resources/{comfy,forge,invokeai}_model_packs.json` define categories with URLs pointing to external JSON lists of models.

**UI process control:** `worker/restart_program.py` calls shell scripts (e.g., `/notebooks/stop_process.sh`) that are external to this repo.

**Download history:** `history_manager.py` — in-memory async dict tracking download status (`DownloadStatus` enum: `IN_QUEUE`, `DOWNLOADING`, `RETRYING`, `COMPLETED`, `FAILED`), keyed by UUID5 of the URL. `update_status` is typed to `DownloadStatus`.

**Env tokens:** `env_manager.py` — runtime-mutable `CIVITAI_TOKEN` / `HUGGINGFACE_TOKEN` updated via `PUT /api/update_env/{civitai|huggingface}` with JSON body `{"value": ...}`. Token problems return HTTP 422 with detail code `TOKEN_REQUIRED` from `/api/download_custom_model`.

**Custom node install flow:** `POST /api/install_custom_node` creates a job (`create_install_job`) and runs it as a background asyncio task (`start_install_job` -> `run_install_job` -> `install_custom_nodes`), returning HTTP 202 with a job snapshot. Nodes install sequentially behind `_install_lock`; each item reports `queued` -> `installing` -> `installed` / `already_installed` / `failed`, and ComfyUI is restarted once at the end if anything was installed (a restart failure is reported in the job message, not as a job failure). Progress is broadcast over the WebSocket as `custom_node` (item status), `custom_node_log` (one command output line) and `custom_node_job` (job status: `queued`, `running`, `completed`, `partial`, `failed`) events; clients map rows by the request `index`, not the URL. Jobs are kept in memory only (last 20, finished ones pruned first) and `GET /api/install_custom_node/{job_id}` returns the snapshot, including the last 200 log lines per node. Dependencies: `install.py` (if present) runs as-is and is not constrained; `requirements.txt` is parsed (comments, continuations, `-r` includes, `--index-url`/`--extra-index-url`/`--find-links`/`--no-index`/`--constraint`/`--pre`, `-e`) and installed line-by-line with `uv pip install`. Protected packages (`PROTECTED_PACKAGES`: numpy, torch, torchaudio, torchvision, transformers) are always skipped, and their currently installed versions (read with `uv pip list --format json`; if that fails the node fails and nothing is installed) are pinned through a temporary constraints file passed with `-c` so other requirements cannot change them. A failed requirement line is logged and skipped: the node stays `installed` with a message listing the failed lines (`failed_requirements`) and the job becomes `partial`; the node only fails when every installable line fails. Each requirement is logged as `SKIPPED` / `Installing` / `OK` / `FAILED` with timings, followed by a `Dependencies finished` summary.

**Export:** `worker/export_zip.py` — zips `OUTPUT_PATH` directory and streams it as a download via `GET /api/download-images`.

**Logging:** `log_manager.py` — Rich-based console logger (no file output); `worker/create_log_file.py` — ensures log files exist on startup.

**Utils:**
- `utils/generate_uuid.py` — UUID5 generation from URLs
- `utils/checksum.py` — SHA256 helpers: `compute_sha256(filepath)` hashes a local file in a thread executor and `fetch_hf_sha256(owner, repo, filepath, token)` fetches the expected hash from the Hugging Face Hub API. CivitAI file metadata is resolved in `worker/download.py` so its filename, size, file ID, and checksum stay consistent.
- `utils/enums.py` — `DownloadStatus(str, Enum)` with values `IN_QUEUE`, `DOWNLOADING`, `RETRYING`, `COMPLETED`, `FAILED`
- `utils/ws_messages.py` — Pydantic models for WebSocket message serialization: `DownloadMessage`/`DownloadData` (download events), `TokenRequiredMessage`/`TokenRequiredData` (token_required events), `CustomNodeMessage`/`CustomNodeData`, `CustomNodeLogMessage`/`CustomNodeLogData`, `CustomNodeJobMessage`/`CustomNodeJobData` (custom_node, custom_node_log, custom_node_job events), `MonitorMessage`/`MonitorData` (process status), `LogMessage`/`LogData` (log lines). Use `model_dump_json()` to broadcast.

## API Endpoints

| Method | Path                         | Description                                                         |
| ------ | ---------------------------- | ------------------------------------------------------------------- |
| GET    | `/api/checkcuda`             | CUDA/GPU info + process status                                      |
| GET    | `/api/download_history`      | In-memory download queue                                            |
| GET    | `/api/get_model_packs`       | Load model packs JSON for current UI type                           |
| GET    | `/api/logs`                  | Current program log buffer                                          |
| GET    | `/api/download-images`       | Stream zip of output images                                         |
| PUT    | `/api/update_env/{type}`     | Update CivitAI or HuggingFace token at runtime                      |
| POST   | `/api/download_custom_model` | Queue a single model download                                       |
| POST   | `/api/import_models`         | Queue multiple model downloads                                      |
| POST   | `/api/download_selected`     | Queue selected pack models                                          |
| POST   | `/api/restart`               | Trigger UI process restart                                          |
| POST   | `/api/install_custom_node`   | Queue a background install of one or more ComfyUI custom nodes (202 + `job_id`); progress over WebSocket; restarts ComfyUI once at the end |
| GET    | `/api/install_custom_node/{job_id}` | Job snapshot (per-node status + last 200 log lines); 404 if unknown |
| WS     | `/ws/{client_id}`            | WebSocket for real-time events                                      |

## Skills

When working on FastAPI routes, Pydantic models, or any API-related code in this project, automatically trigger the `fastapi` skill before making changes.

## Key External Dependencies

- `aria2c` must be installed on the system (used for CivitAI and ordinary HTTP downloads, and as the Hugging Face fallback)
- `hf` is optional; when the Hugging Face CLI is on `PATH`, Hugging Face file URLs use `hf download`
- Shell scripts at `/notebooks/` or `/invokeai/` for start/stop operations (external, deployment-specific)
- PyTorch (optional) — imported conditionally for CUDA/GPU detection; `ZIMAGE` UI type skips this
