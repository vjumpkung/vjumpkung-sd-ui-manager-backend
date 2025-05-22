import os
import dotenv

dotenv.load_dotenv(override=True)

PORT = os.getenv("PORT") or "8000"  # port for running app

RELOAD = "true" == os.getenv("RELOAD")
HOST = os.getenv("HOST") or "127.0.0.1"  # host for running app

UI_TYPE = os.getenv("UI_TYPE") or "COMFY"  # COMFY, FORGE, INVOKEAI
RESOURCE_PATH = os.getenv("RESOURCE_PATH") or "./my-runpod-volume/models"
LOG_PATH = os.getenv("LOG_PATH") or "./backend.log"
PROGRAM_LOG = os.getenv("PROGRAM_LOG") or "./program.log"

RUNPOD_POD_ID = os.environ.get("RUNPOD_POD_ID") or "xxxxxxxxxxxxxx"

JUPYTER_LAB_PORT = os.environ.get("JUPYTER_LAB_PORT") or "8888"
