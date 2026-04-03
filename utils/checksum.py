import asyncio
import hashlib

import httpx


async def compute_sha256(filepath: str) -> str:
    loop = asyncio.get_event_loop()

    def _hash() -> str:
        sha256 = hashlib.sha256()
        with open(filepath, "rb") as f:
            for chunk in iter(lambda: f.read(1024 * 1024), b""):
                sha256.update(chunk)
        return sha256.hexdigest()

    return await loop.run_in_executor(None, _hash)


async def fetch_hf_sha256(
    owner: str, repo: str, filepath: str, token: str | None = None
) -> str | None:
    api_url = f"https://huggingface.co/api/models/{owner}/{repo}?blobs=true"
    req_headers = {}
    if token:
        req_headers["Authorization"] = f"Bearer {token}"
    try:
        async with httpx.AsyncClient() as client:
            r = await client.get(
                api_url, headers=req_headers, follow_redirects=True, timeout=15
            )
            if r.status_code == 200:
                for sibling in r.json().get("siblings", []):
                    if sibling.get("rfilename") == filepath:
                        sha256 = sibling.get("lfs", {}).get("sha256")
                        if sha256:
                            return sha256.lower()
    except Exception:
        pass
    return None


async def fetch_civitai_sha256(
    model_version_id: str, token: str | None = None
) -> str | None:
    api_url = f"https://civitai.com/api/v1/model-versions/{model_version_id}"
    req_headers = {}
    if token:
        req_headers["Authorization"] = f"Bearer {token}"
    try:
        async with httpx.AsyncClient() as client:
            r = await client.get(
                api_url, headers=req_headers, follow_redirects=True, timeout=10
            )
            if r.status_code == 200:
                for f in r.json().get("files", []):
                    sha256 = f.get("hashes", {}).get("SHA256")
                    if sha256:
                        return sha256.lower()
    except Exception:
        pass
    return None
