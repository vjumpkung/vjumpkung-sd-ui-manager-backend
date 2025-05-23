import os
import zipfile
import asyncio
from pathlib import Path


async def _create_zip_file(output_dir: Path, temp_zip_path: str) -> None:
    """
    Create zip file in a separate thread to avoid blocking the event loop.
    """

    def _zip_files():
        with zipfile.ZipFile(temp_zip_path, "w", zipfile.ZIP_DEFLATED) as zipf:
            # Walk through all files in the directory recursively
            for root, dirs, files in os.walk(output_dir):
                for file in files:
                    file_path = Path(root) / file
                    # Calculate relative path from output_dir
                    arcname = file_path.relative_to(output_dir)
                    zipf.write(file_path, arcname)

    # Run the blocking zip operation in a thread pool
    loop = asyncio.get_event_loop()
    await loop.run_in_executor(None, _zip_files)
