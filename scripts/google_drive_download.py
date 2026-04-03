import argparse
import os
import subprocess


def main(path: str, url: str) -> str:
    os.chdir(path)
    result = subprocess.run(
        ["gdown", "-q", "--fuzzy", url],
    )
    return "success" if result.returncode == 0 else "failed"


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Download files from Google Drive and extract if ZIP"
    )

    parser.add_argument(
        "--path",
        type=str,
        required=True,
        help="Path where the file will be downloaded and extracted",
    )

    parser.add_argument(
        "--url",
        type=str,
        required=True,
        help="Google Drive URL of the file to download",
    )

    args = parser.parse_args()
    status = main(args.path, args.url)
    if status == "success":
        exit(0)
    else:
        exit(1)
