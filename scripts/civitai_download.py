#!/usr/bin/env python3
"""Run one resumable CivitAI aria2c transfer without asyncio subprocesses."""

import argparse
import os
import subprocess
import sys
import urllib.parse


def _validated_output_name(value: str) -> str:
    if (
        not value
        or value in {".", ".."}
        or os.path.basename(value.replace("\\", "/")) != value
        or any(character in value for character in ("\r", "\n", "\x00"))
    ):
        raise ValueError("invalid output filename")
    return value


def _validated_sha256(value: str) -> str:
    digest = value.lower()
    if len(digest) != 64 or any(
        character not in "0123456789abcdef" for character in digest
    ):
        raise ValueError("invalid SHA-256")
    return digest


def _read_signed_url() -> str:
    signed_url = sys.stdin.readline().rstrip("\r\n")
    parsed = urllib.parse.urlparse(signed_url)
    if (
        parsed.scheme != "https"
        or not parsed.hostname
        or any(character in signed_url for character in ("\r", "\n", "\x00"))
    ):
        raise ValueError("invalid signed URL")
    return signed_url


def _aria_command(
    destination: str, output_name: str, expected_sha256: str
) -> list[str]:
    return [
        "aria2c",
        "--input-file=-",
        "--max-connection-per-server=8",
        "--split=8",
        "--min-split-size=1M",
        "--continue=true",
        "--file-allocation=none",
        "--auto-file-renaming=false",
        "--allow-overwrite=true",
        "--check-certificate=true",
        "--check-integrity=true",
        f"--checksum=sha-256={expected_sha256}",
        "--max-tries=5",
        "--retry-wait=2",
        "--connect-timeout=15",
        "--timeout=60",
        "--summary-interval=0",
        "--show-console-readout=false",
        "--console-log-level=warn",
        "--download-result=hide",
        f"--dir={destination}",
        f"--out={output_name}",
    ]


def download(destination: str, output_name: str, expected_sha256: str) -> int:
    destination = os.path.abspath(destination)
    if not os.path.isdir(destination) or any(
        character in destination for character in ("\r", "\n", "\x00")
    ):
        raise ValueError("invalid destination")
    output_name = _validated_output_name(output_name)
    expected_sha256 = _validated_sha256(expected_sha256)
    signed_url = _read_signed_url()
    aria_input = (
        f"{signed_url}\n"
        f"  dir={destination}\n"
        f"  out={output_name}\n"
        f"  checksum=sha-256={expected_sha256}\n"
    )
    process_env = os.environ.copy()
    for key in tuple(process_env):
        if key.casefold() == "civitai_token":
            process_env.pop(key)

    try:
        result = subprocess.run(
            _aria_command(destination, output_name, expected_sha256),
            input=aria_input,
            text=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            env=process_env,
            check=False,
        )
    except FileNotFoundError:
        print("aria2c is not installed", file=sys.stderr)
        return 127
    return result.returncode


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--destination", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--sha256", required=True)
    args = parser.parse_args()
    try:
        return download(args.destination, args.output, args.sha256)
    except (OSError, ValueError) as exc:
        print(f"CivitAI download helper failed: {exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
