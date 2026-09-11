import io
import os
import tempfile
import unittest
from unittest.mock import patch

from scripts import civitai_download


class FakeResult:
    returncode = 0


class CivitaiDownloadScriptTests(unittest.TestCase):
    def test_signed_url_is_passed_to_aria_through_stdin(self) -> None:
        signed_url = "https://storage.example/model?X-Amz-Signature=secret"
        expected_sha256 = "a" * 64
        captured = {}

        def run(command, **kwargs):
            captured["command"] = command
            captured.update(kwargs)
            return FakeResult()

        with (
            tempfile.TemporaryDirectory() as temp_dir,
            patch.object(civitai_download.sys, "stdin", io.StringIO(signed_url)),
            patch.object(civitai_download.subprocess, "run", side_effect=run),
            patch.dict(
                os.environ,
                {"CIVITAI_TOKEN": "secret-token", "KEEP_ME": "yes"},
                clear=True,
            ),
        ):
            return_code = civitai_download.download(
                temp_dir,
                ".civitai-11-partial.part",
                expected_sha256,
            )

        self.assertEqual(return_code, 0)
        self.assertEqual(captured["command"][0], "aria2c")
        self.assertNotIn(signed_url, captured["command"])
        aria_input = captured["input"].splitlines()
        self.assertEqual(aria_input[0], signed_url)
        self.assertIn("  out=.civitai-11-partial.part", aria_input)
        self.assertIn(f"  checksum=sha-256={expected_sha256}", aria_input)
        self.assertNotIn("CIVITAI_TOKEN", captured["env"])
        self.assertEqual(captured["env"]["KEEP_ME"], "yes")


if __name__ == "__main__":
    unittest.main()
