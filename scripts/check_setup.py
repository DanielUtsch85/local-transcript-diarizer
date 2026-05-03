from __future__ import annotations

import importlib.util
import shutil
import sys


def main() -> int:
    checks = [
        ("Python >= 3.10", sys.version_info >= (3, 10)),
        ("ffmpeg", shutil.which("ffmpeg") is not None),
        ("altair", importlib.util.find_spec("altair") is not None),
        ("diarize", importlib.util.find_spec("diarize") is not None),
        ("pandas", importlib.util.find_spec("pandas") is not None),
        ("psutil", importlib.util.find_spec("psutil") is not None),
        ("streamlit", importlib.util.find_spec("streamlit") is not None),
        ("python-docx", importlib.util.find_spec("docx") is not None),
        ("whisperx", importlib.util.find_spec("whisperx") is not None),
    ]

    failed = False
    for label, ok in checks:
        marker = "OK" if ok else "FEHLT"
        print(f"{marker:5} {label}")
        failed = failed or not ok

    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(main())
