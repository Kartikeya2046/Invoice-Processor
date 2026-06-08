"""Application configuration loaded from environment variables."""

import os
import sys
from pathlib import Path

from dotenv import load_dotenv


load_dotenv(dotenv_path=Path(__file__).resolve().parent / ".env")

def _resolve_mongo_uri(uri_value: str | None) -> str:
    if not uri_value:
        return "mongodb://localhost:27017"

    normalized = uri_value.strip()
    if "<db_password>" in normalized or "<" in normalized or ">" in normalized:
        return "mongodb://localhost:27017"
    return normalized


MONGO_URI = _resolve_mongo_uri(os.getenv("MONGO_URI"))
MONGO_DB_NAME = os.getenv("MONGO_DB_NAME", "invoice_processor")


import sys

def _resolve_tesseract_path(path_value: str) -> str:
    path = Path(path_value)
    if path.is_dir():
        binary = "tesseract.exe" if sys.platform == "win32" else "tesseract"
        return str(path / binary)
    return path_value


TESSERACT_PATH = _resolve_tesseract_path(os.getenv("TESSERACT_PATH", "tesseract"))

