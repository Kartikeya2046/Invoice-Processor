"""Application configuration loaded from environment variables."""

import os
from pathlib import Path

from dotenv import load_dotenv


load_dotenv(dotenv_path=Path(__file__).resolve().parent / ".env")

MONGO_URI = os.getenv("MONGO_URI", "mongodb://localhost:27017")
MONGO_DB_NAME = os.getenv("MONGO_DB_NAME", "invoice_processor")


def _resolve_tesseract_path(path_value: str) -> str:
	path = Path(path_value)
	if path.is_dir():
		return str(path / "tesseract.exe")
	return path_value


TESSERACT_PATH = _resolve_tesseract_path(os.getenv("TESSERACT_PATH", "tesseract"))

