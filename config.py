"""Application configuration loaded from environment variables."""

import os

from dotenv import load_dotenv


load_dotenv()

MONGO_URI = os.getenv("MONGO_URI", "mongodb://localhost:27017")
MONGO_DB_NAME = os.getenv("MONGO_DB_NAME", "invoice_processor")
TESSERACT_PATH = os.getenv("TESSERACT_PATH", "tesseract")

