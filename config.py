"""Application configuration loaded from environment variables."""

import os

from dotenv import load_dotenv


load_dotenv()

MONGO_URI = os.getenv("MONGO_URI", "mongodb+srv://guptakartikeya2046_db_user:<Kartikeya.2046>@cluster0.ivw6vlc.mongodb.net/")
MONGO_DB_NAME = os.getenv("MONGO_DB_NAME", "invoice_processor")
TESSERACT_PATH = os.getenv("TESSERACT_PATH", "tesseract")

