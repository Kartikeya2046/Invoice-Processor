"""Upload endpoints for invoice files."""

import os
from datetime import datetime
from pathlib import Path
from typing import List
from uuid import uuid4

from fastapi import APIRouter, File, HTTPException, UploadFile

from database import invoices_collection
from models import InvoiceData
from ocr_processor import process_file


router = APIRouter()

UPLOADS_DIR = Path(__file__).resolve().parent.parent / "uploads"
SUPPORTED_EXTENSIONS = {".pdf", ".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff"}


def _is_supported_file(file_name: str) -> bool:
	return Path(file_name).suffix.lower() in SUPPORTED_EXTENSIONS


@router.post("/upload")
async def upload_file(file: UploadFile = File(...)):
	if not file.filename:
		raise HTTPException(status_code=400, detail="No file name provided")

	if not _is_supported_file(file.filename):
		raise HTTPException(status_code=400, detail="Unsupported file type")

	UPLOADS_DIR.mkdir(parents=True, exist_ok=True)
	file_extension = Path(file.filename).suffix.lower()
	temp_name = f"{uuid4().hex}{file_extension}"
	temp_path = UPLOADS_DIR / temp_name

	try:
		with temp_path.open("wb") as buffer:
			content = await file.read()
			buffer.write(content)

		extracted_text = process_file(str(temp_path))
		if extracted_text.startswith("Error") or extracted_text.startswith("Unsupported"):
			raise HTTPException(status_code=400, detail=extracted_text)

		metadata = {
			"file_size": temp_path.stat().st_size,
			"timestamp": datetime.utcnow().isoformat(),
		}

		invoice = InvoiceData(
			filename=file.filename,
			extracted_text=extracted_text,
			metadata=metadata,
		)

		result = invoices_collection.insert_one(invoice.dict())
		return {
			"message": "File processed successfully",
			"file_id": str(result.inserted_id),
			"filename": file.filename,
			"text_length": len(extracted_text),
		}
	except HTTPException:
		raise
	except Exception as error:
		raise HTTPException(status_code=500, detail=f"Upload failed: {error}")
	finally:
		try:
			if temp_path.exists():
				temp_path.unlink()
		except OSError:
			pass


@router.post("/upload-folder/{folder_path:path}")
async def upload_folder(folder_path: str):
	if not os.path.isdir(folder_path):
		raise HTTPException(status_code=400, detail="Folder path does not exist")

	files: List[Path] = [
		Path(folder_path) / name
		for name in os.listdir(folder_path)
		if _is_supported_file(name)
	]

	results = []
	processed_count = 0

	for file_path in files:
		try:
			extracted_text = process_file(str(file_path))
			if extracted_text.startswith("Error") or extracted_text.startswith("Unsupported"):
				raise ValueError(extracted_text)

			metadata = {
				"file_size": file_path.stat().st_size,
				"timestamp": datetime.utcnow().isoformat(),
			}

			invoice = InvoiceData(
				filename=file_path.name,
				extracted_text=extracted_text,
				metadata=metadata,
			)

			result = invoices_collection.insert_one(invoice.dict())
			processed_count += 1
			results.append(
				{
					"filename": file_path.name,
					"status": "success",
					"file_id": str(result.inserted_id),
				}
			)
		except Exception as error:
			results.append(
				{
					"filename": file_path.name,
					"status": "error",
					"file_id": None,
					"message": str(error),
				}
			)

	return {
		"total_files": len(files),
		"processed_count": processed_count,
		"results": results,
	}

