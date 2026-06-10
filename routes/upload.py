"""Upload endpoints for invoice files."""

import os
from datetime import datetime
from pathlib import Path
from typing import List
from uuid import uuid4

from fastapi import APIRouter, File, HTTPException, UploadFile

from database import invoices_collection
from llm_extractor import extract_invoice_fields
from models import InvoiceData, ProcessingStatus, ExtractedInvoiceData, ValidationResult
from preprocessor import clean_ocr_text, classify_document, extract_pages_pdfplumber, deduplicate_pages
from ocr_processor import process_file
from validator import validate_extracted_fields


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

	file_size = 0
	try:
		with temp_path.open("wb") as buffer:
			content = await file.read()
			buffer.write(content)
		file_size = temp_path.stat().st_size
		extracted_text = process_file(str(temp_path))
		pdf_pages = extract_pages_pdfplumber(str(temp_path)) if file_extension == ".pdf" else None
	except Exception as error:
		raise HTTPException(status_code=500, detail=f"Upload failed: {error}")
	finally:
		try:
			if temp_path.exists():
				temp_path.unlink()
		except OSError:
			pass

	if extracted_text.startswith("Error") or extracted_text.startswith("Unsupported"):
		raise HTTPException(status_code=400, detail=extracted_text)

	metadata = {
		"file_size": file_size,
		"timestamp": datetime.utcnow().isoformat(),
	}

	# --- New intelligence pipeline ---
	if file_extension == ".pdf" and pdf_pages:
		text_for_llm = deduplicate_pages(pdf_pages)
	else:
		text_for_llm = clean_ocr_text(extracted_text)

	doc_classification = classify_document(text_for_llm)

	processing_status = ProcessingStatus.failed
	extracted_fields = None
	validation_result = None
	extraction_error = None
	processing_logs = None

	if doc_classification["type"] == "invoice":
		try:
			raw_fields = extract_invoice_fields(text_for_llm)
			if raw_fields:
				extracted_fields = ExtractedInvoiceData(**{k: v for k, v in raw_fields.items() if k != "line_items"})
				line_items_data = raw_fields.get("line_items") or []
				from models import LineItem
				extracted_fields.line_items = [LineItem(**item) for item in line_items_data if isinstance(item, dict)]
				validation_result_dict = validate_extracted_fields(raw_fields)
				validation_result = ValidationResult(**validation_result_dict)
				processing_status = (
					ProcessingStatus.review_required if validation_result.needs_review else ProcessingStatus.extracted
				)
			else:
				extraction_error = "LLM returned empty response"
				processing_status = ProcessingStatus.failed
		except ConnectionError as e:
			processing_logs = f"Connection error: {e}"
			extraction_error = f"Connection error: {e}"
			processing_status = ProcessingStatus.failed
		except ValueError as e:
			processing_logs = f"Parsing error: {e}"
			extraction_error = f"Parsing error: {e}"
			processing_status = ProcessingStatus.failed
		except Exception as e:
			processing_logs = f"Extraction error: {str(e)}"
			extraction_error = f"Extraction error: {str(e)}"
			processing_status = ProcessingStatus.failed
	else:
		extraction_error = f"Document classified as '{doc_classification['type']}', not an invoice"
		processing_status = ProcessingStatus.failed

	invoice = InvoiceData(
		filename=file.filename,
		extracted_text=extracted_text,
		metadata=metadata,
		processing_status=processing_status,
		document_type=doc_classification["type"],
		document_type_confidence=doc_classification["confidence"],
		extracted_fields=extracted_fields,
		validation_result=validation_result,
		extraction_error=extraction_error,
		processing_logs=processing_logs,
	)

	result = invoices_collection.insert_one(invoice.dict())
	return {
		"message": "File processed successfully",
		"file_id": str(result.inserted_id),
		"filename": file.filename,
		"text_length": len(extracted_text),
		"document_type": doc_classification["type"],
		"processing_status": processing_status.value,
		"confidence_score": validation_result.confidence_score if validation_result else None,
		"needs_review": validation_result.needs_review if validation_result else None,
	}


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

