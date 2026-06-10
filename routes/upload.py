import os
import uuid
from pathlib import Path
from fastapi import APIRouter, UploadFile, File, HTTPException

from models import (
    InvoiceData,
    ExtractedInvoiceData,
    LineItem,
    ValidationResult,
    ProcessingStatus
)
from llm_extractor import extract_invoice_fields
from preprocessor import (
    clean_ocr_text,
    classify_document,
    extract_pages_pdfplumber,
    deduplicate_pages
)
from ocr_processor import process_file
from database import invoices_collection

router = APIRouter()

ALLOWED_EXTENSIONS = {".pdf", ".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff"}
UPLOADS_DIR = Path("uploads")
UPLOADS_DIR.mkdir(exist_ok=True)

@router.post("/upload")
async def upload_file(file: UploadFile = File(...)):
    ext = Path(file.filename).suffix.lower()
    if ext not in ALLOWED_EXTENSIONS:
        raise HTTPException(status_code=400, detail="Invalid file extension")

    temp_id = str(uuid.uuid4())
    temp_path = UPLOADS_DIR / f"{temp_id}{ext}"

    with open(temp_path, "wb") as f:
        f.write(await file.read())

    raw_text = ""
    text_for_llm = ""
    processing_status = ProcessingStatus.pending
    extracted_fields = None
    validation_result = None
    extraction_error = None
    doc_type = None
    doc_confidence = None

    try:
        raw_text = process_file(str(temp_path))
        
        pages = []
        if ext == ".pdf":
            try:
                pages = extract_pages_pdfplumber(str(temp_path))
            except Exception as e:
                pass # Fallback to raw_text
                
        if ext == ".pdf" and pages:
            text_for_llm = deduplicate_pages(pages)
        else:
            text_for_llm = clean_ocr_text(raw_text)

        classification = classify_document(text_for_llm)
        doc_type = classification.get("type")
        doc_confidence = classification.get("confidence")

        if doc_type != "invoice":
            processing_status = ProcessingStatus.failed
            extraction_error = "Document is not classified as an invoice"
        else:
            try:
                extracted_dict = extract_invoice_fields(text_for_llm)
                if extracted_dict:
                    line_items_data = extracted_dict.pop("line_items", [])
                    line_items = []
                    for item in line_items_data:
                        line_items.append(LineItem(**item))
                    
                    extracted_fields = ExtractedInvoiceData(**extracted_dict, line_items=line_items)
                    
                    # Validation
                    required_fields = ["invoice_number", "vendor_name", "grand_total"]
                    missing_fields = []
                    non_null_count = 0
                    
                    # Exclude line_items from the 20 fields count
                    fields_to_check = [
                        "invoice_number", "invoice_date", "due_date", "vendor_name", 
                        "vendor_address", "vendor_email", "vendor_phone", "customer_name", 
                        "customer_address", "billing_address", "shipping_address", 
                        "po_number", "payment_terms", "currency", "subtotal", 
                        "tax_amount", "tax_rate", "discount_amount", "shipping_amount", "grand_total"
                    ]
                    
                    for field in fields_to_check:
                        val = getattr(extracted_fields, field, None)
                        if val is not None and val != "":
                            non_null_count += 1
                        else:
                            if field in required_fields:
                                missing_fields.append(field)
                                
                    confidence_score = non_null_count / 20.0
                    needs_review = confidence_score < 0.70 or len(missing_fields) > 0
                    
                    validation_result = ValidationResult(
                        confidence_score=confidence_score,
                        needs_review=needs_review,
                        missing_fields=missing_fields,
                        issues=["Missing required fields"] if missing_fields else []
                    )
                    
                    if not needs_review:
                        processing_status = ProcessingStatus.extracted
                    else:
                        processing_status = ProcessingStatus.review_required
                else:
                    processing_status = ProcessingStatus.failed
                    extraction_error = "Failed to parse extraction results"
            except Exception as e:
                processing_status = ProcessingStatus.failed
                extraction_error = str(e)

    finally:
        if temp_path.exists():
            temp_path.unlink()

    invoice = InvoiceData(
        filename=file.filename,
        extracted_text=text_for_llm if text_for_llm else raw_text,
        processing_status=processing_status,
        document_type=doc_type,
        document_type_confidence=doc_confidence,
        extracted_fields=extracted_fields,
        validation_result=validation_result,
        extraction_error=extraction_error,
        correction_history=[]
    )

    doc = invoice.model_dump()
    result = invoices_collection.insert_one(doc)

    return {
        "file_id": str(result.inserted_id),
        "filename": invoice.filename,
        "processing_status": invoice.processing_status,
        "confidence_score": validation_result.confidence_score if validation_result else 0.0,
        "needs_review": validation_result.needs_review if validation_result else True
    }
