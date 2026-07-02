import os
from models import InvoiceData, ProcessingStatus, ValidationResult
from llm_extractor import extract_invoice_fields
from preprocessor import clean_ocr_text, classify_document
from ocr_processor import process_file
from validator import validate_invoice_fields

os.environ["PATH"] = "D:\\llama.cpp\\llama-b9844-bin-win-cuda-13.3-x64;" + os.environ["PATH"]

raw_text = process_file("test_data/sample_invoice.png")
text_for_llm = clean_ocr_text(raw_text)
classification = classify_document(text_for_llm)
doc_type = classification.get("type")
doc_confidence = classification.get("confidence")

extracted_fields = extract_invoice_fields(text_for_llm)
validation_dict = validate_invoice_fields(extracted_fields.model_dump())

validation_result = ValidationResult(
    confidence_score=validation_dict["confidence_score"],
    needs_review=validation_dict["needs_review"],
    missing_fields=validation_dict["missing_fields"],
    issues=validation_dict["issues"],
)
processing_status = ProcessingStatus.review_required if validation_dict["needs_review"] else ProcessingStatus.extracted

invoice = InvoiceData(
    filename="sample_invoice.png",
    extracted_text=text_for_llm,
    processing_status=processing_status,
    document_type=doc_type,
    document_type_confidence=doc_confidence,
    extracted_fields=extracted_fields,
    validation_result=validation_result,
    correction_history=[],
)

print(invoice.model_dump_json(indent=2))
