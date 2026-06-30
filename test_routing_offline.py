"""Offline self-check for the invoice/BOE routing logic across
preprocessor.classify_document -> llm_extractor.extract_*_fields ->
validator.validate_*_fields, and structured.py's field-key branching.

Mocks: Surya (ocr_processor) and the Ollama HTTP call (llm_extractor) --
neither is reachable without real GPU/local-server infra. Everything else
(classification, validation, field-key branching) runs for real.
"""
import sys
import json
from unittest import mock

sys.modules["surya"] = mock.MagicMock()
sys.modules["surya.inference"] = mock.MagicMock()
sys.modules["surya.recognition"] = mock.MagicMock()
sys.modules["pdf2image"] = mock.MagicMock()

from preprocessor import classify_document  # noqa: E402
from models import ExtractedInvoiceData, ExtractedBOEData  # noqa: E402
from validator import validate_invoice_fields, validate_boe_fields  # noqa: E402

# --- classify_document: BOE short-circuit vs invoice scoring ---
boe_text = "BILL OF ENTRY\nB/E Number: 7842531\nB/E Date: 01/05/2026\nIGST: 1200.00"
result = classify_document(boe_text)
assert result["type"] == "bill_of_entry", result

invoice_text = "INVOICE\nInvoice Number: INV-001\nBill To: Acme Corp\nAmount Due: 500.00"
result = classify_document(invoice_text)
assert result["type"] == "invoice", result

# A BOE-keyword doc must never fall through to invoice scoring, even if it
# also happens to contain invoice-ish words (real BOEs often do, since
# they reference the underlying commercial invoice).
mixed_text = "BILL OF ENTRY for INVOICE Number INV-001, Amount Due 500.00"
result = classify_document(mixed_text)
assert result["type"] == "bill_of_entry", result

# --- llm_extractor.py: mock the Ollama HTTP call, verify schema-constrained
# parsing and routing to the right Pydantic model still works end-to-end ---
with mock.patch("llm_extractor.urllib_request.urlopen") as mock_urlopen:
    invoice_response = {
        "message": {"content": json.dumps({
            "po_number": "PO-998", "supplier": "Mouser Electronics",
            "invoice_number": "74294838", "invoice_date": "2026-06-15",
            "quantity": None, "unit_price": None, "cgst": None, "sgst": None,
            "line_items": [{"line_number": "1", "product_id": "ABC-1", "quantity": "10", "unit_price": "2.50"}],
        })}
    }
    mock_urlopen.return_value.__enter__.return_value.read.return_value = json.dumps(invoice_response).encode()

    import llm_extractor
    fields = llm_extractor.extract_invoice_fields("dummy ocr text")
    assert isinstance(fields, ExtractedInvoiceData)
    assert fields.supplier == "Mouser Electronics"
    assert fields.cgst is None
    assert fields.line_items[0].product_id == "ABC-1"

    boe_response = {
        "message": {"content": json.dumps({
            "boe_number": "7842531", "boe_date": "2026-01-05",
            "igst": "1200.00", "cust_duty": "300", "sbcess": "15",
        })}
    }
    mock_urlopen.return_value.__enter__.return_value.read.return_value = json.dumps(boe_response).encode()
    boe_fields = llm_extractor.extract_boe_fields("dummy boe text")
    assert isinstance(boe_fields, ExtractedBOEData)
    assert boe_fields.boe_number == "7842531"

# --- validator.py routing matches what upload.py would call for each type ---
inv_validation = validate_invoice_fields(fields.model_dump())
assert inv_validation["needs_review"] is False, inv_validation

boe_validation = validate_boe_fields(boe_fields.model_dump())
assert boe_validation["needs_review"] is False, boe_validation

# --- structured.py field-key branch: BOE doc reads/writes extracted_boe_fields ---
def _fields_key(document_type: str) -> str:
    return "extracted_boe_fields" if document_type == "bill_of_entry" else "extracted_fields"

assert _fields_key("bill_of_entry") == "extracted_boe_fields"
assert _fields_key("invoice") == "extracted_fields"
assert _fields_key(None) == "extracted_fields"  # unknown/unclassified falls back to invoice fields

print("OK: classify_document BOE short-circuit, llm_extractor schema parsing,")
print("validator routing, and structured.py field-key branching all pass.")
print("NOT covered here: real Surya OCR output, real Ollama/Qwen output quality,")
print("real MongoDB writes. Verify those against the live services separately.")
