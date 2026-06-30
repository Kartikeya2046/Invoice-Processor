"""LLM-based structured field extraction via local Ollama (Qwen2.5:3b).

Replaces the previous Mistral-API version. Uses Ollama's structured-output
feature (format=<pydantic model>.model_json_schema()) so the model's response
is constrained to valid JSON matching ExtractedInvoiceData/ExtractedBOEData
directly -- no markdown-fence stripping, no brace-hunting, no JSON decode
retry loop needed, since Ollama enforces the schema before returning.
"""

import json
import logging
import os
from urllib import error as urllib_error
from urllib import request as urllib_request

from models import ExtractedBOEData, ExtractedInvoiceData

logger = logging.getLogger(__name__)

OLLAMA_BASE_URL = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")
OLLAMA_MODEL = os.getenv("OLLAMA_MODEL", "qwen2.5:3b")

# ponytail: 90s timeout + num_predict cap, calibrated from the same local-Ollama
# behavior documented for v2's SLM validation calls (30-50s typical generation
# time on shared/local hardware; timeout=30 was observed to be insufficient
# there). Revisit if the 16-page Mouser invoice needs more headroom.
_REQUEST_TIMEOUT = 90.0
_NUM_PREDICT = 2000


def _call_ollama(prompt: str, schema: dict) -> dict:
    """Call Ollama's /api/chat with a JSON-schema-constrained response."""
    payload = {
        "model": OLLAMA_MODEL,
        "messages": [{"role": "user", "content": prompt}],
        "format": schema,
        "stream": False,
        "options": {"temperature": 0.0, "num_predict": _NUM_PREDICT},
    }

    req = urllib_request.Request(
        f"{OLLAMA_BASE_URL}/api/chat",
        data=json.dumps(payload).encode("utf-8"),
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    try:
        with urllib_request.urlopen(req, timeout=_REQUEST_TIMEOUT) as response:
            body = response.read().decode("utf-8")
    except urllib_error.URLError as e:
        raise ConnectionError(
            f"Could not reach Ollama at {OLLAMA_BASE_URL} -- is `ollama serve` running? ({e})"
        ) from e

    data = json.loads(body)
    content = data.get("message", {}).get("content")
    if not content:
        raise RuntimeError(f"Ollama response missing message.content: {data}")
    return json.loads(content)


_INVOICE_PROMPT = """You are a commercial document data extraction assistant for
Indian import/local trade invoices. Extract exactly these fields:

- po_number: Purchase Order number
- supplier: Supplier / seller name
- invoice_number: Invoice number / Inv #
- invoice_date: Invoice date, format YYYY-MM-DD
- quantity: Total/header quantity, if shown at the document/header level (not per line item)
- unit_price: Header unit price, if shown at the document/header level (not per line item)
- cgst: CGST tax amount -- ONLY if explicitly labeled "CGST" appears in the document, else null
- sgst: SGST tax amount -- ONLY if explicitly labeled "SGST" appears in the document, else null
- line_items: array of per-product rows, each with:
    - line_number: the line's number as printed (numbers may skip, e.g. 1,2,3,6,7 -- keep as printed)
    - product_id: Mouser Part No, MFG Part No, or equivalent product/part code
    - quantity: quantity for this line
    - unit_price: unit price for this line

RULES:
- Missing field = null. Never guess or invent a value.
- Do not copy example values from these instructions into the output.
- Numbers: no currency symbols, no thousands-separator commas.
- Extract every line item found; do not stop early.

DOCUMENT TEXT:
{text}"""

_BOE_PROMPT = """You are a customs document data extraction assistant for
Indian Bill of Entry (BOE) documents. Extract exactly these fields:

- boe_number: B/E Number
- boe_date: B/E Date, format YYYY-MM-DD
- igst: IGST amount
- cust_duty: Customs Duty total (also seen as "Cust. Duty Total" or "Basic Customs Duty")
- sbcess: SBCESS amount

RULES:
- Missing field = null. Never guess or invent a value.
- Numbers: no currency symbols, no thousands-separator commas.

DOCUMENT TEXT:
{text}"""


def extract_invoice_fields(cleaned_text: str) -> ExtractedInvoiceData:
    """Extract invoice fields from OCR text via local Qwen2.5:3b."""
    result = _call_ollama(
        _INVOICE_PROMPT.format(text=cleaned_text),
        ExtractedInvoiceData.model_json_schema(),
    )
    return ExtractedInvoiceData.model_validate(result)


def extract_boe_fields(cleaned_text: str) -> ExtractedBOEData:
    """Extract Bill of Entry fields from OCR text via local Qwen2.5:3b."""
    result = _call_ollama(
        _BOE_PROMPT.format(text=cleaned_text),
        ExtractedBOEData.model_json_schema(),
    )
    return ExtractedBOEData.model_validate(result)
