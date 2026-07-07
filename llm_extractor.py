"""LLM-based structured field extraction via local Ollama (Qwen2.5:7b).

Regex extraction is temporarily disabled. All fields — headers and line items —
are extracted exclusively via the LLM. This isolates the LLM path for testing
and generalization across unseen invoice formats.
"""

import json
import logging
import os
from urllib import error as urllib_error
from urllib import request as urllib_request

from models import ExtractedBOEData, ExtractedInvoiceData

logger = logging.getLogger(__name__)

OLLAMA_BASE_URL = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")
OLLAMA_MODEL = os.getenv("OLLAMA_MODEL", "qwen2.5:7b")

_REQUEST_TIMEOUT = 180.0
_NUM_PREDICT = 8000


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

    from pathlib import Path
    debug_path = Path(__file__).parent / "debug_last_prompt.txt"
    with open(debug_path, "w", encoding="utf-8") as f:
        f.write(prompt)
    logger.debug(f"Prompt character count: {len(prompt)}")

    try:
        with urllib_request.urlopen(req, timeout=_REQUEST_TIMEOUT) as response:
            body = response.read().decode("utf-8")
    except urllib_error.HTTPError as e:
        body = e.read().decode("utf-8", errors="replace")
        raise RuntimeError(f"Ollama rejected the request ({e.code}): {body}") from e
    except urllib_error.URLError as e:
        raise ConnectionError(
            f"Could not reach Ollama at {OLLAMA_BASE_URL} -- is `ollama serve` running? ({e})"
        ) from e

    data = json.loads(body)
    content = data.get("message", {}).get("content")
    if not content:
        raise RuntimeError(f"Ollama response missing message.content: {data}")
    return json.loads(content)


# ---------------------------------------------------------------------------
# Prompts
# ---------------------------------------------------------------------------

_INVOICE_PROMPT = """You are a strict invoice data extraction engine. Extract structured fields from the invoice text and return a valid JSON object. No explanation, no markdown, no text outside the JSON.

FIELDS:
- invoice_number: Unique invoice ID after labels like "Invoice No.", "Invoice #", "INV#", "No.:". Can be numeric or alphanumeric. Never use an address or date.
- po_number: Purchase order number(s) after labels like "PO:", "P.O. No.", "Purchase Order No.", or inside a Remarks block. If multiple POs exist, return them all comma-separated. Never use a street address, city, or postal code as the PO number — a PO number is a short numeric or alphanumeric code, not a location.
- supplier: Name of the company that issued this invoice. Found in the header under "Seller", "Seller / Exporter", "Sold By", "From", or as the company name at the top. Never use the buyer or consignee name. Never use a bank name (e.g. containing 'Bank', 'NA', 'Chase', 'HSBC') or a wire-transfer/remittance block as the supplier — the supplier is the company that issued and is billing for the invoice, usually named at the very top of the document or near 'Sold By'/'From'.
- invoice_date: Date the invoice was issued. Found after "Invoice Date:", "Date:", "Inv Dt.". Return exactly as it appears in the document.
- line_items: Array of every product row in the line items table. For each row extract:
    - line_number: Row sequence number. If not printed, number them yourself starting from 1.
    - product_id: The product code, part number, model number, or SKU. Never use a prose description.
    - quantity: Integer quantity only — no decimals, no fractional units. Example: 10, not 10.0 or 10.5.
    - unit_price: Price per single unit. This is NOT the extended price or Amount column. No currency symbols, no commas.

EXAMPLES:
Example A (Qty before Part Number):
Qty  Part No.       Unit Price
10   ABC-123-XY     4.50
5    DEF-456-Z      12.00
->
[{"line_number": "1", "product_id": "ABC-123-XY", "quantity": "10", "unit_price": "4.50"},
 {"line_number": "2", "product_id": "DEF-456-Z", "quantity": "5", "unit_price": "12.00"}]

Example B (Line No. first, Description column present but ignored):
Line  Item Code   Description        Qty   Rate
1     SKU9981     Steel bracket 2in  20    3.25
2     SKU4471     Rubber gasket      100   0.45
->
[{"line_number": "1", "product_id": "SKU9981", "quantity": "20", "unit_price": "3.25"},
 {"line_number": "2", "product_id": "SKU4471", "quantity": "100", "unit_price": "0.45"}]

Example C (no line numbers printed at all, must be inferred):
Product      Ordered   Price/Unit
WIDGET-A     50        1.20
WIDGET-B     30        2.75
->
[{"line_number": "1", "product_id": "WIDGET-A", "quantity": "50", "unit_price": "1.20"},
 {"line_number": "2", "product_id": "WIDGET-B", "quantity": "30", "unit_price": "2.75"}]

RULES:
- If a field is not found, return null. Never guess or hallucinate.
- Do not put addresses, bank details, or descriptions into any field.
- Bank details, wire transfer instructions, and remittance addresses are NEVER a valid value for any field except when explicitly extracting banking information (which this schema does not request).
- Return numbers as plain digits only — no currency symbols, no angle brackets, no surrounding punctuation of any kind.
- unit_price x quantity should equal the line Amount — use the smaller per-unit value, not the total.
- Do not include subtotal, grand total, or header rows in line_items.
- Column order, column count, and column labels vary by vendor — identify each field by its MEANING (a quantity is a whole number of units ordered/shipped, a unit price is a small per-item currency value, a product_id is an alphanumeric code not a sentence), not by position in the row.
- Extract EVERY row in the line items table, in order, regardless of table length. Do not stop early, summarize, or skip rows because the table is long.
- Ignore description/text columns entirely — never place a description into product_id.

DOCUMENT TEXT:
__DOCUMENT_TEXT__"""

_LINE_ITEMS_PROMPT = """You are a strict invoice data extraction engine. Extract ONLY line items from the invoice text chunk and return a valid JSON object. No explanation, no markdown, no text outside the JSON.

FIELDS:
- line_items: Array of every product row in the line items table. For each row extract:
    - line_number: Row sequence number. If not printed, number them yourself starting from 1.
    - product_id: The product code, part number, model number, or SKU. Never use a prose description.
    - quantity: Integer quantity only — no decimals, no fractional units. Example: 10, not 10.0 or 10.5.
    - unit_price: Price per single unit. This is NOT the extended price or Amount column. No currency symbols, no commas.

EXAMPLES:
Example A (Qty before Part Number):
Qty  Part No.       Unit Price
10   ABC-123-XY     4.50
5    DEF-456-Z      12.00
->
[{"line_number": "1", "product_id": "ABC-123-XY", "quantity": "10", "unit_price": "4.50"},
 {"line_number": "2", "product_id": "DEF-456-Z", "quantity": "5", "unit_price": "12.00"}]

RULES:
- If a field is not found, return null. Never guess or hallucinate.
- Return numbers as plain digits only — no currency symbols, no angle brackets, no surrounding punctuation of any kind.
- unit_price x quantity should equal the line Amount — use the smaller per-unit value, not the total.
- Do not include subtotal, grand total, or header rows in line_items.
- Column order, column count, and column labels vary by vendor — identify each field by its MEANING (a quantity is a whole number of units ordered/shipped, a unit price is a small per-item currency value, a product_id is an alphanumeric code not a sentence), not by position in the row.
- Extract EVERY row in the line items table, in order, regardless of table length. Do not stop early, summarize, or skip rows because the table is long.
- Ignore description/text columns entirely — never place a description into product_id.

DOCUMENT TEXT:
__DOCUMENT_TEXT__"""

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
__DOCUMENT_TEXT__"""


# ---------------------------------------------------------------------------
# Preprocessing — kept active since it only cleans whitespace and removes
# repeated Mouser boilerplate. Safe to run on any invoice.
# ---------------------------------------------------------------------------

def _preprocess_invoice_text(raw_text: str) -> str:
    import re
    from collections import Counter

    # 1. Clean up basic whitespace but preserve intentional blank lines
    lines = raw_text.split('\n')
    clean_lines = [line.strip() for line in lines]
    text = '\n'.join(clean_lines)

    # 2. Split into paragraph chunks (separated by 1 or more blank lines)
    chunks = [c.strip() for c in re.split(r'\n{2,}', text) if c.strip()]

    def normalize(text_block):
        # Normalize spaces and lowercase for comparison
        return re.sub(r'\s+', ' ', text_block).lower()

    # 3. Remove exact paragraph chunks that appear >= 3 times
    chunk_counts = Counter(normalize(c) for c in chunks)
    seen_chunks = set()
    filtered_chunks = []
    
    removed_paragraph_chars = 0
    for chunk in chunks:
        norm = normalize(chunk)
        if chunk_counts[norm] >= 3:
            if norm not in seen_chunks:
                seen_chunks.add(norm)
                filtered_chunks.append(chunk)
            else:
                removed_paragraph_chars += len(chunk)
        else:
            filtered_chunks.append(chunk)

    final_text = '\n\n'.join(filtered_chunks)
    
    orig_len = len(raw_text)
    new_len = len(final_text)
    reduction = ((orig_len - new_len) / orig_len * 100) if orig_len > 0 else 0
    if removed_paragraph_chars > 0:
        logger.info(f"Removed {removed_paragraph_chars} chars of duplicate paragraph boilerplate")
        
    logger.info(f"Preprocessing reduced length from {orig_len} to {new_len} chars ({reduction:.2f}% reduction)")
    return final_text


# ---------------------------------------------------------------------------
# REGEX HEADER EXTRACTION — DISABLED
# ---------------------------------------------------------------------------
#
# def _extract_header_fields(text: str) -> dict:
#     import re
#     result = {
#         "invoice_number": None,
#         "po_number": None,
#         "invoice_date": None,
#         "supplier": None,
#         "total_amount": None
#     }
#     # ... all regex patterns removed ...
#     return result


# ---------------------------------------------------------------------------
# REGEX LINE ITEM EXTRACTION — DISABLED
# ---------------------------------------------------------------------------
#
# def _extract_line_items_regex(text: str) -> list:
#     import re
#     line_items = []
#     matches = re.finditer(
#         r'^\s*(\d+)\s+'
#         r'([A-Z0-9]{2,4}-[A-Z0-9/\.\-]+)\s+'
#         r'(\d+)\s+'
#         r'(\d+)\s+'
#         r'(\d+)\s+'
#         r'([0-9]+\.[0-9]{2,3})\s+'
#         r'([0-9]+\.[0-9]{2})',
#         text,
#         re.MULTILINE
#     )
#     for m in matches:
#         line_items.append({ ... })
#     return line_items


# ---------------------------------------------------------------------------
# Main extraction — LLM only
# ---------------------------------------------------------------------------

def _extract_universal_hints(text: str) -> str:
    import re
    hints = []
    
    def _looks_plausible(value: str) -> bool:
        return bool(re.search(r"[0-9]", value)) and len(value.strip()) >= 3
    
    # Invoice Number
    m_inv = re.search(r"(?:Invoice\s+No|Invoice\s+Number|Invoice\s+#|INV#)\s*[:\.]?\s*(\S+)", text, re.IGNORECASE)
    if m_inv:
        val = m_inv.group(1).strip()
        if _looks_plausible(val):
            hints.append(f"invoice_number: {val}")
        
    # PO Number
    m_po = re.search(r"(?:PO\s+Number|P\.O\.|Purchase\s+Order|PO:)\s*[:\.]?\s*(\S+)", text, re.IGNORECASE)
    if m_po:
        val = m_po.group(1).strip()
        if _looks_plausible(val):
            hints.append(f"po_number: {val}")
        
    # Invoice Date
    m_date = re.search(r"^\s*(?:Invoice\s+Date|Date)\s*[:\.]?\s*([^\n]+)", text, re.IGNORECASE | re.MULTILINE)
    if m_date:
        hints.append(f"invoice_date: {m_date.group(1).strip()}")
        
    if hints:
        return "HINTS (unverified, may be wrong):\n" + "\n".join(hints)
    return ""

def extract_invoice_fields(raw_text: str) -> ExtractedInvoiceData:
    """Extract all invoice fields exclusively via LLM (regex disabled)."""

    # Step 1 — Clean whitespace only (no vendor-specific stripping)
    cleaned_text = _preprocess_invoice_text(raw_text)

    # Step 2 — Single LLM call for HEADERS ONLY using first ~3000 chars
    header_text_slice = cleaned_text[:3000]
    logger.info(f"Extracting headers from first {len(header_text_slice)} chars...")

    header_schema = {
        "type": "object",
        "properties": {
            "invoice_number": {"type": "string"},
            "po_number":      {"type": "string"},
            "supplier":       {"type": "string"},
            "invoice_date":   {"type": "string"}
        },
        "required": ["invoice_number", "po_number", "supplier", "invoice_date"]
    }

    hints_text = _extract_universal_hints(header_text_slice)
    if hints_text:
        text_with_hints = f"{hints_text}\n\n{header_text_slice}"
    else:
        text_with_hints = header_text_slice

    prompt = _INVOICE_PROMPT.replace("__DOCUMENT_TEXT__", text_with_hints)
    header_result = _call_ollama(prompt, header_schema)

    # Step 3 — Chunked extraction for LINE ITEMS
    chunk_size = 4000
    overlap = 200
    chunks = []
    start = 0
    while start < len(cleaned_text):
        end = start + chunk_size
        if end < len(cleaned_text):
            newline_idx = cleaned_text.rfind('\n', start + chunk_size - overlap, end)
            if newline_idx != -1:
                end = newline_idx
        chunks.append(cleaned_text[start:end])
        start = end
        
    logger.info(f"Split {len(cleaned_text)} chars into {len(chunks)} chunks for line item extraction.")

    line_items_schema = {
        "type": "object",
        "properties": {
            "line_items": {
                "type": "array",
                "items": {
                    "type": "object",
                    "properties": {
                        "line_number": {"type": "string"},
                        "product_id":  {"type": "string"},
                        "quantity":    {"type": "string"},
                        "unit_price":  {"type": "string"}
                    },
                    "required": ["line_number", "product_id", "quantity", "unit_price"]
                }
            }
        },
        "required": ["line_items"]
    }

    raw_items = []
    seen_combinations = set()
    
    for i, chunk_text in enumerate(chunks, 1):
        logger.info(f"Extracting line items from chunk {i}/{len(chunks)}...")
        chunk_prompt = _LINE_ITEMS_PROMPT.replace("__DOCUMENT_TEXT__", chunk_text)
        chunk_result = _call_ollama(chunk_prompt, line_items_schema)
        
        chunk_items = chunk_result.get("line_items") or []
        logger.info(f"Chunk {i} returned {len(chunk_items)} items.")
        
        for item in chunk_items:
            pid = str(item.get("product_id") or "").strip()
            qty = str(item.get("quantity") or "").strip()
            price = str(item.get("unit_price") or "").strip()
            
            if not pid and not qty and not price:
                raw_items.append(item)
                continue
                
            combo = f"{pid}|{qty}|{price}"
            if combo not in seen_combinations:
                seen_combinations.add(combo)
                raw_items.append(item)

    logger.info(f"Total deduplicated line items across all chunks: {len(raw_items)}")

    # Step 4 — Sanitize concatenated line items
    import re
    
    def _extract_numeric(raw: str) -> str:
        match = re.search(r"[0-9]+(?:\.[0-9]+)?", raw)
        return match.group(0) if match else ""
        
    formatted_items = []
    for idx, item in enumerate(raw_items, 1):
        qty_raw   = str(item.get("quantity")   or "").replace(",", "").strip()
        price_raw = str(item.get("unit_price") or "").replace(",", "").strip()
        
        qty = _extract_numeric(qty_raw)
        price = _extract_numeric(price_raw)
        line_num = str(idx)

        qty_valid   = bool(re.fullmatch(r"[0-9]+", qty))
        price_valid = bool(re.fullmatch(r"[0-9]+(\.[0-9]+)?", price))

        if not qty_valid:
            logger.warning(f"Non-numeric quantity, keeping row with null: qty={qty_raw!r} item={item}")
        if not price_valid:
            logger.warning(f"Non-numeric unit_price, keeping row with null: price={price_raw!r} item={item}")

        formatted_items.append({
            "line_number": line_num,
            "product_id":  str(item.get("product_id")  or ""),
            "quantity":    qty if qty_valid else None,
            "unit_price":  price if price_valid else None,
        })

    logger.info(f"Sanitization complete. Final line items count: {len(formatted_items)}.")

    # Step 5 — Assemble result
    result = {
        "invoice_number": header_result.get("invoice_number"),
        "po_number":      header_result.get("po_number"),
        "invoice_date":   header_result.get("invoice_date"),
        "supplier":       header_result.get("supplier"),
        "total_amount":   None,  # not in schema yet
        "line_items":     formatted_items,
    }

    # Coerce non-null header fields to strings
    for key in ["invoice_number", "po_number", "invoice_date", "supplier"]:
        if result[key] is not None:
            result[key] = str(result[key])

    logger.info(
        f"Final extraction — invoice_number={result['invoice_number']}, "
        f"po_number={result['po_number']}, supplier={result['supplier']}, "
        f"invoice_date={result['invoice_date']}, line_items={len(formatted_items)}"
    )

    # Step 5 — Validate and return
    _validate_extraction(result, cleaned_text)
    return ExtractedInvoiceData.model_validate(result)


# ---------------------------------------------------------------------------
# Validation — loosened to accept alphanumeric fields from any vendor
# ---------------------------------------------------------------------------

def _validate_extraction(result: dict, source_text: str) -> None:
    import re
    total = 0
    passed = 0

    def log_check(field, is_pass, val):
        nonlocal total, passed
        total += 1
        if is_pass:
            passed += 1
            logger.info(f"Validation [{field}]: PASS - {val}")
        else:
            logger.info(f"Validation [{field}]: FAIL - {val}")

    # invoice_number: accept any non-empty alphanumeric string (not just digits)
    inv = result.get("invoice_number")
    log_check("invoice_number", bool(inv and re.search(r"[A-Z0-9]", str(inv), re.IGNORECASE)), inv)

    # po_number: accept any non-empty string (may be comma-separated list)
    po = result.get("po_number")
    log_check("po_number", bool(po and str(po).strip()), po)

    # supplier: reject if it looks like an address
    sup = result.get("supplier")
    if sup:
        sup_s = str(sup)
        has_addr = bool(re.search(r"\b(Street|Main|Road|Ave|Blvd|TX|Plot|Survey)\b", sup_s, re.IGNORECASE))
        log_check("supplier", not has_addr, sup)
    else:
        log_check("supplier", False, sup)

    # invoice_date: accept any non-empty string
    date = result.get("invoice_date")
    log_check("invoice_date", bool(date and str(date).strip()), date)

    # line_items: at least one item
    items = result.get("line_items")
    li_pass = bool(items and isinstance(items, list) and len(items) > 0)
    log_check("line_items", li_pass, f"{len(items)} items" if isinstance(items, list) else items)

    status = "POOR"
    if total > 0:
        ratio = passed / total
        if ratio == 1.0:
            status = "GOOD"
        elif ratio > 0.5:
            status = "PARTIAL"

    logger.info(f"Extraction validation: {passed}/{total} fields passed. Status: {status}")


def extract_boe_fields(cleaned_text: str) -> ExtractedBOEData:
    """Extract Bill of Entry fields from OCR text via local Qwen2.5:7b."""
    result = _call_ollama(
        _BOE_PROMPT.replace("__DOCUMENT_TEXT__", cleaned_text),
        ExtractedBOEData.model_json_schema(),
    )
    return ExtractedBOEData.model_validate(result)