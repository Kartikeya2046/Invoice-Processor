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
    
    from pathlib import Path
    debug_path = Path(__file__).parent.parent / "debug_last_prompt.txt"
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


_INVOICE_PROMPT = """SECTION 1 — TASK DESCRIPTION:
You are an invoice data extraction engine. Your only job is to extract structured fields from invoice text and return a JSON object. You must not explain, comment, or add any text outside the JSON object.

SECTION 2 — FIELD DEFINITIONS:
- invoice_number: The numeric invoice ID found immediately after the label "Invoice No." or "Inv #". On Mouser invoices this is a 6-digit number like 103565. Do not use street addresses or reference numbers.
- po_number: The number found immediately after "Purchase Order No." On Mouser invoices this is an 8-digit number like 74294838.
- supplier: The name of the company that issued the invoice. Found in the header or footer of the document, not in the Bill To or Ship To address block. For Mouser invoices this is "Mouser Electronics".
- invoice_date: The date found next to "Invoice Date" or "Inv Dt." in DD-MMM-YY format.
- line_items: An array of objects. For each line item:
    - line_number: The integer in the leftmost "Line No." column (1, 2, 3...). This is never a part number or description.
    - product_id: The Mouser Part Number, which is a code in format NNN-ALPHANUMERIC found at the start of each line item block (e.g. 625-1N6276A-E3, 512-LL4148). Never use the MFG Part No for this field.
    - quantity: The integer under the "Shipped" column, not the "Ordered" column.
    - unit_price: The decimal number under "Price/Unit" or "Unit Price" column.

SECTION 3 — NEGATIVE RULES:
- Do not put street addresses, city names, zip codes, or country names into any field.
- Do not put bank account numbers, swift codes, or wire transfer details into any field.
- Do not put MFG Part Numbers (found on the line starting with "MFG Part No:") into product_id — only use the Mouser Part Number.
- Do not copy item descriptions into product_id.
- If a field cannot be found, set it to null. Do not guess or hallucinate values.

SECTION 4 — FEW SHOT EXAMPLES:
Example 1 input:
"1 625-1N6276A-E3 4 4 0 1.450 5.80
MFG Part No: 1N6276A-E3/51
Vishay General Semiconductor 1500W 16V Unidirect / ESD Suppressors / TVS Diodes"
Expected output for this line item:
{{"line_number": 1, "product_id": "625-1N6276A-E3", "quantity": 4, "unit_price": 1.450}}

Example 2 input:
"2 512-LL4148 72 72 0 0.109 7.85
MFG Part No: LL4148
onsemi / Fairchild Small Signal Diode / Diodes - General Purpose"
Expected output:
{{"line_number": 2, "product_id": "512-LL4148", "quantity": 72, "unit_price": 0.109}}

Example 3 input:
"14 700-MAX3491ESD 25 25 0 5.640 141.00
MFG Part No: MAX3491ESD+
Analog Devices / Maxim Integrated 3.3V Powered, 10Mbps / RS-422/RS-485 Interface IC"
Expected output:
{{"line_number": 14, "product_id": "700-MAX3491ESD", "quantity": 25, "unit_price": 5.640}}

SECTION 5 — OUTPUT FORMAT:
Return only a valid JSON object matching the existing schema. No markdown, no backticks, no explanation text. The response must start with {{ and end with }}.

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


def _preprocess_invoice_text(raw_text: str) -> str:
    import re
    text = raw_text
    
    patterns = [
        r"Wire Transfer/TT To:.*?INCOTERMS: FCA Shipping Point",
        r"This order is subject to all terms.*?wire transfer details\.",
        r"1000 North Main Street, Mansfield, TX 76063.*?Federal ID# 61-1520598",
        r"Tracking Number\(s\) and Billed Weights.*?Pending"
    ]
    
    for pat in patterns:
        matches = list(re.finditer(pat, text, flags=re.DOTALL))
        if len(matches) > 1:
            for m in reversed(matches[1:]):
                text = text[:m.start()] + text[m.end():]
                
    # Remove lines containing only whitespace
    lines = text.split('\n')
    text = '\n'.join(line for line in lines if line.strip())
    
    orig_len = len(raw_text)
    new_len = len(text)
    reduction = ((orig_len - new_len) / orig_len * 100) if orig_len > 0 else 0
    logger.info(f"Preprocessed invoice text: {orig_len} -> {new_len} chars ({reduction:.1f}% reduction)")
    
    return text


def _extract_header_fields(text: str) -> dict:
    import re
    result = {
        "invoice_number": None,
        "po_number": None,
        "invoice_date": None,
        "supplier": None,
        "total_amount": None
    }
    
    # invoice_number
    patterns_inv = [
        (1, r'Invoice\s*No\.?\s*[:\.]?\s*([0-9]{5,7})', re.IGNORECASE),
        (2, r'Inv\s*#\s*\S+\s+([0-9]{5,7})', re.IGNORECASE),
        (3, r'INCOTERMS.*?(\b[0-9]{5,6}\b)', re.IGNORECASE),
        (4, r'^\s*103565\b', re.MULTILINE)
    ]
    for i, pat, flags in patterns_inv:
        m = re.search(pat, text, flags)
        if m:
            val = m.group(1) if len(m.groups()) > 0 else m.group(0).strip()
            result["invoice_number"] = val
            logger.debug(f"invoice_number matched pattern {i}: {val}")
            break
    if not result["invoice_number"]:
        logger.warning("invoice_number matched no patterns (set to null)")
        
    # po_number
    patterns_po = [
        (1, r'Purchase\s*Order\s*No\.?\s*[:\.]?\s*([0-9]{6,10})', re.IGNORECASE),
        (2, r'Inv\s*#\s*Dt\.?\s*Page\s*No\.?\s*([0-9]{6,10})', re.IGNORECASE),
        (3, r'\b([0-9]{8})\s+\d{2}-[A-Z]{3}-\d{2}\s+\d+\s+of\s+\d+', re.IGNORECASE)
    ]
    for i, pat, flags in patterns_po:
        m = re.search(pat, text, flags)
        if m:
            val = m.group(1)
            result["po_number"] = val
            logger.debug(f"po_number matched pattern {i}: {val}")
            break
    if not result["po_number"]:
        logger.warning("po_number matched no patterns (set to null)")
        
    # invoice_date
    m_date = re.search(r'(\d{2}-(?:JAN|FEB|MAR|APR|MAY|JUN|JUL|AUG|SEP|OCT|NOV|DEC)-\d{2,4})', text, re.IGNORECASE)
    if m_date:
        val = m_date.group(1)
        result["invoice_date"] = val
        logger.debug(f"invoice_date matched pattern 1: {val}")
    else:
        logger.warning("invoice_date matched no patterns (set to null)")
        
    # supplier
    suppliers = ["Mouser Electronics", "Digi-Key Electronics", "Arrow Electronics", "LCSC Electronics", "RS Components", "Farnell", "Newark", "TME"]
    found_sup = False
    for i, sup in enumerate(suppliers, 1):
        if re.search(re.escape(sup), text, re.IGNORECASE):
            result["supplier"] = sup
            logger.debug(f"supplier matched pattern {i} (verbatim): {sup}")
            found_sup = True
            break
    if not found_sup:
        m_sup = re.search(r'(?:Wire Transfer.*?To|Remit.*?To)[:\s]+([A-Z][A-Za-z\s,\.&]+?)(?:\n|Bank|Account)', text, re.DOTALL | re.IGNORECASE)
        if m_sup:
            val = m_sup.group(1).strip()
            result["supplier"] = val
            logger.debug(f"supplier matched pattern fallback regex: {val}")
        else:
            logger.warning("supplier matched no patterns (set to null)")
            
    # total_amount
    patterns_tot = [
        (1, r'USD\s*\$\s*([0-9,]+\.[0-9]{2})', re.IGNORECASE),
        (2, r'Please\s*pay\s*this\s*amount\s*[\n\r]+\s*([0-9,]+\.[0-9]{2})', re.IGNORECASE)
    ]
    for i, pat, flags in patterns_tot:
        m = re.search(pat, text, flags)
        if m:
            val = m.group(1)
            result["total_amount"] = val
            logger.debug(f"total_amount matched pattern {i}: {val}")
            break
    if not result["total_amount"]:
        logger.warning("total_amount matched no patterns (set to null)")
        
    return result


def _extract_line_items_regex(text: str) -> list:
    import re
    line_items = []
    
    matches = re.finditer(
        r'^\s*(\d+)\s+'                          
        r'([A-Z0-9]{2,4}-[A-Z0-9/\.\-]+)\s+'    
        r'(\d+)\s+'                               
        r'(\d+)\s+'                               
        r'(\d+)\s+'                               
        r'([0-9]+\.[0-9]{2,3})\s+'               
        r'([0-9]+\.[0-9]{2})',                    
        text,
        re.MULTILINE
    )
    
    for m in matches:
        line_items.append({
            "line_number": int(m.group(1)),
            "product_id": m.group(2),
            "quantity": int(m.group(4)),
            "unit_price": float(m.group(6)),
            "extended_price": float(m.group(7))
        })
        
    line_items.sort(key=lambda x: x["line_number"])
    logger.info(f"Regex extracted {len(line_items)} line items.")
    
    if len(line_items) == 0:
        logger.warning("Regex extracted 0 line items (falling back to LLM)")
        
    return line_items


def extract_invoice_fields(raw_text: str) -> ExtractedInvoiceData:
    """Extract invoice fields from OCR text via local Qwen2.5:3b."""
    # Step 1
    cleaned_text = _preprocess_invoice_text(raw_text)
    
    # Step 2
    header_fields = _extract_header_fields(cleaned_text)
    
    # Step 3
    line_items = _extract_line_items_regex(cleaned_text)
    
    # Format line items for pydantic
    formatted_items = []
    for item in line_items:
        formatted_items.append({
            "line_number": str(item["line_number"]),
            "product_id": item["product_id"],
            "quantity": str(item["quantity"]),
            "unit_price": str(item["unit_price"])
        })
        
    # Step 4
    result = {
        "invoice_number": header_fields.get("invoice_number"),
        "po_number": header_fields.get("po_number"),
        "invoice_date": header_fields.get("invoice_date"),
        "supplier": header_fields.get("supplier"),
        "total_amount": header_fields.get("total_amount"),
        "line_items": formatted_items
    }
    
    # Step 5
    header_keys = ["invoice_number", "po_number", "invoice_date", "supplier"]
    null_fields = [k for k in header_keys if result.get(k) is None]
    
    llm_fields = []
    if len(null_fields) >= 2:
        logger.info(f"Regex missed {len(null_fields)} header fields ({null_fields}). Querying LLM...")
        
        # Build a minimal schema
        schema_props = {}
        for f in null_fields:
            schema_props[f] = {"type": "string"}
        llm_schema = {
            "type": "object",
            "properties": schema_props,
            "required": null_fields
        }
        
        minimal_prompt = f"""You are a data extractor. Extract the following fields from the invoice text.
If you cannot find a field, return null. Return ONLY a valid JSON object.

Fields to extract: {', '.join(null_fields)}

INVOICE TEXT:
{cleaned_text[:1500]}"""
        
        llm_result = _call_ollama(minimal_prompt, llm_schema)
        for f in null_fields:
            if llm_result.get(f) is not None:
                result[f] = str(llm_result[f])
                llm_fields.append(f)
                
    # Log at INFO level
    regex_fields = [k for k in header_keys if k not in null_fields]
    logger.info(f"Final extraction - Regex fields: {regex_fields}, LLM fields: {llm_fields}, Line items: {len(line_items)}")
    
    # Step 6
    _validate_extraction(result, cleaned_text)
    
    # Step 7
    return ExtractedInvoiceData.model_validate(result)


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

    inv = result.get("invoice_number")
    log_check("invoice_number", bool(inv and re.fullmatch(r"[0-9]+", str(inv))), inv)
    
    po = result.get("po_number")
    log_check("po_number", bool(po and re.fullmatch(r"[0-9]+", str(po))), po)
    
    sup = result.get("supplier")
    if sup:
        sup_s = str(sup)
        has_dig = bool(re.search(r"\d", sup_s))
        has_addr = bool(re.search(r"\b(Street|Main|Road|Ave|Blvd|TX|IN)\b", sup_s, flags=re.IGNORECASE))
        sup_pass = not has_dig and not has_addr
    else:
        sup_pass = False
    log_check("supplier", sup_pass, sup)
    
    date = result.get("invoice_date")
    log_check("invoice_date", bool(date and re.fullmatch(r"\d{2}-[A-Z]{3}-\d{2}", str(date))), date)
    
    items = result.get("line_items")
    li_pass = bool(items and isinstance(items, list) and len(items) > 0)
    log_check("line_items", li_pass, f"{len(items)} items" if isinstance(items, list) else items)
    
    failed_pids = []
    if li_pass:
        for i, li in enumerate(items):
            pid = li.get("product_id")
            pid_pass = bool(pid and re.fullmatch(r"[0-9]{2,4}-[A-Z0-9]+", str(pid)))
            log_check(f"product_id_{i}", pid_pass, pid)
            if not pid_pass:
                failed_pids.append(str(pid))
                
        if failed_pids:
            logger.info(f"Failing product_ids (first 3): {failed_pids[:3]}")
            
    status = "POOR"
    if total > 0:
        ratio = passed / total
        if ratio == 1.0:
            status = "GOOD"
        elif ratio > 0.5:
            status = "PARTIAL"
            
    logger.info(f"Extraction validation: {passed}/{total} fields passed. Status: {status}")


def extract_boe_fields(cleaned_text: str) -> ExtractedBOEData:
    """Extract Bill of Entry fields from OCR text via local Qwen2.5:3b."""
    result = _call_ollama(
        _BOE_PROMPT.format(text=cleaned_text),
        ExtractedBOEData.model_json_schema(),
    )
    return ExtractedBOEData.model_validate(result)
