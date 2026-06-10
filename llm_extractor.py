"""LLM-based structured field extraction via Ollama."""

import json
import logging
import os
import re
from urllib import error as urllib_error
from urllib import request as urllib_request

logger = logging.getLogger(__name__)

OLLAMA_BASE_URL = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")
OLLAMA_MODEL = os.getenv("OLLAMA_MODEL", "gemma3")


def build_invoice_prompt(cleaned_text: str) -> str:
    return f"""You are a commercial document data extraction assistant.
The document may be a standard invoice, commercial invoice, proforma invoice,
tax invoice, or purchase order. Field labels and layouts vary widely between
vendors and countries — use context clues to identify the correct values
regardless of the exact label used.

Respond ONLY with a valid JSON object. No explanation. No markdown. No code fences. Just raw JSON.

=== LAYOUT WARNINGS — READ CAREFULLY ===

This text was extracted from a PDF and may have layout artifacts:

1. TWO-COLUMN HEADERS: Bill To and Ship To addresses may appear side by side
   on the same lines, merged together by the PDF extractor. The LEFT side is
   the Bill To / customer address. The RIGHT side is the Ship To / delivery address.
   Separate them carefully based on context (different postcodes, ATTN names etc).

2. FOOTER-LOCATED FIELDS: Invoice number and invoice date may NOT appear at the
   top of the document. Look for them in page footers or sidebars. Common patterns:
   "Inv # 4408466731", "Invoice No. XXXX", "Dt. 21-JUN-23", "Invoice Date: XXXX".
   Search the ENTIRE text including footers, not just the top section.

3. REPEATED BOILERPLATE: Multi-page invoices repeat headers and footers on every
   page. The same address block, payment instructions, and terms may appear many
   times. Extract these values ONCE — do not duplicate them.

4. LINE ITEM FORMAT (3-line pattern): Each line item may span 3 lines like this:
   Line 1: [line_no] [part_number] [qty_ordered] [qty_shipped] [qty_pending] [unit_price] [extended_price]
   Line 2: MFG Part No: [manufacturer_part_number]
   Line 3: [manufacturer] [description] / [category]
   Treat all 3 lines as ONE line item. The description is on line 3.

=== FIELD EXTRACTION RULES ===

- vendor_name: look for Seller, Exporter, From, Supplier, Issued By, company letterhead at top
- vendor_address: address associated with the seller/vendor
- vendor_email: email of seller/vendor
- vendor_phone: phone of seller/vendor
- customer_name: look for Consignee, Buyer, Bill To, Ship To, Sold To, recipient name
- customer_address: address associated with the buyer/consignee (left column if two-column layout)
- billing_address: explicit billing address if different from customer address
- shipping_address: explicit shipping/delivery address if present (right column if two-column layout)
- invoice_number: look for Invoice No, INV#, Inv #, Invoice Number, Reference No, Document No — CHECK FOOTERS
- invoice_date: look for Invoice Date, Date, Issued Date, Dt. — CHECK FOOTERS
- due_date: look for Due Date, Payment Due, Pay By
- po_number: look for PO, PO#, Purchase Order, Order No, Reference
- payment_terms: look for Terms, Payment Terms, Net 30, Due on Receipt
- currency: look for Currency, USD, INR, EUR symbol
- subtotal: amount before tax and shipping
- tax_amount: tax charged
- tax_rate: percentage rate of tax
- discount_amount: any discount applied
- shipping_amount: freight or shipping charge
- grand_total: final total amount due — look for "Please pay this amount", "Total Due", "Grand Total", "Amount Due"

Line items array — each object must have:
- line_number: the line/item number
- part_number: the vendor's part number (e.g. Mouser part number)
- manufacturer_part_number: the MFG Part No if present
- description: product description
- quantity: quantity shipped (not ordered or pending)
- unit_price: price per unit
- total_price: line total (quantity x unit_price)

=== GENERAL RULES ===

- If a field is not present in the document, use null
- Never guess or hallucinate values
- For numbers, return only the numeric value (no currency symbols, no commas)
- For dates, convert to YYYY-MM-DD format if possible
- If the document has multiple PO numbers, put them all comma-separated in po_number
- For large invoices with many line items, extract ALL line items you can find

DOCUMENT TEXT:
{cleaned_text}"""


def call_ollama(prompt: str) -> str:
   
    """Call Ollama API and return raw response text."""
    headers = {
        "Content-Type": "application/json",
        "ngrok-skip-browser-warning": "true"
        }
    url = f"{OLLAMA_BASE_URL}/api/generate"
    payload = {
        "model": OLLAMA_MODEL,
        "prompt": prompt,
        "stream": False,
        "options": {
            "num_ctx": 32768,
            "num_predict": 4096,
        },
    }

    request = urllib_request.Request(
        url,
        data=json.dumps(payload).encode("utf-8"),
        headers=headers,
        method="POST",
    )

    with urllib_request.urlopen(request, timeout=300) as response:
        response_body = response.read().decode("utf-8")

    try:
        data = json.loads(response_body)
    except json.JSONDecodeError as exc:
        raise RuntimeError(f"Ollama returned an invalid JSON response: {exc}") from exc

    if not isinstance(data, dict) or "response" not in data:
        raise RuntimeError("Ollama returned an invalid response payload: missing 'response' field")

    response_text = data["response"]
    if not isinstance(response_text, str):
        raise RuntimeError("Ollama returned an invalid response payload: 'response' is not a string")

    return response_text


def parse_llm_response(response_text: str) -> dict | None:
    """Parse JSON from LLM response, handling stray text."""
    if not response_text:
        return None

    try:
        return json.loads(response_text.strip())
    except json.JSONDecodeError:
        pass

    start = response_text.find('{')
    end = response_text.rfind('}')
    if start != -1 and end != -1 and end > start:
        try:
            return json.loads(response_text[start:end + 1])
        except json.JSONDecodeError:
            pass

    logger.error(f"Failed to parse LLM response: {response_text[:500]}")
    return None





def _clean_llm_response(resp: str) -> str:
    """Remove markdown fences and extract the JSON block.
    Handles optional ```json fences.
    """
    # Strip surrounding markdown fences
    cleaned = resp.strip()
    # Remove leading/trailing ``` or ```json fences
    cleaned = re.sub(r"^```(?:json)?\s*", "", cleaned, flags=re.MULTILINE)
    cleaned = re.sub(r"\s*```$", "", cleaned, flags=re.MULTILINE)
    # Find the first '{' and the last '}'
    start = cleaned.find('{')
    end = cleaned.rfind('}')
    if start != -1 and end != -1 and end > start:
        return cleaned[start : end + 1]
    return cleaned


def extract_invoice_fields(ocr_text: str) -> dict | None:
    """Full extraction pipeline: prompt -> LLM -> parse with robust handling."""

    prompt = build_invoice_prompt(ocr_text)
    try:
        raw_response = call_ollama(prompt)
        
        # Strip markdown fences
        cleaned_response = re.sub(r"```json|```", "", raw_response)
        start = cleaned_response.find('{')
        end = cleaned_response.rfind('}')
        if start != -1 and end != -1 and end >= start:
            cleaned_response = cleaned_response[start:end+1]

        result = json.loads(cleaned_response)
        return result
    except urllib_error.HTTPError as e:
        raise ConnectionError(f"Ollama HTTP error: {e}") from e
    except urllib_error.URLError as e:
        raise ConnectionError(f"Ollama connection error: {e}") from e
    except json.JSONDecodeError as e:
        snippet = raw_response[:300] if 'raw_response' in locals() else ''
        raise ValueError(f"LLM JSON decode error: {e}. Response snippet: {snippet}") from e
    except Exception as e:
        logger.error(f"Extraction failed: {e}")
        raise
