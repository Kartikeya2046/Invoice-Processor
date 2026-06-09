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

Extraction rules for varied formats:
- vendor_name: look for Seller, Exporter, From, Supplier, Issued By, company letterhead at top
- vendor_address: address associated with the seller/exporter
- vendor_email: email of seller/exporter
- vendor_phone: phone of seller/exporter
- customer_name: look for Consignee, Buyer, Bill To, Ship To, Sold To, recipient name
- customer_address: address associated with the buyer/consignee
- billing_address: explicit billing address if different from customer address
- shipping_address: explicit shipping/delivery address if present
- invoice_number: look for Invoice No, INV#, Invoice Number, Reference No, Document No, No., Inv#
- invoice_date: look for Invoice Date, Date, Issued Date, Document Date
- due_date: look for Due Date, Payment Due, Pay By
- po_number: look for PO, PO#, Purchase Order, Order No, Reference
- payment_terms: look for Terms, Payment Terms, Terms of Payment, Terms of Delivery and Payment
- currency: infer from symbols or explicit currency codes (USD, INR, GBP, KRW etc.)
- subtotal: amount before tax/freight, may be labelled Subtotal, Merchandise, Net Amount
- tax_amount: any tax, GST, VAT, or duty amount
- tax_rate: tax percentage if shown
- discount_amount: any discount applied
- grand_total: look for Grand Total, Total Amount Due, Amount Due, Total USD, Please Pay, 
  final payable amount — this is the most important field
- notes: any remarks, special instructions, or comments
- line_items: extract ALL line items as an array; each item should have:
    - description: product name, part number, or service description
    - quantity: numeric quantity ordered or shipped
    - unit_price: price per unit
    - total_price: line total (quantity x unit_price)

Additional rules:
- If a field is not present in the document, use null
- Never guess or hallucinate values
- For numbers, return only the numeric value (no currency symbols or commas)
- For dates, convert to YYYY-MM-DD format if possible
- If the document has multiple PO numbers, put them all comma-separated in po_number
- For 16+ page invoices with many line items, extract as many line items as you can fit

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
        "stream": False
    }

    request = urllib_request.Request(
        url,
        data=json.dumps(payload).encode("utf-8"),
        headers=headers,
        method="POST",
    )

    with urllib_request.urlopen(request, timeout=120) as response:
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


def _truncate_text(text: str, max_len: int = 6000) -> str:
    """Keep first 3000 and last 3000 chars if text exceeds max_len.
    Inserts a marker in the middle.
    """
    if len(text) <= max_len:
        return text
    part = max_len // 2
    return text[:part] + "\n\n...[middle truncated]...\n\n" + text[-part:]


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
    if len(ocr_text) > 6000:
        ocr_text = ocr_text[:3000] + "\n\n...[middle truncated]...\n\n" + ocr_text[-3000:]

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
