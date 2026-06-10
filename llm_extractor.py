"""LLM-based structured field extraction via Groq API."""

import json
import logging
import os
import re
from urllib import error as urllib_error
from urllib import request as urllib_request

logger = logging.getLogger(__name__)

GROQ_API_KEY = os.getenv("GROQ_API_KEY")
GROQ_MODEL = os.getenv("GROQ_MODEL", "llama-3.1-8b-instant")


def build_invoice_prompt(cleaned_text: str) -> str:
    return f"""You are a commercial document data extraction assistant.
The document may be a standard invoice, commercial invoice, proforma invoice,
tax invoice, or purchase order. Field labels and layouts vary widely between
vendors and countries — use context clues to identify the correct values
regardless of the exact label used.

Respond ONLY with a valid JSON object. No explanation. No markdown. No code fences. Just raw JSON.

=== LAYOUT WARNINGS — READ CAREFULLY ===

1. TWO-COLUMN HEADERS: Bill To and Ship To addresses may appear side by side on the same lines.
   The LEFT side is Bill To / customer address. The RIGHT side is Ship To / delivery address.

2. FOOTER-LOCATED FIELDS: Invoice number and date may appear in page footers, not headers.
   Look for patterns like "Inv # XXXX", "Invoice No. XXXX", "Dt. DD-MON-YY" anywhere in the text.

3. REPEATED BOILERPLATE: Multi-page invoices repeat headers/footers on every page.
   Extract values ONCE — do not duplicate.

4. LINE ITEM FORMAT (3-line pattern):
   Line 1: [line_no] [part_number] [qty_ordered] [qty_shipped] [qty_pending] [unit_price] [extended_price]
   Line 2: MFG Part No: [manufacturer_part_number]
   Line 3: [manufacturer] [description] / [category]
   Treat all 3 lines as ONE line item.

=== FIELD EXTRACTION RULES ===

- vendor_name: Seller, Exporter, Supplier, company letterhead at top
- vendor_address: address of the seller/vendor
- vendor_email: email of seller/vendor
- vendor_phone: phone of seller/vendor
- customer_name: Consignee, Buyer, Bill To name, Sold To
- customer_address: buyer address (left column if two-column layout)
- billing_address: explicit billing address if different from customer
- shipping_address: delivery address (right column if two-column layout)
- invoice_number: Invoice No, INV#, Inv #, Document No — CHECK FOOTERS
- invoice_date: Invoice Date, Date, Dt. — CHECK FOOTERS — format as YYYY-MM-DD
- due_date: Due Date, Payment Due, Pay By — format as YYYY-MM-DD
- po_number: PO, PO#, Purchase Order No, Order No
- payment_terms: Net 30, Due on Receipt, etc.
- currency: USD, INR, EUR, etc.
- subtotal: amount before tax and shipping (number only)
- tax_amount: tax charged (number only)
- tax_rate: percentage (number only, e.g. 18 not 18%)
- discount_amount: discount applied (number only)
- shipping_amount: freight/shipping charge (number only)
- grand_total: final total — look for "Please pay this amount", "Total Due", "Grand Total", "Amount Due" (number only)

Line items array — each object:
- line_number, part_number, manufacturer_part_number, description
- quantity: quantity SHIPPED (not ordered)
- unit_price: price per unit (number only)
- total_price: line total (number only)

=== GENERAL RULES ===
- Missing field = null, never guess
- Numbers: no currency symbols, no commas
- Dates: YYYY-MM-DD format
- Extract ALL line items found

DOCUMENT TEXT:
{cleaned_text}"""


def call_groq(prompt: str) -> str:
    """Call Groq API and return raw response text."""
    if not GROQ_API_KEY:
        raise RuntimeError("GROQ_API_KEY environment variable not set")

    headers = {
        "Content-Type": "application/json",
        "Authorization": f"Bearer {GROQ_API_KEY}"
    }

    payload = {
        "model": GROQ_MODEL,
        "messages": [{"role": "user", "content": prompt}],
        "temperature": 0.0,
        "max_tokens": 4096,
    }

    req = urllib_request.Request(
        "https://api.groq.com/openai/v1/chat/completions",
        data=json.dumps(payload).encode("utf-8"),
        headers=headers,
        method="POST",
    )

    with urllib_request.urlopen(req, timeout=60) as response:
        body = response.read().decode("utf-8")

    try:
        data = json.loads(body)
    except json.JSONDecodeError as exc:
        raise RuntimeError(f"Groq returned invalid JSON: {exc}") from exc

    if "choices" not in data or not data["choices"]:
        raise RuntimeError(f"Groq response missing choices: {data}")

    return data["choices"][0]["message"]["content"]


def parse_llm_response(response_text: str) -> dict | None:
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


def extract_invoice_fields(ocr_text: str) -> dict | None:
    """Full extraction pipeline: prompt -> Groq -> parse."""
    prompt = build_invoice_prompt(ocr_text)
    try:
        raw_response = call_groq(prompt)
        cleaned = re.sub(r"```json|```", "", raw_response)
        start = cleaned.find('{')
        end = cleaned.rfind('}')
        if start != -1 and end != -1 and end >= start:
            cleaned = cleaned[start:end + 1]
        return json.loads(cleaned)
    except urllib_error.HTTPError as e:
        raise ConnectionError(f"Groq HTTP error: {e}") from e
    except urllib_error.URLError as e:
        raise ConnectionError(f"Groq connection error: {e}") from e
    except json.JSONDecodeError as e:
        snippet = raw_response[:300] if 'raw_response' in locals() else ''
        raise ValueError(f"LLM JSON decode error: {e}. Snippet: {snippet}") from e
    except Exception as e:
        logger.error(f"Extraction failed: {e}")
        raise
