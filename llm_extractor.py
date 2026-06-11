"""LLM-based structured field extraction via Mistral AI API."""

import json
import logging
import os
import re
from urllib import error as urllib_error
from urllib import request as urllib_request
import time

logger = logging.getLogger(__name__)


def call_mistral(prompt: str) -> str:
    """Call Mistral API and return raw response text. Retries on 429."""
    MISTRAL_API_KEY = os.getenv("MISTRAL_API_KEY")
    MISTRAL_MODEL = os.getenv("MISTRAL_MODEL", "mistral-small-latest")

    if not MISTRAL_API_KEY:
        raise RuntimeError("MISTRAL_API_KEY environment variable not set")

    url = "https://api.mistral.ai/v1/chat/completions"

    payload = {
        "model": MISTRAL_MODEL,
        "messages": [
            {"role": "user", "content": prompt}
        ],
        "temperature": 0.0,
        "max_tokens": 32000,
    }

    MAX_RETRIES = 3
    RETRY_DELAYS = [15, 30, 60]
    last_error = None

    for attempt in range(MAX_RETRIES):
        try:
            req = urllib_request.Request(
                url,
                data=json.dumps(payload).encode("utf-8"),
                headers={
                    "Content-Type": "application/json",
                    "Authorization": f"Bearer {MISTRAL_API_KEY}",
                },
                method="POST",
            )
            with urllib_request.urlopen(req, timeout=60) as response:
                body = response.read().decode("utf-8")

            try:
                data = json.loads(body)
            except json.JSONDecodeError as exc:
                raise RuntimeError(f"Mistral returned invalid JSON: {exc}") from exc

            try:
                return data["choices"][0]["message"]["content"]
            except (KeyError, IndexError) as exc:
                raise RuntimeError(f"Mistral response missing expected fields: {data}") from exc

        except urllib_error.HTTPError as e:
            if e.code == 429 and attempt < MAX_RETRIES - 1:
                wait = RETRY_DELAYS[attempt]
                logger.warning(f"Mistral 429 rate limit hit, retrying in {wait}s (attempt {attempt + 1}/{MAX_RETRIES})")
                time.sleep(wait)
                last_error = e
                continue
            raise ConnectionError(f"Mistral HTTP error: {e}") from e

    raise ConnectionError(f"Mistral rate limit exceeded after {MAX_RETRIES} attempts: {last_error}")


def build_invoice_prompt(cleaned_text: str) -> str:
    return f"""You are a commercial document data extraction assistant.
The document may be a standard invoice, commercial invoice, proforma invoice,
tax invoice, or purchase order.

CRITICAL: Field labels vary enormously across vendors, countries, and document layouts.
You MUST scan the entire document for any label that could reasonably refer to a field —
do NOT skip a field just because its label does not exactly match the examples below.
Common variations to watch for:
- Invoice number: "Invoice No", "Inv No", "Inv #", "INV", "Invoice #", "Document No", "Doc No", "Bill No", "Ref No", "Reference"
- Invoice date: "Invoice Date", "Date", "Dt", "Dated", "Issue Date", "Billing Date", "Tax Invoice Date"
- Due date: "Due Date", "Payment Due", "Pay By", "Due On", "Net Due Date", "Expiry"
- Vendor: "Seller", "Supplier", "From", "Issued By", "Exporter", "Billed From", company letterhead at top of document
- Customer: "Bill To", "Sold To", "Buyer", "Consignee", "Client", "Ship To" (if no separate billing)
- PO number: "PO", "PO#", "Purchase Order", "Order No", "Order #", "Cust PO", "Your Order"
- Grand total: "Total Due", "Amount Due", "Total Payable", "Balance Due", "Please Pay", "Net Payable", "Invoice Total"
- Subtotal: "Sub Total", "Net Amount", "Taxable Amount", "Amount Before Tax"
- Tax: "GST", "VAT", "HST", "Tax", "CGST", "SGST", "IGST", "Service Tax"
- Payment terms: "Terms", "Net 30", "Due on Receipt", "Payment Terms", "Credit Period"

If a value is present in the document but under an unlisted label, use context and position to identify it and extract it anyway. Missing field = null, never guess.

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
- grand_total: final total — "Please pay this amount", "Total Due", "Grand Total", "Amount Due" (number only)

Line items array — each object:
- line_number, part_number, manufacturer_part_number, description
- quantity: quantity SHIPPED (not ordered)
- unit_price: price per unit (number only)
- total_price: line total (number only)

=== GENERAL RULES ===
IMPORTANT — RESPONSE LENGTH: If the document has many line items and you are at risk of
hitting a token limit, prioritize the header fields first (all fields except line_items),
then include as many line items as fit. Never truncate a JSON string mid-value — always
close all brackets and braces to produce valid JSON. If you must stop early on line items,
close the line_items array and the root object properly.

- Missing field = null, never guess
- Numbers: no currency symbols, no commas
- Dates: YYYY-MM-DD format
- Extract ALL line items found

DOCUMENT TEXT:
{cleaned_text}"""


def extract_invoice_fields(ocr_text: str) -> dict | None:
    """Full extraction pipeline: prompt -> Mistral -> parse."""
    prompt = build_invoice_prompt(ocr_text)
    try:
        raw_response = call_mistral(prompt)
        cleaned = re.sub(r"```json|```", "", raw_response)
        start = cleaned.find('{')
        end = cleaned.rfind('}')
        if start != -1 and end != -1 and end >= start:
            cleaned = cleaned[start:end + 1]
        return json.loads(cleaned)
    except urllib_error.HTTPError as e:
        raise ConnectionError(f"Mistral HTTP error: {e}") from e
    except urllib_error.URLError as e:
        raise ConnectionError(f"Mistral connection error: {e}") from e
    except json.JSONDecodeError as e:
        snippet = raw_response[:300] if 'raw_response' in locals() else ''
        raise ValueError(f"LLM JSON decode error: {e}. Snippet: {snippet}") from e
    except Exception as e:
        logger.error(f"Extraction failed: {e}")
        raise
