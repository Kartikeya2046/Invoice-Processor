"""LLM-based structured field extraction via Ollama."""

import json
import logging
import os
from urllib import error as urllib_error
from urllib import request as urllib_request

logger = logging.getLogger(__name__)

OLLAMA_BASE_URL = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")
OLLAMA_MODEL = os.getenv("OLLAMA_MODEL", "gemma3")


def build_invoice_prompt(cleaned_text: str) -> str:
    return f"""You are a document data extraction assistant. Extract structured data from the invoice text below.

Respond ONLY with a valid JSON object. No explanation. No markdown. No code fences. Just raw JSON.

Extract these fields:
- invoice_number: string or null
- invoice_date: string (YYYY-MM-DD format) or null
- due_date: string (YYYY-MM-DD format) or null
- vendor_name: string or null
- vendor_address: string or null
- vendor_email: string or null
- vendor_phone: string or null
- customer_name: string or null
- customer_address: string or null
- billing_address: string or null
- shipping_address: string or null
- po_number: string or null
- payment_terms: string or null
- currency: string (e.g. USD, INR, GBP) or null
- subtotal: number or null
- tax_amount: number or null
- tax_rate: number (percentage) or null
- discount_amount: number or null
- grand_total: number or null
- notes: string or null
- line_items: array of objects, each with:
    - description: string
    - quantity: number
    - unit_price: number
    - total_price: number

Rules:
- If a field is not present in the document, use null
- Never guess or hallucinate values
- For numbers, return only the numeric value (no currency symbols)
- For dates, convert to YYYY-MM-DD format if possible

INVOICE TEXT:
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

    try:
        with urllib_request.urlopen(request, timeout=120) as response:
            response_body = response.read().decode("utf-8")
    except urllib_error.HTTPError as exc:
        error_body = exc.read().decode("utf-8", errors="replace") if exc.fp else str(exc)
        raise RuntimeError(f"Ollama request failed with HTTP {exc.code}: {error_body}") from exc
    except urllib_error.URLError as exc:
        raise RuntimeError(f"Failed to reach Ollama at {url}: {exc}") from exc
    except Exception as exc:
        raise RuntimeError(f"Unexpected Ollama request failure: {exc}") from exc

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


def extract_invoice_fields(cleaned_text: str) -> dict | None:
    """Full extraction pipeline: prompt -> LLM -> parse."""
    try:
        prompt = build_invoice_prompt(cleaned_text)
        raw_response = call_ollama(prompt)
        result = parse_llm_response(raw_response)
        if result is None:
            logger.error("LLM returned unparseable response")
        return result
    except Exception as e:
        logger.error(f"Extraction failed: {e}")
        raise
