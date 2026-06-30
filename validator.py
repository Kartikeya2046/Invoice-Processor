"""Validation and confidence scoring for extracted invoice / BOE fields.

Simplified from the previous version: that one checked cross-field arithmetic
(qty x unit_price == total_price, subtotal + tax - discount == grand_total)
against fields that don't exist in this schema (po_number/cgst/sgst/boe
fields have no such relationships to check). This version checks required-
field presence and numeric format only.
"""

CONFIDENCE_THRESHOLD = 0.70

INVOICE_REQUIRED_FIELDS = ["invoice_number", "supplier"]
INVOICE_NUMERIC_FIELDS = ["quantity", "unit_price", "cgst", "sgst"]
LINE_ITEM_NUMERIC_FIELDS = ["quantity", "unit_price"]

BOE_REQUIRED_FIELDS = ["boe_number"]
BOE_NUMERIC_FIELDS = ["igst", "cust_duty", "sbcess"]


def _to_float(value) -> float | None:
    if value is None:
        return None
    try:
        return float(str(value).replace(",", "").strip())
    except (ValueError, TypeError):
        return None


def _check_numeric_fields(extracted: dict, numeric_fields: list[str], score: float, failed_checks: list[str]) -> float:
    for field in numeric_fields:
        raw = extracted.get(field)
        if raw is None:
            continue
        val = _to_float(raw)
        if val is None:
            failed_checks.append(f"{field} is not a valid number")
            score -= 0.10
        elif val < 0:
            failed_checks.append(f"{field} is negative")
            score -= 0.05
    return score


def validate_invoice_fields(extracted: dict) -> dict:
    """Validate ExtractedInvoiceData: required fields present, numbers well-formed."""
    missing_fields = []
    failed_checks = []
    score = 1.0

    for field in INVOICE_REQUIRED_FIELDS:
        if not extracted.get(field):
            missing_fields.append(field)
            score -= 0.15

    score = _check_numeric_fields(extracted, INVOICE_NUMERIC_FIELDS, score, failed_checks)

    for i, item in enumerate(extracted.get("line_items") or []):
        score = _check_numeric_fields(item, LINE_ITEM_NUMERIC_FIELDS, score, failed_checks)
        if not item.get("product_id"):
            failed_checks.append(f"Line item {i + 1}: missing product_id")
            score -= 0.05

    score = round(max(0.0, min(1.0, score)), 2)
    return {
        "confidence_score": score,
        "missing_fields": missing_fields,
        "issues": failed_checks,
        "needs_review": score < CONFIDENCE_THRESHOLD or len(missing_fields) > 0,
    }


def validate_boe_fields(extracted: dict) -> dict:
    """Validate ExtractedBOEData: required fields present, numbers well-formed."""
    missing_fields = []
    failed_checks = []
    score = 1.0

    for field in BOE_REQUIRED_FIELDS:
        if not extracted.get(field):
            missing_fields.append(field)
            score -= 0.15

    score = _check_numeric_fields(extracted, BOE_NUMERIC_FIELDS, score, failed_checks)

    score = round(max(0.0, min(1.0, score)), 2)
    return {
        "confidence_score": score,
        "missing_fields": missing_fields,
        "issues": failed_checks,
        "needs_review": score < CONFIDENCE_THRESHOLD or len(missing_fields) > 0,
    }