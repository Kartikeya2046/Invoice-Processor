"""Validation and confidence scoring for extracted invoice fields."""

from datetime import datetime


REQUIRED_FIELDS = ['invoice_number', 'vendor_name', 'grand_total']
NUMERIC_FIELDS = ['subtotal', 'tax_amount', 'grand_total', 'discount_amount']
DATE_FIELDS = ['invoice_date', 'due_date']
CONFIDENCE_THRESHOLD = 0.70


def _to_float(value) -> float | None:
    if value is None:
        return None
    try:
        return float(str(value).replace(',', '').strip())
    except (ValueError, TypeError):
        return None


def _to_date(value) -> datetime | None:
    if not value:
        return None
    for fmt in ('%Y-%m-%d', '%d/%m/%Y', '%m/%d/%Y', '%d-%m-%Y', '%B %d, %Y', '%b %d, %Y'):
        try:
            return datetime.strptime(str(value).strip(), fmt)
        except ValueError:
            continue
    return None


def validate_extracted_fields(extracted: dict) -> dict:
    """Run all validation checks and return a validation result dict."""

    missing_fields = []
    failed_checks = []
    score = 1.0

    for field in REQUIRED_FIELDS:
        if not extracted.get(field):
            missing_fields.append(field)
            score -= 0.15

    numeric_values = {}
    for field in NUMERIC_FIELDS:
        val = _to_float(extracted.get(field))
        numeric_values[field] = val
        if extracted.get(field) is not None and val is None:
            failed_checks.append(f"{field} is not a valid number")
            score -= 0.10
        if val is not None and val < 0:
            failed_checks.append(f"{field} is negative")
            score -= 0.05

    line_items = extracted.get('line_items') or []
    line_item_total = 0.0
    for i, item in enumerate(line_items):
        qty = _to_float(item.get('quantity'))
        unit = _to_float(item.get('unit_price'))
        total = _to_float(item.get('total_price'))
        if qty and unit and total:
            expected = qty * unit
            if abs(expected - total) / max(expected, 0.01) > 0.01:
                failed_checks.append(f"Line item {i+1}: qty x unit_price != total_price")
                score -= 0.05
            line_item_total += total

    subtotal = numeric_values.get('subtotal')
    tax = numeric_values.get('tax_amount') or 0.0
    discount = numeric_values.get('discount_amount') or 0.0
    grand = numeric_values.get('grand_total')

    if subtotal and grand:
        expected_grand = subtotal + tax - discount
        if abs(expected_grand - grand) / max(grand, 0.01) > 0.02:
            failed_checks.append("subtotal + tax - discount != grand_total")
            score -= 0.10

    if subtotal and line_item_total > 0:
        if abs(line_item_total - subtotal) / max(subtotal, 0.01) > 0.02:
            failed_checks.append("Sum of line items != subtotal")
            score -= 0.10

    invoice_date = _to_date(extracted.get('invoice_date'))
    due_date = _to_date(extracted.get('due_date'))

    for field, val_raw in [('invoice_date', extracted.get('invoice_date')),
                            ('due_date', extracted.get('due_date'))]:
        if val_raw and _to_date(val_raw) is None:
            failed_checks.append(f"{field} is not a valid date")
            score -= 0.05

    if invoice_date and due_date and due_date < invoice_date:
        failed_checks.append("due_date is before invoice_date")
        score -= 0.05

    score = round(max(0.0, min(1.0, score)), 2)
    needs_review = score < CONFIDENCE_THRESHOLD or len(missing_fields) > 0

    return {
        'is_valid': len(failed_checks) == 0 and len(missing_fields) == 0,
        'confidence_score': score,
        'missing_fields': missing_fields,
        'failed_checks': failed_checks,
        'needs_review': needs_review,
    }