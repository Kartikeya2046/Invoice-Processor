"""Self-check for the numeric-pattern guard on LineItem/ExtractedInvoiceData
quantity/unit_price fields. Run: python test_numeric_field_guard.py
"""
from pydantic import ValidationError

from models import ExtractedInvoiceData, LineItem

# valid numeric strings pass
LineItem(quantity="12.50", unit_price="500")
ExtractedInvoiceData(quantity="1000", unit_price="55.70")

# null still allowed (missing field)
LineItem(quantity=None, unit_price=None)

# hallucinated prose is rejected
prose = "bytes not shown in the document, but assumed to be 12.50 based on PO number VIZ5077"
try:
    LineItem(quantity=prose)
    raise SystemExit("FAIL: prose string should have been rejected")
except ValidationError:
    pass

try:
    ExtractedInvoiceData(unit_price=prose)
    raise SystemExit("FAIL: prose string should have been rejected on header field")
except ValidationError:
    pass

# comma/currency-formatted numbers still rejected (per the extraction rules --
# these should have been cleaned before this point, not silently accepted)
try:
    LineItem(unit_price="$1,000.00")
    raise SystemExit("FAIL: currency-formatted string should have been rejected")
except ValidationError:
    pass

print("OK: numeric field guard behaves as expected")
