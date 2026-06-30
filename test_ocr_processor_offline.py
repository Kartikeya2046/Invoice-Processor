"""Runnable self-check for ocr_processor.py logic that doesn't need a live Surya server.

Covers: _TableHTMLParser (HTML -> rows) and _find (regex field extraction).
Does NOT cover: extract_text/extract_tables/process_file, which need a running
Surya inference backend. Run those manually against the real Mouser PDF once
the GPU/Docker server is up.
"""
import sys
from unittest import mock

# Stub out surya + pdf2image so this file can run without the real packages
# or a GPU, since we're only testing the pure-Python parsing/regex logic.
sys.modules["surya"] = mock.MagicMock()
sys.modules["surya.inference"] = mock.MagicMock()
sys.modules["surya.recognition"] = mock.MagicMock()
sys.modules["surya.table_rec"] = mock.MagicMock()
sys.modules["pdf2image"] = mock.MagicMock()

from ocr_processor import _TableHTMLParser, _find  # noqa: E402

# --- _TableHTMLParser ---
html = """
<table>
<tr><th>Line No</th><th>Mouser Part No</th><th>Qty</th><th>Unit Price</th></tr>
<tr><td>1</td><td>ABC-123</td><td>10</td><td>2.50</td></tr>
<tr><td>6</td><td>XYZ-789</td><td>3</td><td>1,234.50</td></tr>
</table>
"""
parser = _TableHTMLParser()
parser.feed(html)
assert len(parser.rows) == 3, f"expected 3 rows (1 header + 2 data), got {len(parser.rows)}"
assert parser.rows[0] == ["Line No", "Mouser Part No", "Qty", "Unit Price"]
assert parser.rows[1] == ["1", "ABC-123", "10", "2.50"]
assert parser.rows[2][0] == "6"  # skipped line numbers preserved, not renumbered

# Header row's first cell is non-numeric -> the "first cell isdigit()" filter
# in parse_invoice correctly excludes it without needing extra header-detection code.
assert not parser.rows[0][0].isdigit()
assert parser.rows[1][0].isdigit()

# --- _find regexes used in parse_invoice / parse_boe ---
sample_text = """
Supplier: Mouser Electronics
P.O. Number: PO-998877
Invoice Number: 74294838
Invoice Date: 06/15/2026
CGST: 450.00
"""
assert _find(r"Supplier\s*:?\s*(.+)", sample_text) == "Mouser Electronics"
assert _find(r"P\.?O\.?\s*(?:Number|No\.?)\s*:?\s*([A-Za-z0-9\-]+)", sample_text) == "PO-998877"
assert _find(r"Inv(?:oice)?\.?\s*(?:Number|No\.?|#)\s*:?\s*([A-Za-z0-9\-]+)", sample_text) == "74294838"
assert _find(r"CGST\s*:?\s*([0-9,.]+)", sample_text) == "450.00"
assert _find(r"SGST\s*:?\s*([0-9,.]+)", sample_text) is None  # not present -> None, not "None"

boe_text = "B/E Number: 7842531\nB/E Date: 01/05/2026\nIGST: 1200.00\nSBCESS: 15.00"
assert _find(r"B/E\s*Number\s*:?\s*([A-Za-z0-9\-]+)", boe_text) == "7842531"
assert _find(r"SBCESS\s*:?\s*([0-9,.]+)", boe_text) == "15.00"

print("OK: _TableHTMLParser and _find regexes pass on synthetic input.")
print("NOTE: extract_text/extract_tables/process_file are UNTESTED here —")
print("they need a live Surya server. Run a real upload before trusting them.")
