"""Regression guard: product_id instruction in the extraction prompt must
stay generic (no vendor-specific wording) and must explicitly exclude the
item description. Run: python test_product_id_prompt_wording.py
"""
from llm_extractor import _INVOICE_PROMPT

assert "mouser" not in _INVOICE_PROMPT.lower(), (
    "product_id instruction regressed to vendor-specific (Mouser) wording"
)
assert "NOT the item description" in _INVOICE_PROMPT, (
    "product_id instruction no longer tells the model to avoid the description"
)

print("OK: product_id prompt wording is generic and excludes description")
