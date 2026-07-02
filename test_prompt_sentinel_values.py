"""Regression guard: the few-shot example in _INVOICE_PROMPT must use
obviously-fake sentinel values (9999 / ZZ-9999-FAKE), never realistic-looking
ones -- documented failure mode is small models copying literal example
values into real output when uncertain.
Run: python test_prompt_sentinel_values.py
"""
from llm_extractor import _INVOICE_PROMPT

assert "ZZ-9999-FAKE" in _INVOICE_PROMPT, "sentinel product_id example missing"
assert "9999" in _INVOICE_PROMPT, "sentinel qty/price example missing"
assert "never output these" in _INVOICE_PROMPT, (
    "example must warn the model not to copy these values"
)
assert "RMB" not in _INVOICE_PROMPT, (
    "example must not use a realistic-looking part code pattern"
)

print("OK: prompt example uses fake sentinel values as intended")
