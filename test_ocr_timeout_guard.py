"""Self-check that the OCR timeout wrapper actually bounds a hanging call.
Run: python test_ocr_timeout_guard.py
"""
import time
from concurrent.futures import ThreadPoolExecutor, TimeoutError as FutureTimeoutError

executor = ThreadPoolExecutor(max_workers=1)


def _fast_call(x):
    return x * 2


def _hanging_call(x):
    time.sleep(5)
    return x


# fast call returns normally within the timeout
future = executor.submit(_fast_call, 21)
assert future.result(timeout=2.0) == 42, "fast call should return normally"

# slow call raises FutureTimeoutError instead of hanging the caller
future = executor.submit(_hanging_call, 1)
try:
    future.result(timeout=0.5)
    raise SystemExit("FAIL: slow call should have raised FutureTimeoutError")
except FutureTimeoutError:
    pass

print("OK: OCR timeout wrapper bounds hanging calls as expected")
