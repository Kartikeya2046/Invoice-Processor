import requests
import time
import os
from pathlib import Path
import json
import traceback

print("STEP 1 — Verify all required processes are running")
# 1. Ollama
try:
    r = requests.get("http://localhost:11434")
    print(f"Ollama running: YES (status {r.status_code})")
except Exception as e:
    print(f"Ollama running: NO ({e})")
    print("Command to start: ollama serve")

# 2. Backend
try:
    r = requests.get("http://localhost:8001")
    print(f"Backend running: YES (status {r.status_code})")
except Exception as e:
    print(f"Backend running: NO ({e})")
    print('Command to start: $env:PATH = "D:\\poppler\\poppler-24.02.0\\Library\\bin;D:\\llama.cpp\\llama-b9844-bin-win-cuda-13.3-x64;" + $env:PATH; uvicorn main:app --reload --host 0.0.0.0 --port 8001')

# 3. Frontend
try:
    r = requests.get("http://localhost:8080")
    print(f"Frontend running: YES (status {r.status_code})")
except Exception as e:
    print(f"Frontend running: NO ({e})")
    print("Command to start: python -m http.server 8080")

print("\nSTEP 2 — Verify the backend health endpoint")
try:
    r = requests.get("http://localhost:8001/health")
    if r.status_code != 404:
        print("Response from /health:", r.text)
    else:
        r = requests.get("http://localhost:8001/docs")
        if r.status_code != 404:
            print("Response from /docs (truncated):", r.text[:200])
        else:
            r = requests.get("http://localhost:8001/")
            print("Response from /:", r.text)
except Exception:
    print("Backend unreachable, skipping Step 2 details.")

print("\nSTEP 3 — Run a live upload test against the actual running backend")
uploads_dir = Path("uploads")
try:
    if not uploads_dir.exists():
        print("uploads directory not found.")
    else:
        pdfs = list(uploads_dir.glob("*.pdf"))
        if not pdfs:
            print("No PDF files found in uploads.")
        else:
            latest_pdf = max(pdfs, key=os.path.getmtime)
            print(f"Testing with: {latest_pdf.name}")
            start_t = time.time()
            with open(latest_pdf, "rb") as f:
                r = requests.post("http://localhost:8001/api/upload", files={"file": f})
            end_t = time.time()
            print(f"HTTP Status Code: {r.status_code}")
            print(f"Time Taken: {end_t - start_t:.2f}s")
            try:
                data = r.json()
                print("Response body:")
                print(json.dumps(data, indent=2))
                if r.status_code == 200 and "extracted_fields" in data:
                    print("\nExtracted Fields:")
                    ext = data["extracted_fields"]
                    for k in ["invoice_number", "po_number", "supplier", "invoice_date"]:
                        print(f"{k}: {ext.get(k)}")
            except Exception as e:
                print("Failed to parse JSON response:", str(e))
                print("Raw text:", r.text)
except Exception as e:
    print("Upload test failed exception:", str(e))
    traceback.print_exc()
