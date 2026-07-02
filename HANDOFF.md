SECTION 1 — PROJECT OVERVIEW
One paragraph describing what this system does: accepts invoice PDF uploads via a FastAPI backend, extracts structured data (invoice number, PO number, supplier, date, line items), and returns a JSON response.

SECTION 2 — ARCHITECTURE
Describe the current pipeline in this exact order with one paragraph per step:
1. Upload handler (routes/upload.py) — receives the file, saves it to uploads/, calls process_file()
2. PDF text extraction (ocr_processor.py) — first tries pdfplumber embedded text extraction via _try_extract_embedded_text(). If the PDF has embedded text (character count > 100 * page count), returns it immediately. If not (scanned PDF), falls through to Surya OCR per-page with 120s timeout per page and predictor rebuild on timeout.
3. Text preprocessing (llm_extractor.py _preprocess_invoice_text()) — strips repeating boilerplate blocks that appear on every page (wire transfer details, footer, shipping info), reducing token count by ~50%.
4. Regex extraction (_extract_header_fields() and _extract_line_items_regex()) — extracts all structured fields deterministically. Header fields use multi-pattern regex with fallbacks. Line items use a single re.finditer() pass over the full text.
5. Conditional LLM fallback — only called if 2 or more header fields are null after regex. Sends targeted 800-character context snippets for only the null fields to qwen2.5:3b via Ollama. Never overwrites regex-extracted values.
6. Validation layer (_validate_extraction()) — logs PASS/FAIL per field after every extraction for observability.

SECTION 3 — PERFORMANCE BENCHMARKS (from the most recent pipeline_full_test.py run)
- PDF embedded text extraction: 1.60s for 16 pages, 37,035 characters
- Text preprocessing: 0.01s, 49.5% reduction (37,035 → 18,703 characters)
- Regex header extraction: instant, all 4 header fields correct
- Regex line item extraction: instant, 92 line items extracted with 100% accuracy
- LLM call: skipped entirely for this invoice (all fields found by regex)
- Total end-to-end: ~2s for a 16-page Mouser invoice

SECTION 4 — KNOWN ISSUES AND LIMITATIONS
List these exactly:
1. Scanned PDFs: Surya OCR on CPU is too slow for production use. Per-page 120s timeout and predictor rebuild on wedge are in place, but real fix requires GPU inference. RTX 3050 is present and CUDA build of llama.cpp is on PATH — needs validation that Surya is actually dispatching to CUDA.
2. Regex fragility: The current regex patterns are tuned for Mouser Electronics invoice format. A different supplier's invoice layout will likely break header extraction and fall through to LLM fallback. The LLM fallback needs testing against non-Mouser invoices before go-live.
3. Line item regex assumption: The pattern assumes Mouser's fixed column order (line_number, mouser_part, qty_ordered, qty_shipped, qty_pending, unit_price, extended_price). Any supplier that omits a column or reorders them will produce wrong quantity or price values silently.
4. Port binding: Server must be started on port 8001, not 8000. Port 8000 falls in the Windows reserved port range due to Hyper-V/WSL2. The startup command in Section 6 already reflects this.
5. reportlab not installed: Causes a harmless FAIL in Stage 1 of pipeline_full_test.py. Not needed in production.

SECTION 5 — OPEN QUESTIONS FOR NEXT AGENT
List these exactly:
1. Is Surya actually using the RTX 3050 via CUDA, or silently falling back to CPU? Check the startup log for the torch.cuda.is_available() line added to ocr_processor.py. If it says CPU, fix CUDA dispatch before testing any scanned PDF uploads.
2. What other invoice formats does this system need to support beyond Mouser? Each new supplier format needs regex pattern additions or LLM prompt tuning.
3. The frontend shows no error in the console during a long OCR timeout — the fetch() call just hangs silently. Should there be a polling mechanism or a progress indicator for uploads that take longer than 5 seconds?
4. The uploads/ directory is never cleaned up. Every uploaded PDF stays on disk indefinitely. Is there a retention policy needed?

SECTION 6 — STARTUP COMMANDS
Include the exact commands to start the system, copy them verbatim:

Backend:
    cd D:\Downloads\intern\assignment_1_frontend_backend\invoice-processor
    $env:PATH = "D:\poppler\poppler-24.02.0\Library\bin;D:\llama.cpp\llama-b9844-bin-win-cuda-13.3-x64;" + $env:PATH; uvicorn main:app --reload --host 0.0.0.0 --port 8001

Ollama:
    ollama serve
    ollama run qwen2.5:3b

Frontend:
    cd D:\Downloads\intern\assignment_1_frontend_backend\invoice-processor
    python -m http.server 8080
    Access at: http://localhost:8080

Diagnostic test:
    python pipeline_full_test.py
    Open pipeline_report.html in browser to view results.

SECTION 7 — FILES CHANGED IN THIS SESSION
List every file that was modified or created, with a one-line description of what changed:
- ocr_processor.py: replaced single-batch Surya call with per-page loop + 120s timeout per page + predictor rebuild on timeout + _try_extract_embedded_text() fast path using pdfplumber
- llm_extractor.py: added _preprocess_invoice_text(), _extract_header_fields(), _extract_line_items_regex(), _validate_extraction(), rewrote extract() to use regex-first hybrid architecture
- pipeline_full_test.py: new standalone 8-stage diagnostic script
- pipeline_report.html: auto-generated HTML report from pipeline_full_test.py
- debug_last_prompt.txt: auto-generated debug file, written on every LLM call
- HANDOFF.md: this file
