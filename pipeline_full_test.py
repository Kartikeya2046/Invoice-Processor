import sys
import os
import time
import traceback
import json
import subprocess
import logging
import io
import urllib.request
import re
from datetime import datetime

# Setup paths
PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, PROJECT_ROOT)

os.environ["PATH"] = r"D:\poppler\poppler-24.02.0\Library\bin" + ";" + r"D:\llama.cpp\llama-b9844-bin-win-cuda-13.3-x64" + ";" + os.environ.get("PATH", "")

# Stage storage
stages_data = []

def record_stage(name, status, t_taken, output):
    stages_data.append({
        "name": name,
        "status": status,
        "time": t_taken,
        "output": output
    })

def run_stage(name, func):
    print(f"\n--- Running: {name} ---")
    start = time.perf_counter()
    status = "FAIL"
    output = ""
    try:
        status, output = func()
    except Exception:
        output = traceback.format_exc()
    end = time.perf_counter()
    record_stage(name, status, end - start, output)

def get_test_pdf():
    uploads_dir = os.path.join(PROJECT_ROOT, "uploads")
    if os.path.exists(uploads_dir):
        pdfs = [os.path.join(uploads_dir, f) for f in os.listdir(uploads_dir) if f.lower().endswith(".pdf")]
        if pdfs:
            latest_pdf = max(pdfs, key=os.path.getmtime)
            return latest_pdf
            
    print("No PDF found in uploads/, creating fake one with reportlab...")
    try:
        from reportlab.pdfgen import canvas
        fake_pdf = os.path.join(PROJECT_ROOT, "fake_test_invoice.pdf")
        c = canvas.Canvas(fake_pdf)
        c.drawString(100, 750, "Invoice No. 103565")
        c.drawString(100, 730, "Purchase Order No. 74294838")
        c.drawString(100, 710, "Mouser Electronics")
        c.drawString(100, 690, "Invoice Date 21-JUN-23")
        c.drawString(100, 670, "Line No.  Part No.  Shipped  Price/Unit")
        c.drawString(100, 650, "1  625-1N6276A-E3  4  1.450")
        for i in range(2, 12):
            c.drawString(100, 650 - (i*20), f"{i}  512-LL4148  72  0.109")
        c.save()
        return fake_pdf
    except Exception as e:
        print(f"Could not create fake PDF: {e}")
        return None

def stage1_deps():
    out = []
    failed = False
    deps = ['pdfplumber', 'pdf2image', 'PIL', 'surya', 'torch', 'fastapi', 'uvicorn', 'httpx', 'reportlab']
    import importlib
    for dep in deps:
        try:
            mod = importlib.import_module(dep)
            ver = getattr(mod, '__version__', 'unknown')
            out.append(f"{dep} version: {ver}")
        except ImportError as e:
            out.append(f"{dep} FAIL: {e}")
            failed = True
            
    try:
        import torch
        if torch.cuda.is_available():
            out.append(f"GPU: {torch.cuda.get_device_name(0)}")
        else:
            out.append("WARNING: Running on CPU — OCR will be slow")
    except Exception as e:
        out.append(f"torch check FAIL: {e}")
        
    try:
        res = subprocess.run(["pdfinfo", "-v"], capture_output=True, text=True)
        if res.returncode != 0 and res.returncode != 99:
            out.append(f"pdfinfo FAIL: code {res.returncode}\n{res.stderr}")
            failed = True
        else:
            out.append(f"pdfinfo SUCCESS: {res.stdout.splitlines()[0] if res.stdout else 'ok'}")
    except Exception as e:
        out.append(f"pdfinfo subprocess FAIL: {e}")
        failed = True
        
    return "FAIL" if failed else "PASS", "\n".join(out)

# Global variables to pass state between stages
test_pdf = None
raw_extracted_text = ""
preprocessed_text = ""

def stage2_embedded():
    global raw_extracted_text
    import pdfplumber
    out = []
    with pdfplumber.open(test_pdf) as pdf:
        pages = pdf.pages
        out.append(f"Total pages: {len(pages)}")
        counts = []
        text_parts = []
        for p in pages:
            t = p.extract_text()
            if t:
                text_parts.append(t)
                counts.append(len(t))
            else:
                counts.append(0)
        
        raw_extracted_text = "\n".join(text_parts)
        total_chars = len(raw_extracted_text)
        out.append(f"Character count per page: {counts}")
        out.append(f"Total character count: {total_chars}")
        out.append(f"First 500 chars:\n{raw_extracted_text[:500]}")
        out.append(f"Last 200 chars:\n{raw_extracted_text[-200:]}")
        
        status = "PASS" if total_chars > 100 * len(pages) else "FAIL"
        if status == "FAIL":
            out.append("FAIL: Character count below threshold (likely scanned PDF).")
        return status, "\n".join(out)

def stage3_preprocess():
    global preprocessed_text
    from llm_extractor import _preprocess_invoice_text
    out = []
    
    preprocessed_text = _preprocess_invoice_text(raw_extracted_text)
    
    orig_len = len(raw_extracted_text)
    new_len = len(preprocessed_text)
    reduction = ((orig_len - new_len) / orig_len * 100) if orig_len > 0 else 0
    
    out.append(f"Before preprocessing: {orig_len} chars")
    out.append(f"After preprocessing: {new_len} chars")
    out.append(f"Percentage reduction: {reduction:.2f}%")
    out.append(f"First 500 chars:\n{preprocessed_text[:500]}")
    out.append(f"Last 200 chars:\n{preprocessed_text[-200:]}")
    
    status = "PASS" if (new_len > 0 and new_len <= orig_len) else "FAIL"
    if status == "FAIL":
        out.append("FAIL: Preprocessing produced empty output or grew the text (unexpected).")
        
    return status, "\n".join(out)

def stage4_ollama():
    out = []
    failed = False
    
    res = subprocess.run(["ollama", "list"], capture_output=True, text=True)
    out.append(f"ollama list output:\n{res.stdout}")
    if res.returncode != 0 or "qwen2.5:7b" not in res.stdout:
        out.append("FAIL: qwen2.5:7b not found or command failed")
        failed = True
        
    payload = {"model": "qwen2.5:7b", "prompt": "Reply with the word PONG and nothing else.", "stream": False}
    try:
        start = time.perf_counter()
        req = urllib.request.Request("http://localhost:11434/api/generate", data=json.dumps(payload).encode("utf-8"), headers={"Content-Type": "application/json"})
        with urllib.request.urlopen(req, timeout=30) as resp:
            body = resp.read().decode('utf-8')
            end = time.perf_counter()
            out.append(f"HTTP status: {resp.status}")
            out.append(f"Response body: {body}")
            out.append(f"Time taken: {end-start:.2f}s")
            if "PONG" in body.upper():
                pass
            else:
                out.append("FAIL: PONG not found in response")
                failed = True
    except Exception as e:
        out.append(f"HTTP POST FAIL: {traceback.format_exc()}")
        failed = True
        
    return "FAIL" if failed else "PASS", "\n".join(out)

stage5_result_dict = {}

def stage5_extract():
    global stage5_result_dict
    from llm_extractor import extract_invoice_fields
    out = []
    
    text_to_use = preprocessed_text if preprocessed_text else raw_extracted_text
    
    start = time.perf_counter()
    res = extract_invoice_fields(text_to_use)
    end = time.perf_counter()
    
    debug_file = os.path.join(PROJECT_ROOT, "debug_last_prompt.txt")
    if os.path.exists(debug_file):
        with open(debug_file, "r", encoding="utf-8") as f:
            prompt = f.read()
            out.append(f"Prompt length from debug file: {len(prompt)} chars")
            
    res_json = res.model_dump_json()
    out.append(f"Raw JSON string returned (model dump):\n{res_json}")
    
    try:
        stage5_result_dict = json.loads(res_json)
        out.append("Valid JSON: True")
    except Exception as e:
        out.append(f"Valid JSON: False ({e})")
        return "FAIL", "\n".join(out)
        
    out.append(f"Extraction Time: {end-start:.2f}s")
    
    d = stage5_result_dict
    fields = [
        ("invoice_number", d.get("invoice_number"), "103565"),
        ("po_number", d.get("po_number"), "74294838"),
        ("supplier", d.get("supplier"), "Mouser"),
        ("invoice_date", d.get("invoice_date"), "21-JUN-23")
    ]
    
    non_null_count = 0
    for name, val, exp in fields:
        out.append(f"{name}: {val} (expected ~ {exp})")
        if val is not None and str(val).strip() != "":
            non_null_count += 1
            
    li = d.get("line_items", [])
    out.append(f"line_items count: {len(li)} (expected > 10)")
    if li and len(li) > 0:
        non_null_count += 1
        
    out.append("\nFirst 5 line items:")
    for i, item in enumerate(li[:5]):
        out.append(f"  {i+1}: line_number={item.get('line_number')}, product_id={item.get('product_id')}, qty={item.get('quantity')}, price={item.get('unit_price')}")
        
    if non_null_count >= 3:
        return "PASS", "\n".join(out)
    else:
        out.append(f"FAIL: Only {non_null_count} header fields non-null.")
        return "FAIL", "\n".join(out)

def stage6_validation():
    from llm_extractor import _validate_extraction
    out = []
    
    buffer = io.StringIO()
    handler = logging.StreamHandler(buffer)
    logger = logging.getLogger()
    logger.addHandler(handler)
    
    try:
        _validate_extraction(stage5_result_dict, preprocessed_text)
    finally:
        logger.removeHandler(handler)
        
    log_output = buffer.getvalue()
    out.append(log_output)
    
    passed_count = log_output.count("PASS")
    if "GOOD" in log_output or passed_count >= 3:
        return "PASS", "\n".join(out)
    else:
        out.append("FAIL: Did not see GOOD or >= 3 PASS lines.")
        return "FAIL", "\n".join(out)

def stage7_full_ocr():
    from ocr_processor import process_file
    out = []
    start = time.perf_counter()
    text = process_file(test_pdf)
    end = time.perf_counter()
    
    out.append(f"Time taken: {end-start:.2f}s")
    out.append(f"Character count: {len(text)}")
    out.append(f"First 300 chars:\n{text[:300]}")
    
    if len(text) > 0:
        return "PASS", "\n".join(out)
    else:
        out.append("FAIL: OCR/text extraction returned empty content")
        return "FAIL", "\n".join(out)

def stage8_end_to_end():
    from ocr_processor import process_file
    from llm_extractor import extract_invoice_fields
    out = []
    
    start = time.perf_counter()
    text = process_file(test_pdf)
    res = extract_invoice_fields(text)
    end = time.perf_counter()
    
    out.append(f"Total time combined: {end-start:.2f}s")
    res_dict = json.loads(res.model_dump_json())
    out.append(f"Final JSON result:\n{json.dumps(res_dict, indent=2)}")
    
    inv = res_dict.get("invoice_number")
    po = res_dict.get("po_number")
    li = res_dict.get("line_items")
    
    if inv is not None and po is not None and li and len(li) > 0:
        return "PASS", "\n".join(out)
    else:
        out.append("FAIL: Missing required final fields.")
        return "FAIL", "\n".join(out)


def generate_html_report():
    html = [
        "<html><head><title>Pipeline Diagnostic Report</title>",
        "<style>",
        "body { font-family: Arial, sans-serif; background: #f4f4f9; margin: 0; padding: 20px; }",
        ".header { background: #1e1e24; color: white; padding: 20px; border-radius: 8px; margin-bottom: 20px; }",
        ".card { background: white; margin-bottom: 20px; border-radius: 8px; padding: 20px; box-shadow: 0 2px 4px rgba(0,0,0,0.1); border-left: 5px solid #ccc; }",
        ".pass { border-left-color: #28a745; }",
        ".fail { border-left-color: #dc3545; }",
        ".warn { border-left-color: #ffc107; }",
        "pre { background: #f8f9fa; padding: 15px; border-radius: 5px; overflow-x: auto; white-space: pre-wrap; font-size: 13px; }",
        ".badge { padding: 5px 10px; border-radius: 4px; font-weight: bold; color: white; display: inline-block; }",
        ".badge-pass { background: #28a745; }",
        ".badge-fail { background: #dc3545; }",
        ".diagnosis { background: #fff3cd; color: #856404; padding: 20px; border-radius: 8px; margin-top: 20px; }",
        "</style></head><body>"
    ]
    
    pass_cnt = sum(1 for s in stages_data if s['status'] == 'PASS')
    fail_cnt = sum(1 for s in stages_data if s['status'] == 'FAIL')
    if fail_cnt == 0:
        overall = "ALL PASS"
    elif pass_cnt == 0:
        overall = "ALL FAIL"
    else:
        overall = "PARTIAL"
        
    html.append(f"<div class='header'>")
    html.append(f"<h2>Pipeline Diagnostic Report</h2>")
    html.append(f"<p>Timestamp: {datetime.now().isoformat()}</p>")
    html.append(f"<p>Test File: {test_pdf} ({os.path.getsize(test_pdf)} bytes)</p>")
    html.append(f"<p>Python Version: {sys.version}</p>")
    html.append(f"<p>Total Stages: {len(stages_data)} | PASS: {pass_cnt} | FAIL: {fail_cnt} | <strong>Status: {overall}</strong></p>")
    html.append("</div>")
    
    for i, s in enumerate(stages_data, 1):
        cls = "pass" if s['status'] == "PASS" else "fail"
        bdg = "badge-pass" if s['status'] == "PASS" else "badge-fail"
        html.append(f"<div class='card {cls}'>")
        html.append(f"<h3>Stage {i}: {s['name']} <span class='badge {bdg}'>{s['status']}</span></h3>")
        html.append(f"<p>Time Taken: {s['time']:.2f}s</p>")
        html.append(f"<pre>{s['output']}</pre>")
        html.append("</div>")
        
    if fail_cnt > 0:
        html.append("<div class='diagnosis'><h3>DIAGNOSIS & SUGGESTED FIXES</h3><ul>")
        for s in stages_data:
            if s['status'] == 'FAIL':
                if "Stage 1" in s['name']: msg = "Missing dependency: install the failed package via pip"
                elif "Stage 2" in s['name']: msg = "PDF has no embedded text: this is a scanned PDF and will require OCR"
                elif "Stage 3" in s['name']: msg = "_preprocess_invoice_text not found or threw an error: check llm_extractor.py for the function"
                elif "Stage 4" in s['name']: msg = "Ollama is not running or qwen2.5:7b is not pulled: run ollama serve and ollama pull qwen2.5:7b"
                elif "Stage 5" in s['name']: msg = "LLM extraction failed: check debug_last_prompt.txt for what the model received and check llm_extractor.py prompt"
                elif "Stage 6" in s['name']: msg = "Validation layer missing or all fields failing: check prompt tuning and field definitions in llm_extractor.py"
                elif "Stage 7" in s['name']: msg = "process_file() threw an error: check ocr_processor.py and Poppler/Surya setup"
                elif "Stage 8" in s['name']: msg = "End-to-end pipeline broken: check the stage that failed individually above for root cause"
                else: msg = "Unknown failure."
                html.append(f"<li><strong>{s['name']}:</strong> {msg}</li>")
        html.append("</ul></div>")
        
    html.append("</body></html>")
    
    report_path = os.path.join(PROJECT_ROOT, "pipeline_report.html")
    with open(report_path, "w", encoding="utf-8") as f:
        f.write("\n".join(html))
        
    print("\n" + "="*50)
    print("SUMMARY")
    print("="*50)
    for s in stages_data:
        print(f"[{s['status']}] {s['name']} ({s['time']:.2f}s)")
    print(f"\nReport saved to pipeline_report.html — open in browser to view results")


if __name__ == "__main__":
    test_pdf = get_test_pdf()
    if test_pdf:
        print(f"Selected test PDF: {test_pdf} ({os.path.getsize(test_pdf)} bytes)")
        run_stage("Stage 1 - Dependency Check", stage1_deps)
        run_stage("Stage 2 - Embedded Text Extraction", stage2_embedded)
        run_stage("Stage 3 - Text Preprocessing", stage3_preprocess)
        run_stage("Stage 4 - Ollama Connectivity and Model Check", stage4_ollama)
        run_stage("Stage 5 - LLM Extraction on Preprocessed Text", stage5_extract)
        run_stage("Stage 6 - Validation Layer", stage6_validation)
        run_stage("Stage 7 - process_file() Full OCR Path", stage7_full_ocr)
        run_stage("Stage 8 - End to End", stage8_end_to_end)
        
        generate_html_report()
    else:
        print("Fatal error: Could not find or create a test PDF.")
