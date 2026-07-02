import sys
import os
import subprocess
import time
import json
import urllib.request
import traceback

print("Starting Pipeline Diagnostic...\n")

results = []

def print_checkpoint_header(name):
    print(f"\n{'='*80}")
    print(f"RUNNING CHECKPOINT: {name}")
    print(f"{'='*80}")

def register_result(name, status, reason):
    results.append((name, status, reason))
    print(f"\n>>> {name} - {'PASS' if status else 'FAIL'}: {reason}")

# --- Checkpoint 1 ---
def cp1_environment():
    name = "CHECKPOINT 1 - Environment & Dependencies"
    print_checkpoint_header(name)
    try:
        print(f"Python version: {sys.version}")
        
        deps = ['pdfplumber', 'pdf2image', 'PIL', 'surya', 'torch', 'fastapi', 'uvicorn', 'ollama']
        import importlib
        all_imports_passed = True
        for dep in deps:
            try:
                mod = importlib.import_module(dep)
                ver = getattr(mod, '__version__', 'unknown')
                print(f"Import {dep}: SUCCESS (version {ver})")
            except Exception as e:
                print(f"Import {dep}: FAIL ({e})")
                all_imports_passed = False
                
        import torch
        if torch.cuda.is_available():
            print(f"GPU available: {torch.cuda.get_device_name(0)}")
        else:
            print("WARNING: GPU not available. OCR will run on CPU and be slow.")
            
        try:
            out = subprocess.check_output(['pdfinfo', '-v'], stderr=subprocess.STDOUT)
            print(f"pdfinfo: {out.decode('utf-8', errors='ignore').splitlines()[0]}")
        except Exception as e:
            print(f"pdfinfo FAIL: {e}")
            return False, "pdfinfo not found or failed"
            
        try:
            req = urllib.request.Request('http://localhost:11434/')
            with urllib.request.urlopen(req, timeout=5) as resp:
                print(f"Ollama server reachable. Status: {resp.status}")
        except Exception as e:
            print(f"Ollama server FAIL: {e}")
            return False, "Ollama server unreachable at http://localhost:11434"
            
        if all_imports_passed:
            register_result(name, True, "All basic env checks passed")
            return True, None
        else:
            register_result(name, False, "Some imports failed")
            return False, None
    except Exception as e:
        traceback.print_exc()
        register_result(name, False, f"Exception: {e}")
        return False, None

# --- Setup PDF ---
test_pdf_path = None
def get_test_pdf():
    global test_pdf_path
    if test_pdf_path: return test_pdf_path
    uploads_dir = "uploads"
    if os.path.exists(uploads_dir):
        for f in os.listdir(uploads_dir):
            if f.lower().endswith(".pdf"):
                test_pdf_path = os.path.join(uploads_dir, f)
                return test_pdf_path
    if os.path.exists("test_data"):
        for f in os.listdir("test_data"):
            if f.lower().endswith(".pdf"):
                test_pdf_path = os.path.join("test_data", f)
                return test_pdf_path
                
    # Create minimal PDF
    try:
        import reportlab.pdfgen.canvas
        test_pdf_path = "diagnostic_test.pdf"
        c = reportlab.pdfgen.canvas.Canvas(test_pdf_path)
        c.drawString(100, 750, "INVOICE DIAGNOSTIC TEST")
        c.drawString(100, 730, "Amount: $100.00")
        c.drawString(100, 710, "This is a test invoice to ensure extraction works.")
        c.save()
        print(f"Created minimal test PDF at {test_pdf_path}")
    except Exception as e:
        print(f"Could not create test PDF: {e}")
    return test_pdf_path

# --- Checkpoint 2 ---
def cp2_embedded_text():
    name = "CHECKPOINT 2 - PDF Embedded Text Extraction"
    print_checkpoint_header(name)
    try:
        pdf_path = get_test_pdf()
        if not pdf_path:
            register_result(name, False, "No PDF found to test")
            return False, None
            
        import pdfplumber
        print(f"Testing on PDF: {pdf_path}")
        with pdfplumber.open(pdf_path) as pdf:
            pages = pdf.pages
            print(f"Number of pages: {len(pages)}")
            text = []
            for page in pages:
                t = page.extract_text()
                if t: text.append(t)
            full_text = "\n".join(text).strip()
            print(f"Total characters extracted: {len(full_text)}")
            print(f"First 300 chars: {full_text[:300]!r}")
            
            if len(full_text) > 100 * len(pages):
                register_result(name, True, "Embedded text extraction successful")
                return True, full_text
            else:
                register_result(name, False, "Insufficient embedded text found")
                return False, None
    except Exception as e:
        traceback.print_exc()
        register_result(name, False, f"Exception: {e}")
        return False, None

# --- Checkpoint 3 ---
def cp3_pdf2image():
    name = "CHECKPOINT 3 - pdf2image / Poppler"
    print_checkpoint_header(name)
    try:
        pdf_path = get_test_pdf()
        if not pdf_path:
            register_result(name, False, "No PDF found to test")
            return False, None
            
        from pdf2image import convert_from_path
        images = convert_from_path(pdf_path, first_page=1, last_page=1)
        if not images:
            register_result(name, False, "No images returned by convert_from_path")
            return False, None
            
        img = images[0]
        print(f"Image size: {img.size}, mode: {img.mode}")
        out_path = "diagnostic_page1.png"
        img.save(out_path)
        print(f"Saved to {out_path}")
        
        if os.path.exists(out_path) and os.path.getsize(out_path) > 0:
            register_result(name, True, f"Image successfully saved to {out_path}")
            return True, out_path
        else:
            register_result(name, False, "Saved image file is missing or empty")
            return False, None
    except Exception as e:
        traceback.print_exc()
        register_result(name, False, f"Exception: {e}")
        return False, None

# --- Checkpoint 4 ---
def cp4_surya_ocr(image_path):
    name = "CHECKPOINT 4 - Surya OCR on single page"
    print_checkpoint_header(name)
    if not image_path or not os.path.exists(image_path):
        register_result(name, False, "No image to process")
        return False, None
        
    try:
        from PIL import Image
        import re
        from concurrent.futures import ThreadPoolExecutor, TimeoutError
        import ocr_processor
        
        img = Image.open(image_path)
        executor = ThreadPoolExecutor(max_workers=1)
        
        start_t = time.time()
        print("Submitting to _recognition_predictor...")
        future = executor.submit(ocr_processor._recognition_predictor, [img])
        try:
            predictions = future.result(timeout=120.0)
            end_t = time.time()
            
            page_result = predictions[0]
            print(f"Time taken: {end_t - start_t:.2f}s")
            print(f"Number of blocks returned: {len(page_result.blocks)}")
            
            block_texts = [re.sub(r'<[^>]+>', ' ', b.html) for b in page_result.blocks if b.html]
            full_text = "\n".join(t.strip() for t in block_texts if t.strip())
            
            print(f"First 300 chars: {full_text[:300]!r}")
            register_result(name, True, "Surya OCR completed within timeout")
            return True, full_text
        except TimeoutError:
            end_t = time.time()
            print(f"Time taken: {end_t - start_t:.2f}s")
            print("Predictor appears wedged (timeout).")
            register_result(name, False, "OCR timed out after 120s")
            return False, None
    except Exception as e:
        traceback.print_exc()
        register_result(name, False, f"Exception: {e}")
        return False, None

# --- Checkpoint 5 ---
def cp5_ollama():
    name = "CHECKPOINT 5 - Ollama / LLM reachability"
    print_checkpoint_header(name)
    try:
        print("Running `ollama list`...")
        out = subprocess.check_output(['ollama', 'list'], stderr=subprocess.STDOUT)
        out_str = out.decode('utf-8', errors='ignore')
        print(out_str)
        if 'qwen2.5:3b' not in out_str:
            print("WARNING: qwen2.5:3b model not found in `ollama list`")
            
        print("Sending POST request to http://localhost:11434/api/generate ...")
        payload = {
            "model": "qwen2.5:3b",
            "prompt": "Reply with the word PONG and nothing else",
            "stream": False
        }
        data = json.dumps(payload).encode('utf-8')
        req = urllib.request.Request(
            'http://localhost:11434/api/generate', 
            data=data, 
            headers={'Content-Type': 'application/json'}
        )
        
        start_t = time.time()
        try:
            with urllib.request.urlopen(req, timeout=30) as resp:
                body = json.loads(resp.read().decode('utf-8'))
                end_t = time.time()
                print(f"Response status: {resp.status}")
                print(f"Response body: {body.get('response', '')}")
                print(f"Time taken: {end_t - start_t:.2f}s")
                
                if 'PONG' in body.get('response', '').upper():
                    register_result(name, True, "Ollama reachable and responded with PONG")
                    return True, None
                else:
                    register_result(name, False, "Ollama responded but missing PONG")
                    return False, None
        except Exception as e:
            end_t = time.time()
            print(f"Time taken: {end_t - start_t:.2f}s")
            raise e
    except Exception as e:
        traceback.print_exc()
        register_result(name, False, f"Exception: {e}")
        return False, None

# --- Checkpoint 6 ---
def cp6_llm_extractor(real_text):
    name = "CHECKPOINT 6 - LLM Extractor on real text"
    print_checkpoint_header(name)
    if not real_text:
        real_text = "INVOICE #12345\nDate: 2024-01-01\nTotal: $150.00"
        print("No extracted text provided, using hardcoded fallback text.")
        
    try:
        import llm_extractor
        start_t = time.time()
        res = llm_extractor.extract_invoice_fields(real_text)
        end_t = time.time()
        
        # Extract returns a Pydantic object usually, so we dump to JSON
        res_json = res.model_dump_json(indent=2)
        print(f"Time taken: {end_t - start_t:.2f}s")
        print(f"Extracted JSON:\n{res_json}")
        
        # Check if at least one non-null field
        res_dict = json.loads(res_json)
        has_val = any(v is not None and v != [] and v != "" for v in res_dict.values())
        
        if has_val:
            register_result(name, True, "LLM returned structured data with values")
            return True, None
        else:
            register_result(name, False, "LLM returned empty or null fields only")
            return False, None
            
    except Exception as e:
        traceback.print_exc()
        register_result(name, False, f"Exception: {e}")
        return False, None

# --- Checkpoint 7 ---
def cp7_end_to_end():
    name = "CHECKPOINT 7 - Full pipeline end-to-end"
    print_checkpoint_header(name)
    try:
        pdf_path = get_test_pdf()
        if not pdf_path:
            register_result(name, False, "No PDF found to test")
            return False, None
            
        import ocr_processor
        import llm_extractor
        
        print(f"Running ocr_processor on {pdf_path}")
        start_t = time.time()
        text = ocr_processor.process_file(pdf_path)
        end_t = time.time()
        
        print(f"OCR Time taken: {end_t - start_t:.2f}s")
        print(f"Character count: {len(text)}")
        print(f"First 300 chars: {text[:300]!r}")
        
        print("\nPiping text into llm_extractor...")
        start_t = time.time()
        res = llm_extractor.extract_invoice_fields(text)
        end_t = time.time()
        
        res_json = res.model_dump_json(indent=2)
        print(f"LLM Time taken: {end_t - start_t:.2f}s")
        print(f"Final extracted JSON:\n{res_json}")
        
        res_dict = json.loads(res_json)
        has_val = any(v is not None and v != [] and v != "" for v in res_dict.values())
        if has_val:
            register_result(name, True, "Pipeline end-to-end succeeded")
            return True, None
        else:
            register_result(name, False, "Pipeline ran but extraction is empty")
            return False, None
            
    except Exception as e:
        traceback.print_exc()
        register_result(name, False, f"Exception: {e}")
        return False, None

def print_summary():
    print(f"\n{'='*80}")
    print("SUMMARY TABLE")
    print(f"{'='*80}")
    for res in results:
        status_str = "PASS" if res[1] else "FAIL"
        print(f"[{status_str}] {res[0]}\n      {res[2]}")
        
    print(f"\n{'='*80}")
    print("RECOMMENDED NEXT steps")
    print(f"{'='*80}")
    
    failed = [r for r in results if not r[1]]
    if not failed:
        print("All checkpoints passed! The pipeline is fully operational.")
    else:
        for r in failed:
            if "CHECKPOINT 1" in r[0]:
                print("- Fix environment/dependencies. Run `pip install -r requirements.txt`, check Ollama/Poppler path.")
            elif "CHECKPOINT 2" in r[0]:
                print("- PDF embedded text extraction failed. This is a fallback layer, so it's OK if the PDF is scanned, but if it's text-based, pdfplumber might be failing.")
            elif "CHECKPOINT 3" in r[0]:
                print("- Poppler/pdf2image failed. Check if Poppler is installed and in your PATH.")
            elif "CHECKPOINT 4" in r[0]:
                print("- Surya OCR failed or timed out. Check CUDA/GPU availability, or if Surya is wedged, restart the server.")
            elif "CHECKPOINT 5" in r[0]:
                print("- Ollama LLM is unreachable or timing out. Check if `ollama serve` is running and the model qwen2.5:3b is pulled.")
            elif "CHECKPOINT 6" in r[0]:
                print("- LLM Extraction failed on raw text. Check llm_extractor.py logic and the prompt format.")
            elif "CHECKPOINT 7" in r[0]:
                print("- End-to-end failed despite passing earlier checkpoints. There may be an integration issue between OCR output and the LLM prompt.")

if __name__ == '__main__':
    cp1_environment()
    success, text = cp2_embedded_text()
    success, img_path = cp3_pdf2image()
    cp4_surya_ocr(img_path)
    cp5_ollama()
    cp6_llm_extractor(text if text else None)
    cp7_end_to_end()
    
    print_summary()
