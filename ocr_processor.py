"""Document text extraction via Surya 2 (replaces Tesseract/pdfplumber OCR).

process_file() keeps the same signature as the old Tesseract version --
takes a file path, returns plain text -- so preprocessor.py/llm_extractor.py
downstream are unaffected by this swap. Table structure (Surya's table_rec)
is intentionally not used: line items are extracted by the LLM directly from
plain OCR text via llm_extractor.py's prompt, not from table geometry.
"""

import os
import re
import logging
from typing import List
from concurrent.futures import ThreadPoolExecutor, TimeoutError as FutureTimeoutError

import torch
import pdfplumber
from pdf2image import convert_from_path
from PIL import Image

import config
from surya.inference import SuryaInferenceManager
from surya.recognition import RecognitionPredictor

IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff"}

# ponytail: one shared manager/predictor for the process lifetime -- model
# load is the expensive part. Set SURYA_INFERENCE_KEEP_ALIVE=1 so the
# underlying vllm/llama.cpp server also survives across requests, not just
# across calls within one process.
_manager = SuryaInferenceManager(method="llamacpp")
_recognition_predictor = RecognitionPredictor(_manager)

if torch.cuda.is_available():
    logging.info(f"Surya OCR initialized on GPU: {torch.cuda.get_device_name(0)}")
else:
    logging.warning("Surya OCR initialized on CPU. OCR will be slow.")

# ponytail: bounds the shared singleton's blocking call so one hung request
# fails fast instead of wedging every future request on this process for
# good (confirmed: a tiny PNG hung 11+ min after an earlier request never
# returned -- same shared _recognition_predictor serves all requests).
# Ceiling: this only makes the HTTP request fail cleanly; it does NOT
# recover the shared predictor itself, which may still be wedged internally
# after a timeout, so subsequent calls could keep failing until restart.
# Upgrade path: recreate _manager/_recognition_predictor after a timeout
# instead of reusing the same (possibly broken) instance.
_OCR_TIMEOUT = 120.0
_ocr_executor = ThreadPoolExecutor(max_workers=1)


def _load_pages(file_path: str) -> List[Image.Image]:
    """Load a PDF or image file as a list of PIL pages.
    
    Images below 200 DPI are upscaled to 300 DPI equivalent before OCR.
    Surya drops small text at low resolutions (110 DPI loses product codes
    in dense invoice tables entirely).
    """
    ext = os.path.splitext(file_path)[1].lower()
    if ext == ".pdf":
        # PDFs: render at 300 DPI directly via pdf2image
        return convert_from_path(file_path, dpi=300, poppler_path=config.POPPLER_PATH)
    if ext in IMAGE_EXTENSIONS:
        img = Image.open(file_path)
        # Convert RGBA to RGB — Surya doesn't handle alpha channel reliably
        if img.mode == "RGBA":
            img = img.convert("RGB")
        # Upscale if DPI is below 200
        dpi = img.info.get("dpi", (72, 72))
        current_dpi = dpi[0] if isinstance(dpi, tuple) else dpi
        if current_dpi < 200:
            scale = 300 / current_dpi
            new_size = (int(img.width * scale), int(img.height * scale))
            img = img.resize(new_size, Image.LANCZOS)
            logging.info(f"Upscaled image from {current_dpi:.0f} DPI to 300 DPI equivalent: {new_size}")
        return [img]
    raise ValueError(f"Unsupported file format: {ext}")


def _try_extract_embedded_text(file_path: str):
    """Try to extract embedded text from a PDF, falling back to None if sparse."""
    if not file_path.lower().endswith(".pdf"):
        return None
    try:
        with pdfplumber.open(file_path) as pdf:
            text = []
            num_pages = len(pdf.pages)
            for page in pdf.pages:
                page_text = page.extract_text()
                if page_text:
                    text.append(page_text)
            
            full_text = "\n".join(text).strip()
            # Heuristic: return text if total characters > 100 * number of pages
            if len(full_text) > 100 * num_pages:
                return full_text
    except Exception:
        pass
    return None


def process_file(file_path: str) -> str:
    """Extract plain text from a PDF or image via Surya 2 full-page OCR."""
    global _manager, _recognition_predictor
    
    embedded_text = _try_extract_embedded_text(file_path)
    if embedded_text:
        return embedded_text

    try:
        pages = _load_pages(file_path)
        page_texts = []
        for i, page in enumerate(pages, 1):
            future = _ocr_executor.submit(_recognition_predictor, [page])
            try:
                predictions = future.result(timeout=_OCR_TIMEOUT)
                for page_result in predictions:
                    block_texts = [re.sub(r"<[^>]+>", " ", b.html) for b in page_result.blocks if b.html]
                    page_texts.append("\n".join(t.strip() for t in block_texts if t.strip()))
            except FutureTimeoutError:
                print(f"Warning: OCR timed out for page {i} of {file_path}")
                page_texts.append(f"[PAGE {i} OCR FAILED: timed out]")
                # Rebuild the predictor so the next request doesn't inherit a wedged predictor
                _manager = SuryaInferenceManager(method="llamacpp")
                _recognition_predictor = RecognitionPredictor(_manager)
                
        return "\n\n".join(page_texts).strip()
    except Exception as error:
        raise RuntimeError(f"OCR failed for {file_path}: {error}") from error
