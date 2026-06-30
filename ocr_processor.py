"""Document text extraction via Surya 2 (replaces Tesseract/pdfplumber OCR).

process_file() keeps the same signature as the old Tesseract version --
takes a file path, returns plain text -- so preprocessor.py/llm_extractor.py
downstream are unaffected by this swap. Table structure (Surya's table_rec)
is intentionally not used: line items are extracted by the LLM directly from
plain OCR text via llm_extractor.py's prompt, not from table geometry.
"""

import os
import re
from typing import List

from pdf2image import convert_from_path
from PIL import Image

from surya.inference import SuryaInferenceManager
from surya.recognition import RecognitionPredictor

IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff"}

# ponytail: one shared manager/predictor for the process lifetime -- model
# load is the expensive part. Set SURYA_INFERENCE_KEEP_ALIVE=1 so the
# underlying vllm/llama.cpp server also survives across requests, not just
# across calls within one process.
_manager = SuryaInferenceManager(method="llamacpp")
_recognition_predictor = RecognitionPredictor(_manager)


def _load_pages(file_path: str) -> List[Image.Image]:
    """Load a PDF or image file as a list of PIL pages."""
    ext = os.path.splitext(file_path)[1].lower()
    if ext == ".pdf":
        return convert_from_path(file_path)
    if ext in IMAGE_EXTENSIONS:
        return [Image.open(file_path)]
    raise ValueError(f"Unsupported file format: {ext}")


def process_file(file_path: str) -> str:
    """Extract plain text from a PDF or image via Surya 2 full-page OCR."""
    try:
        pages = _load_pages(file_path)
        predictions = _recognition_predictor(pages)
        page_texts = []
        for page in predictions:
            block_texts = [re.sub(r"<[^>]+>", " ", b.html) for b in page.blocks if b.html]
            page_texts.append("\n".join(t.strip() for t in block_texts if t.strip()))
        return "\n\n".join(page_texts).strip()
    except Exception as error:
        return f"Error processing file: {error}"
