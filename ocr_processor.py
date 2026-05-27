"""OCR and document text extraction helpers."""

import os

from PIL import Image
import pdfplumber
from pdf2image import convert_from_path
import pytesseract

from config import TESSERACT_PATH


pytesseract.pytesseract.tesseract_cmd = TESSERACT_PATH


def extract_from_image(image_path: str) -> str:
	"""Extract text from an image file using Tesseract OCR."""

	try:
		with Image.open(image_path) as image:
			text = pytesseract.image_to_string(image)
		return text.strip()
	except Exception as error:
		return f"Error extracting text from image: {error}"


def extract_scanned_pdf(pdf_path: str) -> str:
	"""Extract text from a scanned PDF by converting pages to images first."""

	try:
		pages = convert_from_path(pdf_path)
		extracted_text = []

		for page in pages:
			text = pytesseract.image_to_string(page)
			if text:
				extracted_text.append(text.strip())

		return "\n\n".join(extracted_text).strip()
	except Exception as error:
		return f"Error extracting scanned PDF text: {error}"


def extract_from_pdf(pdf_path: str) -> str:
	"""Extract text from a PDF, falling back to OCR for scanned documents."""

	try:
		extracted_text = []

		with pdfplumber.open(pdf_path) as pdf:
			for page in pdf.pages:
				text = page.extract_text()
				if text:
					extracted_text.append(text.strip())

		combined_text = "\n\n".join(extracted_text).strip()
		if combined_text:
			return combined_text

		return extract_scanned_pdf(pdf_path)
	except Exception as error:
		return f"Error extracting text from PDF: {error}"


def process_file(file_path: str) -> str:
	"""Dispatch text extraction based on the file extension."""

	try:
		file_extension = os.path.splitext(file_path)[1].lower()

		if file_extension == ".pdf":
			return extract_from_pdf(file_path)

		if file_extension in {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff"}:
			return extract_from_image(file_path)

		return f"Unsupported file format: {file_extension}"
	except Exception as error:
		return f"Error processing file: {error}"

