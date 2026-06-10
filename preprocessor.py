"""OCR text preprocessing and document classification."""

import re
import pdfplumber


def clean_ocr_text(raw_text: str) -> str:
    """Clean and normalize raw OCR output."""

    text = raw_text

    # Fix common OCR character substitutions in numeric contexts
    text = re.sub(r'(?<=\d)O(?=\d)', '0', text)
    text = re.sub(r'(?<=\d)l(?=\d)', '1', text)
    text = re.sub(r'\bS(?=\d)', '$', text)

    # Normalize currency symbols
    text = text.replace('£', 'GBP ').replace('€', 'EUR ').replace('¥', 'JPY ')

    # Remove lines that are just a standalone number (page numbers)
    lines = text.split('\n')
    lines = [line for line in lines if not re.match(r'^\s*\d{1,3}\s*$', line)]

    # Remove lines that repeat 3+ times (headers/footers)
    from collections import Counter
    stripped_lines = [line.strip() for line in lines]
    counts = Counter(stripped_lines)
    lines = [line for line in lines if counts[line.strip()] < 3 or line.strip() == '']

    # Collapse 3+ blank lines into 2
    text = '\n'.join(lines)
    text = re.sub(r'\n{3,}', '\n\n', text)

    # Normalize whitespace within lines
    result_lines = []
    prev_stripped = None
    for line in text.split('\n'):
        line = re.sub(r'[ \t]{2,}', '  ', line)
        stripped = line.strip()
        if stripped and stripped == prev_stripped:
            continue
        result_lines.append(line)
        prev_stripped = stripped if stripped else prev_stripped
    text = '\n'.join(result_lines)

    return text.strip()


def classify_document(cleaned_text: str) -> dict:
    """Classify document type based on keyword matching."""

    text_lower = cleaned_text.lower()

    invoice_keywords = [
        'invoice', 'invoice number', 'invoice date', 'inv #', 'inv no',
        'bill to', 'ship to', 'amount due', 'subtotal', 'payment terms',
        'purchase order', 'remittance', 'due date', 'tax invoice',
        'please pay', 'total due', 'balance due'
    ]

    receipt_keywords = [
        'receipt', 'thank you for your purchase', 'change due',
        'cashier', 'transaction id', 'payment received', 'amount tendered'
    ]

    po_keywords = [
        'purchase order', 'po number', 'po #', 'requisition',
        'ordered by', 'delivery required', 'order confirmation'
    ]

    bank_keywords = [
        'bank statement', 'account statement', 'opening balance',
        'closing balance', 'deposits', 'withdrawals', 'account number'
    ]

    def count_matches(keywords):
        return sum(1 for kw in keywords if kw in text_lower)

    scores = {
        'invoice': count_matches(invoice_keywords),
        'receipt': count_matches(receipt_keywords),
        'purchase_order': count_matches(po_keywords),
        'bank_statement': count_matches(bank_keywords),
    }

    best_type = max(scores, key=scores.get)
    best_score = scores[best_type]
    total_keywords = {
        'invoice': len(invoice_keywords),
        'receipt': len(receipt_keywords),
        'purchase_order': len(po_keywords),
        'bank_statement': len(bank_keywords),
    }

    if best_score == 0:
        return {'type': 'invoice', 'confidence': 0.3, 'matched_keywords': 0}

    confidence = min(best_score / 4, 1.0)

    # If best match is not invoice but invoice score is close, prefer invoice
    invoice_score = scores['invoice']
    if best_type != 'invoice' and invoice_score >= best_score - 1:
        best_type = 'invoice'
        confidence = min(invoice_score / 4, 1.0) if invoice_score > 0 else 0.3

    return {
        'type': best_type,
        'confidence': round(confidence, 2),
        'matched_keywords': best_score
    }


def extract_pages_pdfplumber(pdf_path: str) -> list[str]:
    """Extract per-page text from a native PDF using pdfplumber."""
    pages = []
    with pdfplumber.open(pdf_path) as pdf:
        for page in pdf.pages:
            text = page.extract_text()
            if text and text.strip():
                pages.append(text)
    return pages


def deduplicate_pages(pages: list[str]) -> str:
    """Keep page 1 fully intact. Strip repeated boilerplate from pages 2+."""
    if not pages:
        return ""
    if len(pages) == 1:
        return pages[0]
    page1_lines = set(line.strip() for line in pages[0].splitlines() if line.strip())
    result = [pages[0]]
    for page in pages[1:]:
        unique_lines = [
            line for line in page.splitlines()
            if line.strip() and line.strip() not in page1_lines
        ]
        if unique_lines:
            result.append("\n".join(unique_lines))
    return "\n\n--- PAGE BREAK ---\n\n".join(result)