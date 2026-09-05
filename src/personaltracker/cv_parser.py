"""
CV parsing: extract raw text from a candidate's CV so it can later be
compared against a skill roadmap.

Two paths, tried in order:
1. Direct text extraction (pypdf) -- works for "real" PDFs that have an
   actual text layer (exported from Word, Google Docs, Canva, etc).
2. OCR fallback (pdf2image + pytesseract) -- for scanned CVs, where the
   PDF is just an image of a paper document with no text layer at all.
   pypdf won't error on these, it'll just return almost nothing -- so we
   detect "scanned" by checking how much text came back, not by
   catching an exception.
"""

import pypdf
from pdf2image import convert_from_path
import pytesseract

# Below this many characters per page, we assume there's no real text
# layer and this is a scanned image rather than a text-based PDF.
MIN_CHARS_PER_PAGE = 50


def extract_text_from_pdf(pdf_path: str) -> str:
    reader = pypdf.PdfReader(pdf_path)
    num_pages = len(reader.pages)
    pages_text = [page.extract_text() or "" for page in reader.pages]
    text = "\n".join(pages_text)

    if _looks_like_scanned(text, num_pages):
        print("No real text layer found -- falling back to OCR (this is slower).")
        text = _extract_with_ocr(pdf_path)

    return text.strip()


def _looks_like_scanned(text: str, num_pages: int) -> bool:
    avg_chars_per_page = len(text) / max(num_pages, 1)
    return avg_chars_per_page < MIN_CHARS_PER_PAGE


def _extract_with_ocr(pdf_path: str) -> str:
    images = convert_from_path(pdf_path)
    pages_text = [pytesseract.image_to_string(image) for image in images]
    return "\n".join(pages_text)