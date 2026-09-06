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
import pytesseract
from pathlib import Path
from pdf2image import convert_from_path, convert_from_bytes

# Below this many characters per page, we assume there's no real text
# layer and this is a scanned image rather than a text-based PDF.
MIN_CHARS_PER_PAGE = 50


def extract_text_from_pdf(pdf_source) -> str:
    # pypdf.PdfReader already accepts a path string OR a file-like object,
    # so this line needs no change regardless of what pdf_source is.
    reader = PdfReader(pdf_source)
    text = "\n".join(page.extract_text() or "" for page in reader.pages)

    if _looks_like_scanned(text, len(reader.pages)):
        if isinstance(pdf_source, (str, Path)):
            images = convert_from_path(pdf_source)
        else:
            # PdfReader already consumed part of the stream above,
            # so rewind before reading it again for OCR.
            pdf_source.seek(0)
            images = convert_from_bytes(pdf_source.read())
        text = "\n".join(pytesseract.image_to_string(img) for img in images)

    return text


def _looks_like_scanned(text: str, num_pages: int) -> bool:
    avg_chars_per_page = len(text) / max(num_pages, 1)
    return avg_chars_per_page < MIN_CHARS_PER_PAGE


def _extract_with_ocr(pdf_path: str) -> str:
    images = convert_from_path(pdf_path)
    pages_text = [pytesseract.image_to_string(image) for image in images]
    return "\n".join(pages_text)