"""Curriculum document extraction and normalization."""

from __future__ import annotations

import io
import re

from .data_sources import DATA_DIR

SUPPORTED_EXTENSIONS = {".pdf", ".docx", ".txt"}
SAMPLE_PATH = DATA_DIR / "sample_welding_curriculum.txt"


def load_sample_curriculum() -> str:
    return SAMPLE_PATH.read_text(encoding="utf-8")


def normalize_text(text: str) -> str:
    text = text.replace("\r\n", "\n").replace("\r", "\n")
    text = re.sub(r"[ \t]+", " ", text)
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()


def _extract_pdf(data: bytes) -> str:
    import pymupdf

    with pymupdf.open(stream=data, filetype="pdf") as doc:
        return "\n".join(page.get_text() for page in doc)


def _extract_docx(data: bytes) -> str:
    from docx import Document

    doc = Document(io.BytesIO(data))
    parts = [p.text for p in doc.paragraphs]
    for table in doc.tables:
        for row in table.rows:
            parts.append(" | ".join(cell.text for cell in row.cells))
    return "\n".join(parts)


def extract_text(file_bytes: bytes, filename: str) -> str:
    """Extract plain text from an uploaded PDF, DOCX, or TXT document.

    Raises:
        ValueError: for unsupported extensions or unreadable files.
    """
    ext = "." + filename.rsplit(".", 1)[-1].lower() if "." in filename else ""
    if ext not in SUPPORTED_EXTENSIONS:
        raise ValueError(
            f"Unsupported file type '{ext or filename}'. "
            f"Supported: {', '.join(sorted(SUPPORTED_EXTENSIONS))}"
        )
    try:
        if ext == ".pdf":
            raw = _extract_pdf(file_bytes)
        elif ext == ".docx":
            raw = _extract_docx(file_bytes)
        else:
            raw = file_bytes.decode("utf-8", errors="replace")
    except Exception as exc:  # noqa: BLE001 - surface as a clean user error
        raise ValueError(f"Could not read '{filename}': {exc}") from exc

    text = normalize_text(raw)
    if len(text) < 40:
        raise ValueError(
            f"'{filename}' produced almost no readable text. "
            "Is the document mostly images or scanned pages?"
        )
    return text
