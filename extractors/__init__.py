"""Document extractors and unified routing interface."""

from __future__ import annotations

from typing import Callable, Dict

from models.data_models import ExtractionResult
from utils.helpers import detect_file_type, make_document_output_dir


def process_document(file_path: str, output_root: str = "output") -> ExtractionResult:
    file_type = detect_file_type(file_path)
    output_dir = str(make_document_output_dir(file_path, output_root))
    if file_type == "pdf":
        from .pdf_extractor import extract_pdf

        return extract_pdf(file_path, output_dir)
    if file_type == "pptx":
        from .pptx_extractor import extract_pptx

        return extract_pptx(file_path, output_dir)
    if file_type == "docx":
        from .docx_extractor import extract_docx

        return extract_docx(file_path, output_dir)
    if file_type in {"csv", "xlsx", "xls"}:
        from .spreadsheet_extractor import extract_spreadsheet

        return extract_spreadsheet(file_path, output_dir)
    raise ValueError(f"Unsupported file type: {file_type}")


def extract_pdf(file_path: str, output_dir: str) -> ExtractionResult:
    from .pdf_extractor import extract_pdf as _extract_pdf

    return _extract_pdf(file_path, output_dir)


def extract_pptx(file_path: str, output_dir: str) -> ExtractionResult:
    from .pptx_extractor import extract_pptx as _extract_pptx

    return _extract_pptx(file_path, output_dir)


def extract_docx(file_path: str, output_dir: str) -> ExtractionResult:
    from .docx_extractor import extract_docx as _extract_docx

    return _extract_docx(file_path, output_dir)


def extract_spreadsheet(file_path: str, output_dir: str) -> ExtractionResult:
    from .spreadsheet_extractor import extract_spreadsheet as _extract_spreadsheet

    return _extract_spreadsheet(file_path, output_dir)


__all__ = [
    "extract_docx",
    "extract_pdf",
    "extract_pptx",
    "extract_spreadsheet",
    "process_document",
]
