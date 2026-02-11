from __future__ import annotations

import csv
import json
import logging
import os
import re
from pathlib import Path
from typing import Iterable, List, Sequence

import numpy as np
from PIL import Image

from models.data_models import ExtractionResult, TextBlock

SUPPORTED_EXTENSIONS = {
    ".pdf": "pdf",
    ".pptx": "pptx",
    ".docx": "docx",
    ".csv": "csv",
    ".xlsx": "xlsx",
    ".xls": "xls",
}


def setup_logger(name: str = "doc_pipeline", level: int = logging.INFO) -> logging.Logger:
    logger = logging.getLogger(name)
    if not logger.handlers:
        handler = logging.StreamHandler()
        formatter = logging.Formatter("%(asctime)s [%(levelname)s] %(message)s")
        handler.setFormatter(formatter)
        logger.addHandler(handler)
        logger.setLevel(level)
    return logger


def ensure_dir(path: Path | str) -> Path:
    path_obj = Path(path)
    path_obj.mkdir(parents=True, exist_ok=True)
    return path_obj


def slugify_name(value: str) -> str:
    stem = re.sub(r"[^A-Za-z0-9]+", "_", value).strip("_")
    return stem.lower() or "document"


def detect_file_type(file_path: str | Path) -> str:
    ext = Path(file_path).suffix.lower()
    if ext not in SUPPORTED_EXTENSIONS:
        raise ValueError(f"Unsupported file extension: {ext}")
    return SUPPORTED_EXTENSIONS[ext]


def make_document_output_dir(file_path: str | Path, output_root: str | Path) -> Path:
    file_name = Path(file_path).stem
    folder_name = f"{slugify_name(file_name)}_{Path(file_path).suffix.lower().lstrip('.')}"
    return ensure_dir(Path(output_root) / folder_name)


def detect_csv_encoding(file_path: str | Path, sample_bytes: int = 65536) -> str:
    with open(file_path, "rb") as f:
        raw = f.read(sample_bytes)
    try:
        import chardet

        detected = chardet.detect(raw)
        return detected.get("encoding") or "utf-8"
    except Exception:
        return "utf-8"


def detect_csv_delimiter(file_path: str | Path, encoding: str, sample_chars: int = 16384) -> str:
    with open(file_path, "r", encoding=encoding, newline="") as f:
        sample = f.read(sample_chars)
    try:
        sniffed = csv.Sniffer().sniff(sample, delimiters=[",", "\t", ";", "|"])
        return sniffed.delimiter
    except csv.Error:
        return ","


def extract_source_context(text_blocks: Sequence[TextBlock], page_or_slide: int, window: int = 2) -> str:
    page_text = [tb.content for tb in text_blocks if tb.page_or_slide == page_or_slide]
    if not page_text:
        return ""
    combined = "\n".join(page_text[: max(window, 1) * 4])
    return combined[:2000]


def dedupe_key_from_pil(image: Image.Image, hash_size: int = 8) -> str:
    # A compact average-hash for quick near-duplicate suppression.
    gray = image.convert("L").resize((hash_size, hash_size))
    arr = np.asarray(gray, dtype=np.float32)
    mean = float(arr.mean())
    bits = (arr > mean).astype(np.uint8).flatten()
    return "".join(str(int(v)) for v in bits)


def cosine_similarity(query: np.ndarray, matrix: np.ndarray) -> np.ndarray:
    q = query / (np.linalg.norm(query) + 1e-12)
    m = matrix / (np.linalg.norm(matrix, axis=1, keepdims=True) + 1e-12)
    return np.dot(m, q)


def _write_tables(tables_dir: Path, rows: List[List[str]], name: str) -> None:
    with open(tables_dir / name, "w", encoding="utf-8", newline="") as f:
        writer = csv.writer(f)
        writer.writerows(rows)


def save_extraction_outputs(result: ExtractionResult, output_dir: str | Path) -> None:
    output_path = ensure_dir(output_dir)
    texts_dir = ensure_dir(output_path / "texts")
    tables_dir = ensure_dir(output_path / "tables")

    extraction_json_path = output_path / "extraction_result.json"
    with open(extraction_json_path, "w", encoding="utf-8") as f:
        json.dump(result.model_dump(mode="json"), f, indent=2, ensure_ascii=False, default=str)

    full_text = "\n\n".join(block.content for block in result.texts if block.content.strip())
    with open(texts_dir / "full_text.txt", "w", encoding="utf-8") as f:
        f.write(full_text)

    for idx, table in enumerate(result.tables, start=1):
        name = f"page{table.page_or_slide}_table{idx}.csv"
        _write_tables(tables_dir, table.data, name)
