from __future__ import annotations

import io
import re
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import fitz  # PyMuPDF
import pdfplumber
from PIL import Image
from tqdm import tqdm

from models.data_models import ExtractionResult, ImageBlock, TableBlock, TextBlock
from utils.helpers import dedupe_key_from_pil, ensure_dir, setup_logger


def _sorted_text_blocks(page: fitz.Page) -> List[Tuple[float, float, float, float, str]]:
    raw_blocks = page.get_text("blocks")
    normalized = []
    for block in raw_blocks:
        if len(block) < 5:
            continue
        x0, y0, x1, y1, text = block[:5]
        text = (text or "").strip()
        if text:
            normalized.append((float(x0), float(y0), float(x1), float(y1), text))
    normalized.sort(key=lambda item: (round(item[1], 2), round(item[0], 2)))
    return normalized


def _extract_table_heuristic(page: fitz.Page) -> List[List[List[str]]]:
    lines = [ln.strip() for ln in page.get_text("text").splitlines() if ln.strip()]
    candidates = []
    current = []
    for ln in lines:
        if re.search(r"\s{2,}", ln):
            parts = [part.strip() for part in re.split(r"\s{2,}", ln) if part.strip()]
            if len(parts) >= 2:
                current.append(parts)
        elif current:
            if len(current) >= 2:
                candidates.append(current)
            current = []
    if current and len(current) >= 2:
        candidates.append(current)
    return candidates


def _rect_to_dict(rect: fitz.Rect | None) -> Optional[Dict[str, float]]:
    if rect is None:
        return None
    return {"x0": float(rect.x0), "y0": float(rect.y0), "x1": float(rect.x1), "y1": float(rect.y1)}


def _union_rect(rects: Sequence[fitz.Rect]) -> Optional[fitz.Rect]:
    if not rects:
        return None
    x0 = min(r.x0 for r in rects)
    y0 = min(r.y0 for r in rects)
    x1 = max(r.x1 for r in rects)
    y1 = max(r.y1 for r in rects)
    return fitz.Rect(x0, y0, x1, y1)


def _rect_area(rect: fitz.Rect | None) -> float:
    if rect is None:
        return 0.0
    return max(rect.width, 0.0) * max(rect.height, 0.0)


def _rect_iou(a: fitz.Rect, b: fitz.Rect) -> float:
    inter = fitz.Rect(max(a.x0, b.x0), max(a.y0, b.y0), min(a.x1, b.x1), min(a.y1, b.y1))
    inter_area = _rect_area(inter) if inter.x1 > inter.x0 and inter.y1 > inter.y0 else 0.0
    if inter_area <= 0:
        return 0.0
    union = _rect_area(a) + _rect_area(b) - inter_area
    return inter_area / union if union > 0 else 0.0


def _header_from_rows(rows: List[List[str]]) -> Optional[List[str]]:
    if not rows:
        return None
    header = rows[0]
    if not header:
        return None
    non_numeric = 0
    for cell in header:
        val = str(cell).strip()
        if val and not re.fullmatch(r"[-+]?\d*\.?\d+(?:[eE][-+]?\d+)?", val):
            non_numeric += 1
    return header if non_numeric >= max(1, len(header) // 2) else None


def _extract_embedded_images(
    doc: fitz.Document,
    page: fitz.Page,
    page_num: int,
    images_dir: Path,
    dedupe_hashes: set[str],
) -> Tuple[List[ImageBlock], List[fitz.Rect]]:
    image_blocks: List[ImageBlock] = []
    embedded_rects: List[fitz.Rect] = []
    image_counter = 0

    seen_xref = set()
    for img in page.get_images(full=True):
        if not img:
            continue
        xref = img[0]
        if xref in seen_xref:
            continue
        seen_xref.add(xref)

        try:
            extracted = doc.extract_image(xref)
            if not extracted:
                continue
            image_bytes = extracted.get("image")
            image_ext = (extracted.get("ext") or "png").lower()
            pil_image = Image.open(io.BytesIO(image_bytes)).convert("RGB")

            dedupe_key = dedupe_key_from_pil(pil_image)
            if dedupe_key in dedupe_hashes:
                continue
            dedupe_hashes.add(dedupe_key)

            image_counter += 1
            image_name = f"page{page_num}_img{image_counter}.{image_ext}"
            image_path = images_dir / image_name
            pil_image.save(image_path)

            rects = page.get_image_rects(xref)
            rect = rects[0] if rects else None
            if rect is not None:
                embedded_rects.append(rect)

            image_blocks.append(
                ImageBlock(
                    image_path=str(image_path),
                    page_or_slide=page_num,
                    bounding_box=_rect_to_dict(rect),
                    dimensions={"width": pil_image.width, "height": pil_image.height},
                    extraction_method="embedded",
                )
            )
        except Exception:
            continue

    return image_blocks, embedded_rects


def _extract_vector_diagram(
    page: fitz.Page,
    page_num: int,
    images_dir: Path,
    embedded_rects: Sequence[fitz.Rect],
    dedupe_hashes: set[str],
    drawing_ops_threshold: int,
    drawing_area_threshold: float,
    diagram_counter: int,
) -> Tuple[Optional[ImageBlock], int]:
    drawings = page.get_drawings()
    if not drawings:
        return None, diagram_counter

    ops_count = sum(max(len(d.get("items", [])), 1) for d in drawings)
    rects = [d["rect"] for d in drawings if d.get("rect") is not None]
    if not rects:
        return None, diagram_counter

    union = _union_rect(rects)
    page_area = max(page.rect.width * page.rect.height, 1.0)
    coverage = _rect_area(union) / page_area

    is_significant = ops_count > drawing_ops_threshold or coverage > drawing_area_threshold
    if not is_significant or union is None:
        return None, diagram_counter

    if any(_rect_iou(union, er) > 0.4 for er in embedded_rects):
        return None, diagram_counter

    pix = page.get_pixmap(dpi=300, alpha=False)
    pil = Image.frombytes("RGB", (pix.width, pix.height), pix.samples)
    scale_x = pix.width / max(page.rect.width, 1e-9)
    scale_y = pix.height / max(page.rect.height, 1e-9)
    crop_box = (
        int(max(0, union.x0 * scale_x)),
        int(max(0, union.y0 * scale_y)),
        int(min(pix.width, union.x1 * scale_x)),
        int(min(pix.height, union.y1 * scale_y)),
    )
    if crop_box[2] - crop_box[0] > 20 and crop_box[3] - crop_box[1] > 20:
        pil = pil.crop(crop_box)

    dedupe_key = dedupe_key_from_pil(pil)
    if dedupe_key in dedupe_hashes:
        return None, diagram_counter
    dedupe_hashes.add(dedupe_key)

    diagram_counter += 1
    out_name = f"page{page_num}_diagram{diagram_counter}.png"
    out_path = images_dir / out_name
    pil.save(out_path)

    block = ImageBlock(
        image_path=str(out_path),
        page_or_slide=page_num,
        bounding_box=_rect_to_dict(union),
        dimensions={"width": pil.width, "height": pil.height},
        extraction_method="vector_render",
    )
    return block, diagram_counter


def extract_pdf(
    file_path: str | Path,
    output_dir: str | Path,
    drawing_ops_threshold: int = 20,
    drawing_area_threshold: float = 0.15,
) -> ExtractionResult:
    logger = setup_logger()
    file_path = Path(file_path)
    base_output = ensure_dir(output_dir)
    images_dir = ensure_dir(base_output / "images")

    texts: List[TextBlock] = []
    tables: List[TableBlock] = []
    images: List[ImageBlock] = []
    warnings: List[str] = []
    dedupe_hashes: set[str] = set()

    try:
        pdf_doc = fitz.open(file_path)
    except Exception as exc:
        raise RuntimeError(f"Failed to open PDF: {file_path}") from exc

    num_pages = pdf_doc.page_count

    with pdfplumber.open(file_path) as plumber_doc:
        diagram_counter = 0
        for page_index in tqdm(range(num_pages), desc=f"Extracting PDF {file_path.name}"):
            page_num = page_index + 1
            try:
                page = pdf_doc[page_index]

                for _, _, _, _, content in _sorted_text_blocks(page):
                    texts.append(TextBlock(content=content, page_or_slide=page_num, block_type="paragraph"))

                plumber_page = plumber_doc.pages[page_index] if page_index < len(plumber_doc.pages) else None
                page_tables = plumber_page.extract_tables() if plumber_page else []
                if page_tables:
                    for tbl in page_tables:
                        cleaned = [[str(cell or "").strip() for cell in row] for row in tbl if row is not None]
                        if not cleaned:
                            continue
                        headers = _header_from_rows(cleaned)
                        tables.append(
                            TableBlock(
                                data=cleaned,
                                headers=headers,
                                page_or_slide=page_num,
                                num_rows=len(cleaned),
                                num_cols=max((len(r) for r in cleaned), default=0),
                            )
                        )
                else:
                    heuristic_tables = _extract_table_heuristic(page)
                    for tbl in heuristic_tables:
                        headers = _header_from_rows(tbl)
                        tables.append(
                            TableBlock(
                                data=tbl,
                                headers=headers,
                                page_or_slide=page_num,
                                num_rows=len(tbl),
                                num_cols=max((len(r) for r in tbl), default=0),
                            )
                        )

                embedded_blocks, embedded_rects = _extract_embedded_images(
                    pdf_doc, page, page_num, images_dir, dedupe_hashes
                )
                images.extend(embedded_blocks)

                vector_block, diagram_counter = _extract_vector_diagram(
                    page=page,
                    page_num=page_num,
                    images_dir=images_dir,
                    embedded_rects=embedded_rects,
                    dedupe_hashes=dedupe_hashes,
                    drawing_ops_threshold=drawing_ops_threshold,
                    drawing_area_threshold=drawing_area_threshold,
                    diagram_counter=diagram_counter,
                )
                if vector_block is not None:
                    images.append(vector_block)
            except Exception as exc:
                warning = f"Page {page_num} extraction failed: {exc}"
                warnings.append(warning)
                logger.warning(warning)

    pdf_doc.close()

    return ExtractionResult(
        file_name=file_path.name,
        file_type="pdf",
        extraction_timestamp=datetime.now(timezone.utc),
        num_pages_or_slides=num_pages,
        texts=texts,
        tables=tables,
        images=images,
        warnings=warnings,
    )
