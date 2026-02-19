from __future__ import annotations

import json
import re
from functools import lru_cache
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
from PIL import Image
from tqdm import tqdm

from ..models.data_models import ExtractionResult, ImageMetadata
from ..utils.helpers import extract_source_context, setup_logger

MOONDREAM_MODEL_ID = "vikhyatk/moondream2"
MOONDREAM_REVISION = "2025-06-21"

DESCRIPTION_PROMPT = (
    "Describe this image in detail. Include the image type "
    "(photo, diagram, chart, schematic, table, etc.), visible objects/elements, "
    "relationships between elements, and likely purpose."
)

OCR_PROMPT = (
    "Transcribe all visible text exactly as written. "
    "If no text is visible, answer with NONE."
)

STOPWORDS = {
    "the",
    "and",
    "that",
    "with",
    "from",
    "this",
    "there",
    "their",
    "into",
    "over",
    "under",
    "image",
    "shows",
    "visible",
    "appears",
}


class LocalVisionCaptioner:
    """Vision captioner using Moondream2 (primary) with BLIP fallback."""

    def __init__(self) -> None:
        self.logger = setup_logger()
        self.mode = "rule_based"
        self._moondream_model = None
        self._caption_pipeline = None  # BLIP fallback
        self._initialized = False

    # ------------------------------------------------------------------
    # Lazy model loading
    # ------------------------------------------------------------------

    def _init(self) -> None:
        if self._initialized:
            return
        self._initialized = True

        # Primary: Moondream2
        try:
            import torch
            from transformers import AutoModelForCausalLM

            if torch.cuda.is_available():
                device = "cuda"
            elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
                device = "mps"
            else:
                device = "cpu"

            self._moondream_model = AutoModelForCausalLM.from_pretrained(
                MOONDREAM_MODEL_ID,
                revision=MOONDREAM_REVISION,
                trust_remote_code=True,
                device_map={"": device},
            )
            if hasattr(self._moondream_model, "eval"):
                self._moondream_model.eval()
            self.mode = "moondream2"
            self.logger.info(
                "Using vision model: %s (revision=%s) on %s",
                MOONDREAM_MODEL_ID,
                MOONDREAM_REVISION,
                device,
            )
            return
        except Exception as exc:
            self.logger.warning("Moondream2 unavailable; fallback to BLIP pipeline. Reason: %s", exc)

        # Fallback: BLIP image-to-text via transformers pipeline.
        try:
            from transformers import pipeline

            self._caption_pipeline = pipeline(
                "image-to-text", model="Salesforce/blip-image-captioning-base"
            )
            self.mode = "blip"
            self.logger.info("Using vision model fallback: BLIP")
            return
        except Exception as exc:
            self.logger.warning("BLIP pipeline unavailable; using rule-based metadata. Reason: %s", exc)

        self.mode = "rule_based"

    # ------------------------------------------------------------------
    # Moondream2 inference
    # ------------------------------------------------------------------

    @staticmethod
    def _extract_value(response: object, key: str) -> str:
        if isinstance(response, dict):
            value = response.get(key, response.get("answer", response.get("text", "")))
            if value is None:
                return ""
            return value.strip() if isinstance(value, str) else str(value).strip()
        if isinstance(response, str):
            return response.strip()
        return ""

    # ------------------------------------------------------------------
    # Response cleanup
    # ------------------------------------------------------------------

    @staticmethod
    def _normalize_ocr(response: str) -> str:
        text = (response or "").strip()
        normalized = text.upper().replace(".", "").strip()
        if normalized in {"NONE", "N/A", "NO TEXT", "NO TEXT FOUND", "EMPTY"}:
            return ""
        return text

    def _moondream_analyze(self, image: Image.Image) -> Tuple[str, str]:
        """Run Moondream2 and return (description, extracted_text)."""
        encoded = self._moondream_model.encode_image(image)
        desc_raw = self._moondream_model.query(encoded, DESCRIPTION_PROMPT)
        text_raw = self._moondream_model.query(encoded, OCR_PROMPT)
        description = self._extract_value(desc_raw, "answer")
        extracted_text = self._normalize_ocr(self._extract_value(text_raw, "answer"))
        return description, extracted_text

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def analyze(self, image: Image.Image) -> Tuple[str, str]:
        """Return *(description, extracted_text)* from model calls.

        * **Moondream2**: query for description + OCR text.
        * **BLIP fallback**: returns description only (extracted_text = "").
        * **Rule-based**: returns ("", "").
        """
        self._init()

        if self.mode == "moondream2" and self._moondream_model is not None:
            try:
                return self._moondream_analyze(image)
            except Exception as exc:
                self.logger.warning("Moondream2 inference failed: %s", exc)
                return "", ""

        if self.mode == "blip" and self._caption_pipeline is not None:
            try:
                output = self._caption_pipeline(image)
                if output and isinstance(output, list):
                    desc = output[0].get("generated_text", "").strip()
                    return desc, ""
                return "", ""
            except Exception:
                return "", ""

        return "", ""

    def describe(self, image: Image.Image) -> str:
        """Backward-compatible wrapper — returns description only."""
        desc, _ = self.analyze(image)
        return desc


@lru_cache(maxsize=1)
def _captioner() -> LocalVisionCaptioner:
    return LocalVisionCaptioner()


def _ocr_text(image: Image.Image) -> str:
    try:
        import pytesseract

        text = pytesseract.image_to_string(image)
        return (text or "").strip()
    except Exception:
        return ""


def _edge_density(image: Image.Image) -> float:
    try:
        import cv2

        arr = np.array(image.convert("L"))
        edges = cv2.Canny(arr, threshold1=80, threshold2=200)
        return float((edges > 0).mean())
    except Exception:
        arr = np.array(image.convert("L"), dtype=np.float32)
        gx = np.abs(np.diff(arr, axis=1)).mean() if arr.shape[1] > 1 else 0.0
        gy = np.abs(np.diff(arr, axis=0)).mean() if arr.shape[0] > 1 else 0.0
        return float((gx + gy) / 510.0)


def _classify_image_type(description: str, detected_text: str, image: Image.Image) -> str:
    text_lower = f"{description} {detected_text}".lower()
    if any(k in text_lower for k in ["schematic", "wiring", "circuit"]):
        return "schematic"
    if any(k in text_lower for k in ["chart", "graph", "plot", "axis"]):
        return "chart"
    if any(k in text_lower for k in ["table", "spreadsheet"]):
        return "table"
    if any(k in text_lower for k in ["diagram", "flowchart", "block"]):
        return "diagram"

    ocr_density = len(detected_text) / max(image.width * image.height, 1)
    edges = _edge_density(image)
    if ocr_density > 0.00035 and edges > 0.04:
        return "diagram"
    if ocr_density > 0.00045:
        return "table"
    if edges < 0.02:
        return "photograph"
    return "other"


def _extract_objects_and_tags(description: str, detected_text: str, image_type: str) -> Tuple[List[str], List[str]]:
    tokens = re.findall(r"[A-Za-z][A-Za-z0-9_-]{2,}", f"{description} {detected_text}".lower())
    filtered = [t for t in tokens if t not in STOPWORDS]
    freq: Dict[str, int] = {}
    for token in filtered:
        freq[token] = freq.get(token, 0) + 1
    ranked = sorted(freq.items(), key=lambda item: (-item[1], item[0]))
    tags = [word for word, _ in ranked[:12]]
    objects = tags[:8]
    if image_type not in tags:
        tags.insert(0, image_type)
    return objects, tags


def enrich_extraction_with_metadata(result: ExtractionResult) -> ExtractionResult:
    captioner = _captioner()

    for image_block in tqdm(result.images, desc="Generating image metadata"):
        image_path = Path(image_block.image_path)
        if not image_path.exists():
            image_block.metadata = ImageMetadata(
                description="Image file not found during metadata generation.",
                detected_text="",
                image_type="other",
                objects_detected=[],
                source_context=extract_source_context(result.texts, image_block.page_or_slide),
                tags=["missing_image"],
            )
            continue

        try:
            image = Image.open(image_path).convert("RGB")
        except Exception:
            image_block.metadata = ImageMetadata(
                description="Failed to read image bytes.",
                detected_text="",
                image_type="other",
                objects_detected=[],
                source_context=extract_source_context(result.texts, image_block.page_or_slide),
                tags=["read_error"],
            )
            continue

        description, vlm_ocr = captioner.analyze(image)
        description = description.strip()
        # Moondream2 can return OCR text via prompting; fall back to Tesseract
        # only when the VLM returned nothing (e.g. BLIP mode or failure).
        ocr = vlm_ocr.strip() if vlm_ocr.strip() else _ocr_text(image)
        if not description:
            description = (
                "Rule-based description: technical visual with extracted OCR and structural features only."
                if ocr
                else "Rule-based description: visual content without model caption."
            )

        source_context = extract_source_context(result.texts, image_block.page_or_slide)
        image_type = _classify_image_type(description, ocr, image)
        objects, tags = _extract_objects_and_tags(description, ocr, image_type)

        image_block.metadata = ImageMetadata(
            description=description,
            detected_text=ocr,
            image_type=image_type,
            objects_detected=objects,
            source_context=source_context,
            tags=tags,
        )

    return result


def generate_metadata_for_directory(input_dir: str | Path) -> Path:
    logger = setup_logger()
    input_path = Path(input_dir)
    result_path = input_path / "extraction_result.json"
    if not result_path.exists():
        raise FileNotFoundError(f"Expected extraction_result.json under {input_path}")

    with open(result_path, "r", encoding="utf-8") as f:
        payload = json.load(f)
    result = ExtractionResult.model_validate(payload)
    result = enrich_extraction_with_metadata(result)

    with open(result_path, "w", encoding="utf-8") as f:
        json.dump(result.model_dump(mode="json"), f, indent=2, ensure_ascii=False, default=str)

    # Keep a lightweight image index usable before embedding generation.
    index_path = input_path / "image_index.json"
    index_payload = {}
    for idx, img in enumerate(result.images):
        index_payload[f"image_{idx}"] = {
            "image_path": img.image_path,
            "page_or_slide": img.page_or_slide,
            "metadata": img.metadata.model_dump() if img.metadata else {},
            "extraction_method": img.extraction_method,
            "source_file": result.file_name,
        }
    with open(index_path, "w", encoding="utf-8") as f:
        json.dump(index_payload, f, indent=2, ensure_ascii=False)

    logger.info("Metadata generated for %d images in %s", len(result.images), input_path)
    return result_path

