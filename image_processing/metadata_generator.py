from __future__ import annotations

import json
import re
from functools import lru_cache
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
from PIL import Image
from tqdm import tqdm

from models.data_models import ExtractionResult, ImageMetadata
from utils.helpers import extract_source_context, setup_logger

VISION_PROMPT = (
    "Describe this image in detail. Include image type (photo, diagram, chart, schematic, etc.), "
    "visible objects/elements, text in the image, relationships between elements, and likely purpose."
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
    def __init__(self) -> None:
        self.logger = setup_logger()
        self.mode = "rule_based"
        self.moondream_model = None
        self.moondream_tokenizer = None
        self.caption_pipeline = None
        self._initialized = False

    def _init(self) -> None:
        if self._initialized:
            return
        self._initialized = True

        # Preferred: moondream2 (trust_remote_code required by upstream repository).
        try:
            from transformers import AutoModelForCausalLM, AutoTokenizer

            self.moondream_model = AutoModelForCausalLM.from_pretrained(
                "vikhyatk/moondream2",
                trust_remote_code=True,
            )
            self.moondream_tokenizer = AutoTokenizer.from_pretrained(
                "vikhyatk/moondream2",
                trust_remote_code=True,
            )
            self.mode = "moondream2"
            self.logger.info("Using vision model: moondream2")
            return
        except Exception as exc:
            self.logger.warning("moondream2 unavailable; fallback to BLIP pipeline. Reason: %s", exc)

        # Fallback: BLIP image-to-text via transformers pipeline.
        try:
            from transformers import pipeline

            self.caption_pipeline = pipeline("image-to-text", model="Salesforce/blip-image-captioning-base")
            self.mode = "blip"
            self.logger.info("Using vision model fallback: BLIP")
            return
        except Exception as exc:
            self.logger.warning("BLIP pipeline unavailable; using rule-based metadata. Reason: %s", exc)

        self.mode = "rule_based"

    def describe(self, image: Image.Image) -> str:
        self._init()
        if self.mode == "moondream2" and self.moondream_model is not None:
            try:
                if hasattr(self.moondream_model, "encode_image") and hasattr(
                    self.moondream_model, "answer_question"
                ):
                    encoded = self.moondream_model.encode_image(image)
                    answer = self.moondream_model.answer_question(
                        encoded,
                        VISION_PROMPT,
                        self.moondream_tokenizer,
                    )
                    if isinstance(answer, str) and answer.strip():
                        return answer.strip()
                return ""
            except Exception:
                return ""

        if self.mode == "blip" and self.caption_pipeline is not None:
            try:
                output = self.caption_pipeline(image)
                if output and isinstance(output, list):
                    text = output[0].get("generated_text", "").strip()
                    return text
                return ""
            except Exception:
                return ""

        return ""


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

        description = captioner.describe(image).strip()
        ocr = _ocr_text(image)
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
