from __future__ import annotations

import json
from functools import lru_cache
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
from PIL import Image
from tqdm import tqdm

from models.data_models import ExtractionResult, ImageMetadata
from utils.helpers import setup_logger


class ClipEmbedder:
    def __init__(self) -> None:
        self.logger = setup_logger()
        self.backend = "none"
        self.initialized = False

        self.device = "cpu"
        self._torch = None

        # OpenCLIP backend
        self.oc_model = None
        self.oc_preprocess = None
        self.oc_tokenizer = None

        # Transformers CLIP fallback
        self.tf_model = None
        self.tf_processor = None

    def _init(self) -> None:
        if self.initialized:
            return
        self.initialized = True

        try:
            import torch

            self._torch = torch
            self.device = "cuda" if torch.cuda.is_available() else "cpu"
        except Exception:
            self.device = "cpu"
            self._torch = None

        try:
            import open_clip

            model, _, preprocess = open_clip.create_model_and_transforms(
                "ViT-B-32",
                pretrained="laion2b_s34b_b79k",
            )
            model.eval()
            if self._torch is not None:
                model = model.to(self.device)
            self.oc_model = model
            self.oc_preprocess = preprocess
            self.oc_tokenizer = open_clip.get_tokenizer("ViT-B-32")
            self.backend = "open_clip"
            self.logger.info("Using OpenCLIP backend for embeddings.")
            return
        except Exception as exc:
            self.logger.warning("OpenCLIP unavailable; trying transformers CLIP fallback. Reason: %s", exc)

        try:
            from transformers import CLIPModel, CLIPProcessor

            self.tf_model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32")
            self.tf_processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")
            if self._torch is not None:
                self.tf_model = self.tf_model.to(self.device)
            self.backend = "transformers_clip"
            self.logger.info("Using transformers CLIP backend for embeddings.")
            return
        except Exception as exc:
            self.logger.warning("No CLIP backend available, embeddings will be zeros. Reason: %s", exc)
            self.backend = "none"

    def _normalize(self, vec: np.ndarray) -> np.ndarray:
        norm = np.linalg.norm(vec) + 1e-12
        return vec / norm

    def encode_image(self, image_path: str | Path) -> np.ndarray:
        self._init()
        if self.backend == "none":
            return np.zeros(512, dtype=np.float32)

        image = Image.open(image_path).convert("RGB")

        if self.backend == "open_clip":
            assert self.oc_model is not None and self.oc_preprocess is not None and self._torch is not None
            with self._torch.no_grad():
                tensor = self.oc_preprocess(image).unsqueeze(0).to(self.device)
                feat = self.oc_model.encode_image(tensor).detach().cpu().numpy()[0]
            return self._normalize(feat.astype(np.float32))

        assert self.tf_model is not None and self.tf_processor is not None and self._torch is not None
        with self._torch.no_grad():
            inputs = self.tf_processor(images=image, return_tensors="pt").to(self.device)
            feat = self.tf_model.get_image_features(**inputs).detach().cpu().numpy()[0]
        return self._normalize(feat.astype(np.float32))

    def encode_text(self, text: str) -> np.ndarray:
        self._init()
        if self.backend == "none":
            return np.zeros(512, dtype=np.float32)

        text = text or ""
        if self.backend == "open_clip":
            assert self.oc_model is not None and self.oc_tokenizer is not None and self._torch is not None
            with self._torch.no_grad():
                token = self.oc_tokenizer([text]).to(self.device)
                feat = self.oc_model.encode_text(token).detach().cpu().numpy()[0]
            return self._normalize(feat.astype(np.float32))

        assert self.tf_model is not None and self.tf_processor is not None and self._torch is not None
        with self._torch.no_grad():
            inputs = self.tf_processor(text=[text], return_tensors="pt", padding=True).to(self.device)
            feat = self.tf_model.get_text_features(**inputs).detach().cpu().numpy()[0]
        return self._normalize(feat.astype(np.float32))


@lru_cache(maxsize=1)
def _embedder() -> ClipEmbedder:
    return ClipEmbedder()


def _metadata_text(metadata: ImageMetadata | None) -> str:
    if metadata is None:
        return ""
    chunks = [
        metadata.description or "",
        metadata.detected_text or "",
        " ".join(metadata.tags or []),
        " ".join(metadata.objects_detected or []),
        metadata.source_context or "",
    ]
    return "\n".join(chunk for chunk in chunks if chunk.strip())


def build_index_for_directory(input_dir: str | Path) -> Tuple[Path, Path]:
    logger = setup_logger()
    root = Path(input_dir)
    result_path = root / "extraction_result.json"
    if not result_path.exists():
        raise FileNotFoundError(f"Expected extraction_result.json under {root}")

    with open(result_path, "r", encoding="utf-8") as f:
        payload = json.load(f)
    result = ExtractionResult.model_validate(payload)

    embedder = _embedder()
    image_vecs: List[np.ndarray] = []
    text_vecs: List[np.ndarray] = []
    hybrid_vecs: List[np.ndarray] = []
    index_payload: Dict[str, Dict] = {}

    for idx, image_block in enumerate(tqdm(result.images, desc="Building embedding index")):
        img_path = Path(image_block.image_path)
        if not img_path.exists():
            logger.warning("Image missing during indexing: %s", img_path)
            continue

        img_vec = embedder.encode_image(img_path)
        txt_vec = embedder.encode_text(_metadata_text(image_block.metadata))
        combo_vec = 0.6 * img_vec + 0.4 * txt_vec
        combo_vec = combo_vec / (np.linalg.norm(combo_vec) + 1e-12)

        image_vecs.append(img_vec.astype(np.float32))
        text_vecs.append(txt_vec.astype(np.float32))
        hybrid_vecs.append(combo_vec.astype(np.float32))

        index_payload[f"image_{len(index_payload)}"] = {
            "image_path": image_block.image_path,
            "source_page": image_block.page_or_slide,
            "source_file": result.file_name,
            "metadata": image_block.metadata.model_dump() if image_block.metadata else {},
            "extraction_method": image_block.extraction_method,
        }

    if image_vecs:
        image_arr = np.vstack(image_vecs)
        text_arr = np.vstack(text_vecs)
        hybrid_arr = np.vstack(hybrid_vecs)
    else:
        image_arr = np.zeros((0, 512), dtype=np.float32)
        text_arr = np.zeros((0, 512), dtype=np.float32)
        hybrid_arr = np.zeros((0, 512), dtype=np.float32)

    embeddings_path = root / "embeddings.npz"
    np.savez_compressed(
        embeddings_path,
        image_embeddings=image_arr,
        text_embeddings=text_arr,
        hybrid_embeddings=hybrid_arr,
    )

    index_path = root / "image_index.json"
    with open(index_path, "w", encoding="utf-8") as f:
        json.dump(index_payload, f, indent=2, ensure_ascii=False)

    logger.info("Saved embeddings to %s and index to %s", embeddings_path, index_path)
    return embeddings_path, index_path
