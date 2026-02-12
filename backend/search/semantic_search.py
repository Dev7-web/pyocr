from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, List

import numpy as np

from ..image_processing.embedding_generator import _embedder
from ..models.data_models import ImageMetadata, SearchResult
from ..utils.helpers import cosine_similarity


def _resolve_index_paths(index_path: str | Path) -> tuple[Path, Path]:
    base = Path(index_path)
    if base.is_dir():
        embeddings_path = base / "embeddings.npz"
        index_json_path = base / "image_index.json"
    else:
        if base.name == "image_index.json":
            embeddings_path = base.parent / "embeddings.npz"
            index_json_path = base
        elif base.suffix == ".npz":
            embeddings_path = base
            index_json_path = base.parent / "image_index.json"
        else:
            raise ValueError("index_path must be a directory, image_index.json, or embeddings.npz path.")

    if not embeddings_path.exists():
        raise FileNotFoundError(f"Missing embeddings file: {embeddings_path}")
    if not index_json_path.exists():
        raise FileNotFoundError(f"Missing image index file: {index_json_path}")
    return embeddings_path, index_json_path


def _embedding_matrix(npz: np.lib.npyio.NpzFile, search_mode: str) -> np.ndarray:
    mode = search_mode.lower()
    if mode == "image_only":
        return npz["image_embeddings"]
    if mode == "text_only":
        return npz["text_embeddings"]
    if mode == "hybrid":
        return npz["hybrid_embeddings"]
    raise ValueError("search_mode must be one of: image_only, text_only, hybrid")


def search_images(
    query: str,
    index_path: str,
    top_k: int = 5,
    search_mode: str = "hybrid",
) -> List[SearchResult]:
    embeddings_path, index_json_path = _resolve_index_paths(index_path)
    vectors = np.load(embeddings_path)
    matrix = _embedding_matrix(vectors, search_mode)

    with open(index_json_path, "r", encoding="utf-8") as f:
        index_payload: Dict[str, Dict] = json.load(f)

    if matrix.shape[0] == 0 or not index_payload:
        return []

    query_vec = _embedder().encode_text(query)
    sims = cosine_similarity(query_vec, matrix)
    order = np.argsort(-sims)

    results: List[SearchResult] = []
    records = list(index_payload.values())
    for idx in order[: max(top_k, 1)]:
        if idx >= len(records):
            continue
        rec = records[idx]
        metadata = ImageMetadata.model_validate(rec.get("metadata", {}))
        results.append(
            SearchResult(
                image_path=rec.get("image_path", ""),
                similarity_score=float(sims[idx]),
                metadata=metadata,
                source_page=int(rec.get("source_page", 0)),
                source_file=rec.get("source_file", ""),
            )
        )

    return results

