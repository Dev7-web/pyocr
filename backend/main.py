from __future__ import annotations

import argparse
from pathlib import Path
from typing import Iterable, List

from .extractors import process_document
from .image_processing.embedding_generator import build_index_for_directory
from .image_processing.metadata_generator import generate_metadata_for_directory
from .search.semantic_search import search_images
from .utils.helpers import SUPPORTED_EXTENSIONS, make_document_output_dir, save_extraction_outputs, setup_logger


def _iter_input_files(file_path: str | None, folder_path: str | None) -> List[Path]:
    if file_path:
        path = Path(file_path)
        if not path.exists():
            raise FileNotFoundError(f"File not found: {path}")
        return [path]

    if not folder_path:
        raise ValueError("Provide either --file or --folder")
    folder = Path(folder_path)
    if not folder.exists():
        raise FileNotFoundError(f"Folder not found: {folder}")
    files = [p for p in folder.rglob("*") if p.is_file() and p.suffix.lower() in SUPPORTED_EXTENSIONS]
    if not files:
        raise ValueError(f"No supported files found in {folder}")
    return files


def cmd_extract(args: argparse.Namespace) -> None:
    logger = setup_logger()
    files = _iter_input_files(args.file, args.folder)

    for file in files:
        result = process_document(str(file), output_root=args.output)
        out_dir = make_document_output_dir(file, args.output)
        save_extraction_outputs(result, out_dir)
        logger.info("Extraction complete for %s -> %s", file.name, out_dir)


def cmd_generate_metadata(args: argparse.Namespace) -> None:
    result_path = generate_metadata_for_directory(args.input)
    setup_logger().info("Metadata updated in %s", result_path)


def cmd_build_index(args: argparse.Namespace) -> None:
    emb_path, index_path = build_index_for_directory(args.input)
    setup_logger().info("Index built: %s | %s", emb_path, index_path)


def cmd_search(args: argparse.Namespace) -> None:
    results = search_images(
        query=args.query,
        index_path=args.index,
        top_k=args.top_k,
        search_mode=args.search_mode,
    )
    if not results:
        print("No results found.")
        return

    print(f"Top {len(results)} results for query: {args.query}")
    for rank, res in enumerate(results, start=1):
        print(f"{rank}. score={res.similarity_score:.4f} path={res.image_path}")
        if res.metadata and res.metadata.description:
            print(f"   type={res.metadata.image_type} tags={', '.join(res.metadata.tags[:6])}")
            print(f"   desc={res.metadata.description[:220]}")


def cmd_pipeline(args: argparse.Namespace) -> None:
    logger = setup_logger()
    files = _iter_input_files(args.file, args.folder)

    for file in files:
        result = process_document(str(file), output_root=args.output)
        out_dir = make_document_output_dir(file, args.output)
        save_extraction_outputs(result, out_dir)
        logger.info("Extracted %s", file.name)

        generate_metadata_for_directory(out_dir)
        logger.info("Generated metadata for %s", file.name)

        build_index_for_directory(out_dir)
        logger.info("Built embeddings index for %s", file.name)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Local document processing and semantic image search pipeline")
    sub = parser.add_subparsers(dest="command", required=True)

    extract_parser = sub.add_parser("extract", help="Extract text, tables, and images from documents")
    extract_parser.add_argument("--file", type=str, help="Path to a single file")
    extract_parser.add_argument("--folder", type=str, help="Path to folder with documents")
    extract_parser.add_argument("--output", type=str, default="output", help="Output directory")
    extract_parser.set_defaults(func=cmd_extract)

    metadata_parser = sub.add_parser("generate-metadata", help="Generate image metadata for extracted images")
    metadata_parser.add_argument("--input", type=str, required=True, help="Document output directory")
    metadata_parser.set_defaults(func=cmd_generate_metadata)

    index_parser = sub.add_parser("build-index", help="Build CLIP embeddings and image search index")
    index_parser.add_argument("--input", type=str, required=True, help="Document output directory")
    index_parser.set_defaults(func=cmd_build_index)

    search_parser = sub.add_parser("search", help="Search indexed images semantically")
    search_parser.add_argument("--index", type=str, required=True, help="Path to index directory")
    search_parser.add_argument("--query", type=str, required=True, help="Search query text")
    search_parser.add_argument("--top-k", type=int, default=5, help="Number of top results")
    search_parser.add_argument(
        "--search-mode",
        type=str,
        default="hybrid",
        choices=["image_only", "text_only", "hybrid"],
        help="Which embedding space to search",
    )
    search_parser.set_defaults(func=cmd_search)

    pipeline_parser = sub.add_parser("pipeline", help="Run extract + metadata + index end-to-end")
    pipeline_parser.add_argument("--file", type=str, help="Path to a single file")
    pipeline_parser.add_argument("--folder", type=str, help="Path to folder with documents")
    pipeline_parser.add_argument("--output", type=str, default="output", help="Output directory")
    pipeline_parser.set_defaults(func=cmd_pipeline)

    return parser


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
