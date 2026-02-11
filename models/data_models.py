from __future__ import annotations

from datetime import datetime
from typing import Dict, List, Optional

from pydantic import BaseModel, Field


class TextBlock(BaseModel):
    content: str
    page_or_slide: int
    heading_level: Optional[int] = None
    source_section: Optional[str] = None
    block_type: str = "paragraph"


class TableBlock(BaseModel):
    data: List[List[str]]
    headers: Optional[List[str]] = None
    page_or_slide: int
    sheet_name: Optional[str] = None
    num_rows: int
    num_cols: int
    has_merged_cells: bool = False
    source_formulas: Optional[Dict[str, str]] = None


class ImageMetadata(BaseModel):
    description: str = ""
    detected_text: str = ""
    image_type: str = "other"
    objects_detected: List[str] = Field(default_factory=list)
    source_context: str = ""
    tags: List[str] = Field(default_factory=list)


class ImageBlock(BaseModel):
    image_path: str
    page_or_slide: int
    bounding_box: Optional[Dict[str, float]] = None
    dimensions: Dict[str, int]
    extraction_method: str
    metadata: Optional[ImageMetadata] = None


class SheetDependency(BaseModel):
    sheet_name: str
    references: List[str] = Field(default_factory=list)
    formulas_with_cross_refs: Dict[str, str] = Field(default_factory=dict)


class ExtractionResult(BaseModel):
    file_name: str
    file_type: str
    extraction_timestamp: datetime
    num_pages_or_slides: int
    texts: List[TextBlock] = Field(default_factory=list)
    tables: List[TableBlock] = Field(default_factory=list)
    images: List[ImageBlock] = Field(default_factory=list)
    sheet_dependencies: Optional[List[SheetDependency]] = None
    warnings: List[str] = Field(default_factory=list)


class SearchResult(BaseModel):
    image_path: str
    similarity_score: float
    metadata: ImageMetadata
    source_page: int
    source_file: str
