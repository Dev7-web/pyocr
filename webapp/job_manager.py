from __future__ import annotations

import json
import shutil
import threading
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, List, Optional
from uuid import uuid4

from extractors import process_document
from image_processing.embedding_generator import build_index_for_directory
from image_processing.metadata_generator import generate_metadata_for_directory
from utils.helpers import ensure_dir, make_document_output_dir, save_extraction_outputs, setup_logger


def _utc_now() -> datetime:
    return datetime.now(timezone.utc)


@dataclass
class JobRecord:
    job_id: str
    original_filename: str
    stored_path: str
    status: str = "queued"
    progress: int = 0
    stage: str = "queued"
    message: str = "Waiting to start"
    error: Optional[str] = None
    output_dir: Optional[str] = None
    summary: Dict[str, int] = field(default_factory=dict)
    created_at: datetime = field(default_factory=_utc_now)
    updated_at: datetime = field(default_factory=_utc_now)


class PipelineJobManager:
    def __init__(self, upload_dir: Path, output_dir: Path, max_workers: int = 2) -> None:
        self.logger = setup_logger("webapp")
        self.upload_dir = ensure_dir(upload_dir)
        self.output_dir = ensure_dir(output_dir)
        self.executor = ThreadPoolExecutor(max_workers=max(1, max_workers))
        self.lock = threading.RLock()
        self.jobs: Dict[str, JobRecord] = {}

    def _serialize(self, record: JobRecord) -> Dict:
        return {
            "job_id": record.job_id,
            "original_filename": record.original_filename,
            "stored_path": record.stored_path,
            "status": record.status,
            "progress": record.progress,
            "stage": record.stage,
            "message": record.message,
            "error": record.error,
            "output_dir": record.output_dir,
            "summary": record.summary,
            "created_at": record.created_at.isoformat(),
            "updated_at": record.updated_at.isoformat(),
        }

    def _patch(self, job_id: str, **updates) -> None:
        with self.lock:
            record = self.jobs[job_id]
            for key, value in updates.items():
                setattr(record, key, value)
            record.updated_at = _utc_now()

    def _get(self, job_id: str) -> JobRecord:
        with self.lock:
            if job_id not in self.jobs:
                raise KeyError(job_id)
            return self.jobs[job_id]

    def create_job(self, original_filename: str, file_bytes: bytes) -> Dict:
        job_id = uuid4().hex
        job_upload_dir = ensure_dir(self.upload_dir / job_id)
        safe_name = Path(original_filename).name
        stored_path = job_upload_dir / safe_name
        stored_path.write_bytes(file_bytes)

        record = JobRecord(
            job_id=job_id,
            original_filename=safe_name,
            stored_path=str(stored_path),
            status="queued",
            progress=0,
            stage="queued",
            message="Queued for processing",
        )

        with self.lock:
            self.jobs[job_id] = record
        self.executor.submit(self._run_pipeline, job_id)
        return self._serialize(record)

    def _run_pipeline(self, job_id: str) -> None:
        record = self._get(job_id)
        source_path = Path(record.stored_path)
        job_output_root = ensure_dir(self.output_dir / job_id)
        try:
            self._patch(
                job_id,
                status="running",
                stage="extract",
                progress=10,
                message="Extracting text, tables, and images",
            )
            result = process_document(str(source_path), output_root=str(job_output_root))
            out_dir = make_document_output_dir(source_path, job_output_root)
            save_extraction_outputs(result, out_dir)

            self._patch(
                job_id,
                stage="metadata",
                progress=55,
                output_dir=str(out_dir),
                summary={
                    "texts": len(result.texts),
                    "tables": len(result.tables),
                    "images": len(result.images),
                    "warnings": len(result.warnings),
                },
                message="Generating image metadata",
            )
            generate_metadata_for_directory(out_dir)

            self._patch(
                job_id,
                stage="index",
                progress=80,
                message="Building embedding index",
            )
            build_index_for_directory(out_dir)

            self._patch(
                job_id,
                status="completed",
                stage="completed",
                progress=100,
                message="Pipeline finished",
                output_dir=str(out_dir),
            )
        except Exception as exc:
            self.logger.exception("Job %s failed", job_id)
            self._patch(
                job_id,
                status="failed",
                stage="failed",
                progress=100,
                error=str(exc),
                message="Pipeline failed",
            )

    def get_job(self, job_id: str) -> Dict:
        return self._serialize(self._get(job_id))

    def list_jobs(self) -> List[Dict]:
        with self.lock:
            records = sorted(self.jobs.values(), key=lambda r: r.created_at, reverse=True)
        return [self._serialize(rec) for rec in records]

    def get_extraction_result(self, job_id: str) -> Dict:
        record = self._get(job_id)
        if not record.output_dir:
            raise FileNotFoundError("Output directory not available yet.")
        result_path = Path(record.output_dir) / "extraction_result.json"
        if not result_path.exists():
            raise FileNotFoundError("extraction_result.json not found.")
        with open(result_path, "r", encoding="utf-8") as f:
            return json.load(f)

    def get_image_index(self, job_id: str) -> Dict:
        record = self._get(job_id)
        if not record.output_dir:
            raise FileNotFoundError("Output directory not available yet.")
        index_path = Path(record.output_dir) / "image_index.json"
        if not index_path.exists():
            return {}
        with open(index_path, "r", encoding="utf-8") as f:
            return json.load(f)

    def remove_job_artifacts(self, job_id: str) -> None:
        record = self._get(job_id)
        if record.output_dir:
            out_dir = Path(record.output_dir)
            if out_dir.exists():
                shutil.rmtree(out_dir, ignore_errors=True)
        upload_dir = self.upload_dir / job_id
        if upload_dir.exists():
            shutil.rmtree(upload_dir, ignore_errors=True)
        with self.lock:
            self.jobs.pop(job_id, None)

    def shutdown(self) -> None:
        self.executor.shutdown(wait=False, cancel_futures=True)
