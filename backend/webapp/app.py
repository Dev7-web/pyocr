from __future__ import annotations

from contextlib import asynccontextmanager
from pathlib import Path
from typing import Dict, List, Literal

from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.responses import FileResponse, HTMLResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field

from ..search.semantic_search import search_images
from ..utils.helpers import SUPPORTED_EXTENSIONS, ensure_dir
from .job_manager import PipelineJobManager


class SearchRequest(BaseModel):
    query: str = Field(min_length=1)
    top_k: int = Field(default=5, ge=1, le=50)
    search_mode: Literal["image_only", "text_only", "hybrid"] = "hybrid"


def create_app(workspace_dir: Path | None = None) -> FastAPI:
    # backend_root points at the top of the backend package (../backend)
    backend_root = workspace_dir or Path(__file__).resolve().parent.parent
    output_dir = ensure_dir(backend_root / "output")
    upload_dir = ensure_dir(backend_root / "data" / "uploads")

    # Frontend lives as a sibling of backend/
    repo_root = backend_root.parent
    frontend_dir = repo_root / "frontend"

    manager = PipelineJobManager(upload_dir=upload_dir, output_dir=output_dir)

    @asynccontextmanager
    async def lifespan(_: FastAPI):
        yield
        manager.shutdown()

    app = FastAPI(title="Doc Pipeline Studio", version="1.0.0", lifespan=lifespan)
    app.state.manager = manager
    app.state.output_dir = output_dir
    app.state.frontend_dir = frontend_dir

    if output_dir.exists():
        app.mount("/output", StaticFiles(directory=str(output_dir)), name="output")
    if frontend_dir.exists():
        app.mount("/assets", StaticFiles(directory=str(frontend_dir)), name="assets")

    def _public_image_url(image_path: str) -> str:
        try:
            image = Path(image_path).resolve()
            rel = image.relative_to(output_dir.resolve())
            return "/output/" + "/".join(rel.parts)
        except Exception:
            return ""

    @app.get("/api/health")
    def health() -> Dict[str, str]:
        return {"status": "ok"}

    @app.get("/api/jobs")
    def list_jobs() -> List[Dict]:
        return manager.list_jobs()

    @app.get("/api/jobs/{job_id}")
    def get_job(job_id: str) -> Dict:
        try:
            return manager.get_job(job_id)
        except KeyError:
            raise HTTPException(status_code=404, detail="Job not found")

    @app.delete("/api/jobs/{job_id}")
    def delete_job(job_id: str) -> Dict[str, str]:
        try:
            manager.remove_job_artifacts(job_id)
        except KeyError:
            raise HTTPException(status_code=404, detail="Job not found")
        return {"status": "deleted"}

    @app.get("/api/jobs/{job_id}/result")
    def get_result(job_id: str) -> Dict:
        try:
            payload = manager.get_extraction_result(job_id)
        except KeyError:
            raise HTTPException(status_code=404, detail="Job not found")
        except FileNotFoundError as exc:
            raise HTTPException(status_code=404, detail=str(exc))
        return payload

    @app.get("/api/jobs/{job_id}/gallery")
    def get_gallery(job_id: str) -> Dict[str, List[Dict]]:
        try:
            index_data = manager.get_image_index(job_id)
        except KeyError:
            raise HTTPException(status_code=404, detail="Job not found")
        except FileNotFoundError as exc:
            raise HTTPException(status_code=404, detail=str(exc))

        images = []
        for key, row in index_data.items():
            images.append(
                {
                    "image_id": key,
                    "image_path": row.get("image_path", ""),
                    "public_image_url": _public_image_url(row.get("image_path", "")),
                    "source_page": row.get("source_page"),
                    "source_file": row.get("source_file"),
                    "metadata": row.get("metadata", {}),
                    "extraction_method": row.get("extraction_method", ""),
                }
            )
        return {"images": images}

    @app.post("/api/jobs/{job_id}/search")
    def search_job(job_id: str, body: SearchRequest) -> Dict[str, List[Dict]]:
        try:
            job = manager.get_job(job_id)
        except KeyError:
            raise HTTPException(status_code=404, detail="Job not found")

        if job["status"] != "completed":
            raise HTTPException(status_code=409, detail="Job is not completed yet.")

        try:
            results = search_images(
                query=body.query,
                index_path=job["output_dir"],
                top_k=body.top_k,
                search_mode=body.search_mode,
            )
        except Exception as exc:
            raise HTTPException(status_code=500, detail=f"Search failed: {exc}")

        response = []
        for row in results:
            response.append(
                {
                    "image_path": row.image_path,
                    "public_image_url": _public_image_url(row.image_path),
                    "similarity_score": row.similarity_score,
                    "metadata": row.metadata.model_dump(),
                    "source_page": row.source_page,
                    "source_file": row.source_file,
                }
            )
        return {"results": response}

    @app.post("/api/upload")
    async def upload(file: UploadFile = File(...)) -> Dict:
        if not file.filename:
            raise HTTPException(status_code=400, detail="Missing filename")
        suffix = Path(file.filename).suffix.lower()
        if suffix not in SUPPORTED_EXTENSIONS:
            supported = ", ".join(sorted(SUPPORTED_EXTENSIONS.keys()))
            raise HTTPException(status_code=400, detail=f"Unsupported file type. Supported: {supported}")

        content = await file.read()
        if not content:
            raise HTTPException(status_code=400, detail="Uploaded file is empty")

        return manager.create_job(original_filename=file.filename, file_bytes=content)

    @app.get("/")
    def serve_index():
        index_file = frontend_dir / "index.html"
        if index_file.exists():
            return FileResponse(index_file)
        return HTMLResponse(
            "<h1>Frontend not found</h1><p>Create frontend/index.html to render the dashboard.</p>",
            status_code=200,
        )

    return app


app = create_app()

