# Doc Pipeline Studio

Local-first document processing and semantic image retrieval with both:

- a **backend API** (job orchestration + search)
- a **frontend dashboard** (upload, progress tracking, results gallery, semantic search)

You can run the complete workflow from the browser without manually chaining CLI commands.

## What It Does

Given a document (`PDF`, `PPTX`, `DOCX`, `CSV`, `XLSX/XLS`), the pipeline:

1. Extracts text blocks, tables, and images.
2. Detects vector-heavy PDF drawings and renders them as diagram images.
3. Generates image metadata using a local vision-model strategy + OCR fallback.
4. Builds local CLIP embeddings (`image`, `text`, `hybrid`) and stores them in `.npz`.
5. Supports semantic image search using cosine similarity.

No database, Redis, background worker service, or external API is required.

## Project Layout

```text
.
├── app.py                     # One-command web app launcher
├── main.py                    # CLI entrypoint (still available)
├── webapp/
│   ├── app.py                 # FastAPI backend + static serving
│   └── job_manager.py         # Async job orchestration
├── frontend/
│   ├── index.html             # Dashboard UI
│   ├── styles.css
│   └── app.js
├── extractors/
├── image_processing/
├── search/
├── models/
├── utils/
└── tests/
```

## Quick Start (Web App)

```bash
python -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt
python app.py
```

Open:

- `http://127.0.0.1:8000`

From the UI:

1. Upload a document.
2. Wait for pipeline completion status.
3. Run semantic queries in the search section.

## CLI Mode (Optional)

CLI remains available if needed:

```bash
python main.py pipeline --file report.pdf --output ./output
python main.py search --index ./output/report_pdf --query "circuit diagram"
```

## API Endpoints

- `GET /api/health`
- `POST /api/upload` (multipart file upload)
- `GET /api/jobs`
- `GET /api/jobs/{job_id}`
- `GET /api/jobs/{job_id}/result`
- `GET /api/jobs/{job_id}/gallery`
- `POST /api/jobs/{job_id}/search`
- `DELETE /api/jobs/{job_id}`

## Output

Per job, artifacts are saved under:

```text
output/<job_id>/<document_name_ext>/
├── extraction_result.json
├── texts/full_text.txt
├── tables/*.csv
├── images/*
├── image_index.json
└── embeddings.npz
```

## Optional System Dependencies

- **Tesseract OCR** binary for better `detected_text` in images.
- **LibreOffice** for PPTX chart/SmartArt fallback rendering.

## Testing

```bash
pytest -q
```
