from __future__ import annotations

import time
from pathlib import Path

import pytest

fastapi = pytest.importorskip("fastapi")
pytest.importorskip("starlette")

from fastapi.testclient import TestClient

from webapp.app import create_app


def test_health_endpoint(tmp_path: Path):
    app = create_app(workspace_dir=tmp_path)
    client = TestClient(app)
    response = client.get("/api/health")
    assert response.status_code == 200
    assert response.json()["status"] == "ok"


def test_upload_and_job_flow_csv(tmp_path: Path):
    app = create_app(workspace_dir=tmp_path)
    client = TestClient(app)

    payload = b"col1,col2\n1,2\n3,4\n"
    response = client.post(
        "/api/upload",
        files={"file": ("sample.csv", payload, "text/csv")},
    )
    assert response.status_code == 200
    job = response.json()
    assert "job_id" in job
    job_id = job["job_id"]

    deadline = time.time() + 20
    while time.time() < deadline:
        poll = client.get(f"/api/jobs/{job_id}")
        assert poll.status_code == 200
        current = poll.json()
        if current["status"] in {"completed", "failed"}:
            break
        time.sleep(0.25)
    assert current["status"] == "completed"

    result = client.get(f"/api/jobs/{job_id}/result")
    assert result.status_code == 200
    data = result.json()
    assert data["file_type"] == "csv"

    search = client.post(
        f"/api/jobs/{job_id}/search",
        json={"query": "table", "top_k": 3, "search_mode": "hybrid"},
    )
    assert search.status_code == 200
    assert "results" in search.json()
