const state = {
  jobId: null,
  pollTimer: null,
};

const el = {
  healthPill: document.getElementById("healthPill"),
  uploadForm: document.getElementById("uploadForm"),
  fileInput: document.getElementById("fileInput"),
  fileLabel: document.getElementById("fileLabel"),
  uploadBtn: document.getElementById("uploadBtn"),
  jobIdBadge: document.getElementById("jobIdBadge"),
  progressBar: document.getElementById("progressBar"),
  jobStatus: document.getElementById("jobStatus"),
  jobStage: document.getElementById("jobStage"),
  jobProgress: document.getElementById("jobProgress"),
  jobMessage: document.getElementById("jobMessage"),
  jobError: document.getElementById("jobError"),
  metricTexts: document.getElementById("metricTexts"),
  metricTables: document.getElementById("metricTables"),
  metricImages: document.getElementById("metricImages"),
  metricWarnings: document.getElementById("metricWarnings"),
  summaryNote: document.getElementById("summaryNote"),
  searchForm: document.getElementById("searchForm"),
  queryInput: document.getElementById("queryInput"),
  searchMode: document.getElementById("searchMode"),
  topKInput: document.getElementById("topKInput"),
  resultGrid: document.getElementById("resultGrid"),
  resultCount: document.getElementById("resultCount"),
};

async function api(path, options = {}) {
  const response = await fetch(path, options);
  const contentType = response.headers.get("content-type") || "";
  const payload = contentType.includes("application/json") ? await response.json() : await response.text();
  if (!response.ok) {
    const detail = payload && payload.detail ? payload.detail : response.statusText;
    throw new Error(detail);
  }
  return payload;
}

function setSummary(summary = {}) {
  el.metricTexts.textContent = summary.texts ?? 0;
  el.metricTables.textContent = summary.tables ?? 0;
  el.metricImages.textContent = summary.images ?? 0;
  el.metricWarnings.textContent = summary.warnings ?? 0;
}

function setJobStatus(job) {
  el.jobIdBadge.textContent = job?.job_id ? job.job_id.slice(0, 12) : "No job yet";
  el.jobStatus.textContent = job?.status || "idle";
  el.jobStage.textContent = job?.stage || "-";
  el.jobProgress.textContent = `${job?.progress ?? 0}%`;
  el.jobMessage.textContent = job?.message || "Upload a document to begin.";
  el.jobError.textContent = job?.error || "";
  el.progressBar.style.width = `${job?.progress ?? 0}%`;
  setSummary(job?.summary || {});
}

function resetResults() {
  el.resultGrid.innerHTML = "";
  el.resultCount.textContent = "0 items";
}

function renderResults(items) {
  resetResults();
  if (!items || items.length === 0) {
    el.resultGrid.innerHTML = `<article class="result-card"><div class="meta">No image results available.</div></article>`;
    return;
  }

  items.forEach((item, i) => {
    const score = typeof item.similarity_score === "number" ? item.similarity_score.toFixed(4) : null;
    const meta = item.metadata || {};
    const tags = (meta.tags || []).slice(0, 5).join(", ");
    const desc = meta.description || "No description";
    const card = document.createElement("article");
    card.className = "result-card";
    card.style.animationDelay = `${i * 0.05}s`;

    const img = item.public_image_url
      ? `<img src="${item.public_image_url}" alt="search result image" loading="lazy" />`
      : `<div class="meta">Image preview unavailable</div>`;

    card.innerHTML = `
      ${img}
      <div class="meta">
        <div><strong>Page/Slide:</strong> ${item.source_page ?? "-"}</div>
        ${score ? `<div><strong>Score:</strong> ${score}</div>` : ""}
        <div><strong>Type:</strong> ${meta.image_type || "other"}</div>
        <div><strong>Tags:</strong> ${tags || "-"}</div>
        <div>${desc.slice(0, 180)}</div>
      </div>
    `;
    el.resultGrid.appendChild(card);
  });

  el.resultCount.textContent = `${items.length} items`;
}

async function loadGallery(jobId) {
  try {
    const payload = await api(`/api/jobs/${jobId}/gallery`);
    renderResults(payload.images || []);
  } catch (err) {
    resetResults();
  }
}

function stopPolling() {
  if (state.pollTimer) {
    clearInterval(state.pollTimer);
    state.pollTimer = null;
  }
}

async function pollJob(jobId) {
  try {
    const job = await api(`/api/jobs/${jobId}`);
    setJobStatus(job);
    if (job.status === "completed") {
      stopPolling();
      el.summaryNote.textContent = "Pipeline completed. You can now run semantic search.";
      await loadGallery(jobId);
    } else if (job.status === "failed") {
      stopPolling();
      el.summaryNote.textContent = "Pipeline failed. Check error details above.";
    }
  } catch (err) {
    stopPolling();
    el.jobError.textContent = err.message;
  }
}

async function submitUpload(event) {
  event.preventDefault();
  const file = el.fileInput.files?.[0];
  if (!file) {
    el.jobError.textContent = "Choose a file first.";
    return;
  }

  el.jobError.textContent = "";
  el.summaryNote.textContent = "Processing...";
  setSummary({});
  resetResults();
  el.uploadBtn.disabled = true;
  el.uploadBtn.textContent = "Uploading...";

  try {
    const data = new FormData();
    data.append("file", file);
    const job = await api("/api/upload", { method: "POST", body: data });
    state.jobId = job.job_id;
    setJobStatus(job);
    stopPolling();
    state.pollTimer = setInterval(() => pollJob(state.jobId), 1400);
    await pollJob(state.jobId);
  } catch (err) {
    el.jobError.textContent = err.message;
  } finally {
    el.uploadBtn.disabled = false;
    el.uploadBtn.textContent = "Start Pipeline";
  }
}

async function submitSearch(event) {
  event.preventDefault();
  if (!state.jobId) {
    el.jobError.textContent = "Run a pipeline job first.";
    return;
  }

  const query = el.queryInput.value.trim();
  if (!query) {
    el.jobError.textContent = "Enter a search query.";
    return;
  }

  try {
    const payload = await api(`/api/jobs/${state.jobId}/search`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({
        query,
        top_k: Number(el.topKInput.value || 5),
        search_mode: el.searchMode.value,
      }),
    });
    renderResults(payload.results || []);
  } catch (err) {
    el.jobError.textContent = err.message;
  }
}

async function checkHealth() {
  try {
    const payload = await api("/api/health");
    if (payload.status === "ok") {
      el.healthPill.textContent = "Backend: online";
    }
  } catch (err) {
    el.healthPill.textContent = "Backend: offline";
    el.jobError.textContent = `Health check failed: ${err.message}`;
  }
}

function bindEvents() {
  el.uploadForm.addEventListener("submit", submitUpload);
  el.searchForm.addEventListener("submit", submitSearch);
  el.fileInput.addEventListener("change", () => {
    const selected = el.fileInput.files?.[0];
    el.fileLabel.textContent = selected ? selected.name : "Choose PDF / PPTX / DOCX / CSV / XLSX";
  });
}

checkHealth();
bindEvents();
