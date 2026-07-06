import os
import uuid
from datetime import datetime
from typing import List, Optional

from fastapi import APIRouter, BackgroundTasks, File, HTTPException, UploadFile
from pydantic import BaseModel, Field

from app.config import DOCS_DIR, AVAILABLE_EMBEDDERS, DEFAULT_EMBEDDER
from app.ingestion.pipeline import run_full_pipeline, run_index_rebuild
from app.core.faiss_store import get_store, invalidate_store

router = APIRouter(tags=["ingestion"])

JOBS: dict = {}


class IngestRequest(BaseModel):
    embedder: str = Field(DEFAULT_EMBEDDER, description="hf | openai")
    mode: str = Field("full", description="'full' = PDFs→Sarvam→FAISS, 'index' = re-index existing Sarvam outputs")
    max_pages: int = Field(10, ge=1, le=50)
    language: str = "en-IN"
    chunk_size: int = Field(1000, ge=100, le=8000)
    chunk_overlap: int = Field(200, ge=0, le=2000)
    filenames: Optional[List[str]] = Field(
        None, description="If set, only these PDFs (in docs/) are processed. Omit to process all PDFs in docs/."
    )


def _run_job(job_id: str, req: IngestRequest):
    JOBS[job_id]["status"] = "running"
    try:
        if req.mode == "index":
            result = run_index_rebuild(req.embedder, chunk_size=req.chunk_size, chunk_overlap=req.chunk_overlap)
        else:
            pdf_paths = None
            if req.filenames:
                pdf_paths = [os.path.join(DOCS_DIR, os.path.basename(f)) for f in req.filenames]
            result = run_full_pipeline(
                req.embedder, pdf_paths=pdf_paths, max_pages=req.max_pages, language=req.language,
                chunk_size=req.chunk_size, chunk_overlap=req.chunk_overlap,
            )
        JOBS[job_id]["status"] = "failed" if result.get("status") == "error" else "completed"
        JOBS[job_id]["result"] = result
    except Exception as e:
        JOBS[job_id]["status"] = "failed"
        JOBS[job_id]["result"] = {"detail": str(e)}
    JOBS[job_id]["finished_at"] = datetime.now().isoformat()
 

@router.post("/upload")
async def upload_pdfs(files: List[UploadFile] = File(...)):
    saved = []
    for f in files:
        if not f.filename or not f.filename.lower().endswith(".pdf"):
            raise HTTPException(400, f"'{f.filename}' is not a PDF")
        safe_name = os.path.basename(f.filename)
        dest = os.path.join(DOCS_DIR, safe_name)
        with open(dest, "wb") as out:
            out.write(await f.read())
        saved.append(safe_name)
    return {"status": "uploaded", "files": saved, "count": len(saved)}


@router.get("/documents")
def list_documents():
    pdfs = [f for f in os.listdir(DOCS_DIR) if f.lower().endswith(".pdf")]
    return {"documents": sorted(pdfs), "count": len(pdfs)}


@router.delete("/documents/{filename}")
def delete_document(filename: str):
    safe_name = os.path.basename(filename)
    path = os.path.join(DOCS_DIR, safe_name)
    if not os.path.exists(path):
        raise HTTPException(404, "Document not found")
    os.remove(path)
 
    pdf_stem = os.path.splitext(safe_name)[0]
    prefix = f"{pdf_stem}_part_"
    removed_counts = {}
    for embedder_kind in AVAILABLE_EMBEDDERS:
        try:
            store = get_store(embedder_kind)
        except FileNotFoundError:
            continue
        removed = store.remove_by_source_prefix(prefix)
        if removed:
            store.save()
            invalidate_store(embedder_kind)
            removed_counts[embedder_kind] = removed
 
    return {"status": "deleted", "file": filename, "vectors_removed": removed_counts}


@router.post("/ingest")
def start_ingestion(req: IngestRequest, background_tasks: BackgroundTasks):
    if req.embedder not in AVAILABLE_EMBEDDERS:
        raise HTTPException(400, f"Unknown embedder '{req.embedder}'")
    if req.mode not in ("full", "index"):
        raise HTTPException(400, "mode must be 'full' or 'index'")
    if req.mode == "full":
        if req.filenames:
            missing = [f for f in req.filenames if not os.path.exists(os.path.join(DOCS_DIR, os.path.basename(f)))]
            if missing:
                raise HTTPException(400, f"Not found in docs/: {missing}")
        elif not any(f.lower().endswith(".pdf") for f in os.listdir(DOCS_DIR)):
            raise HTTPException(400, f"No PDFs in {DOCS_DIR}. Use POST /upload first.")

    job_id = uuid.uuid4().hex[:12]
    JOBS[job_id] = {
        "job_id": job_id,
        "status": "queued",
        "mode": req.mode,
        "embedder": req.embedder,
        "filenames": req.filenames,
        "started_at": datetime.now().isoformat(),
        "result": None,
    }
    background_tasks.add_task(_run_job, job_id, req)
    return {"job_id": job_id, "status": "queued", "poll": f"/ingest/status/{job_id}"}


@router.get("/ingest/status/{job_id}")
def ingest_status(job_id: str):
    if job_id not in JOBS:
        raise HTTPException(404, "Job not found")
    return JOBS[job_id]


@router.get("/ingest/jobs")
def list_jobs():
    return {"jobs": list(JOBS.values())}