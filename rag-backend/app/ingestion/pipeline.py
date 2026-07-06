import os
from langchain_text_splitters import RecursiveCharacterTextSplitter

from app.config import DOCS_DIR, SARVAM_OUTPUT_DIR, IMAGES_DIR, CHUNK_SIZE, CHUNK_OVERLAP
from app.core.embedders import get_embedder, faiss_dir_for
from app.core.faiss_store import FaissStore, invalidate_store
from app.ingestion.pdf_split import split_pdfs
from app.ingestion.sarvam_parser import (
    process_pdf_chunks, unzip_outputs, parse_markdown, decode_and_save_images,
)


def chunk_text_blocks(text_blocks: list, chunk_size: int = CHUNK_SIZE, chunk_overlap: int = CHUNK_OVERLAP) -> list:
    splitter = RecursiveCharacterTextSplitter(
        separators=["\n\n", "\n", ". ", " ", ""],
        chunk_size=chunk_size, chunk_overlap=chunk_overlap, length_function=len)
    out = []
    for block in text_blocks:
        for chunk in splitter.split_text(block["chunk"]):
            out.append({"chunk": chunk, "source": block["source"]})
    return out


def parse_zips(zip_names: list | None = None) -> tuple:
    """zip_names=None -> parse every zip. Otherwise only those specific zips."""
    files = unzip_outputs(SARVAM_OUTPUT_DIR, zip_names=zip_names)
    all_text, all_images, all_tables = [], [], []
    for _, md_path in files:
        c = parse_markdown(md_path)
        all_text.extend(c["text"])
        all_images.extend(c["images"])
        all_tables.extend(c["tables"])
        if c["images"]:
            decode_and_save_images(c["images"], IMAGES_DIR)
    return all_text, all_images, all_tables


def _load_or_create_store(embedder_kind: str, dim: int) -> FaissStore:
    store = FaissStore(dim=dim, index_dir=faiss_dir_for(embedder_kind))
    if os.path.exists(store.index_path):
        store.load()
    return store


def add_to_index(embedder_kind: str, all_text: list, all_images: list, all_tables: list,
                  chunk_size: int = CHUNK_SIZE, chunk_overlap: int = CHUNK_OVERLAP) -> dict:
    """Embeds new content and appends it to the existing FAISS index (incremental)."""
    embedder = get_embedder(embedder_kind)
    text_chunks = chunk_text_blocks(all_text, chunk_size=chunk_size, chunk_overlap=chunk_overlap)
    store = _load_or_create_store(embedder_kind, embedder.dim)
    start = store.size

    if text_chunks:
        embs = embedder.embed_documents([c["chunk"] for c in text_chunks])
        metas = [{"id": f"text_{start + i}", "text": c["chunk"], "content_type": "text", "source": c["source"]}
                 for i, c in enumerate(text_chunks)]
        store.add(embs, metas)

    if all_images:
        embs = embedder.embed_documents([img["embedding_text"] or "image" for img in all_images])
        metas = [{
            "id": f"image_{start + i}", "text": img["embedding_text"], "content_type": "image",
            "image_path": img.get("image_path", ""), "alt_text": img["alt_text"],
            "caption": img.get("caption", ""), "sarvam_description": img.get("sarvam_description", ""),
            "source": img["source"]
        } for i, img in enumerate(all_images)]
        store.add(embs, metas)

    if all_tables:
        summaries = []
        for t in all_tables:
            if t.get("type") == "markdown":
                lines = t.get("markdown", "").split("\n")
                rows = max(0, len([l for l in lines if l.strip().startswith("|")]) - 2)
                summaries.append(f"Markdown table with {rows} rows. Headers: {lines[0][:150] if lines else 'N/A'}")
            else:
                summaries.append(f"HTML table: {t.get('caption', 'HTML Table')}")
        embs = embedder.embed_documents(summaries)
        metas = [{
            "id": f"table_{start + i}", "text": summaries[i], "content_type": "table",
            "table_type": t.get("type", "markdown"),
            "raw_content": t.get("markdown", "") if t.get("type") == "markdown" else t.get("html", ""),
            "source": t["source"]
        } for i, t in enumerate(all_tables)]
        store.add(embs, metas)

    store.save()
    invalidate_store(embedder_kind)

    return {
        "status": "complete",
        "embedder": embedder_kind,
        "dim": embedder.dim,
        "text_chunks_added": len(text_chunks),
        "images_added": len(all_images),
        "tables_added": len(all_tables),
        "total_vectors": store.size,
    }


def run_index_rebuild(embedder_kind: str = "hf", chunk_size: int = CHUNK_SIZE, chunk_overlap: int = CHUNK_OVERLAP) -> dict:
    """Full rebuild from ALL Sarvam zips ever produced. Use for re-index / embedder switch."""
    all_text, all_images, all_tables = parse_zips(zip_names=None)
    if not (all_text or all_images or all_tables):
        return {"status": "error", "detail": f"No Sarvam outputs in {SARVAM_OUTPUT_DIR}. Upload PDFs and run the full pipeline first."}

    embedder = get_embedder(embedder_kind)
    index_dir = faiss_dir_for(embedder_kind)
    index_path = os.path.join(index_dir, "index.faiss")
    meta_path = os.path.join(index_dir, "metadata.json")
    for p in (index_path, meta_path):
        if os.path.exists(p):
            os.remove(p)
    invalidate_store(embedder_kind)

    return add_to_index(embedder_kind, all_text, all_images, all_tables, chunk_size=chunk_size, chunk_overlap=chunk_overlap)


def run_full_pipeline(embedder_kind: str = "hf", pdf_paths: list | None = None,
                      max_pages: int = 10, language: str = "en-IN",
                      chunk_size: int = CHUNK_SIZE, chunk_overlap: int = CHUNK_OVERLAP) -> dict:
    """PDFs → split → Sarvam OCR → parse ONLY the new zips → append to existing FAISS index."""
    if pdf_paths is None:
        pdf_paths = [os.path.join(DOCS_DIR, f)
                     for f in os.listdir(DOCS_DIR) if f.lower().endswith(".pdf")]
    if not pdf_paths:
        return {"status": "error", "detail": f"No PDFs found in {DOCS_DIR}"}

    print(f"[PIPELINE] {len(pdf_paths)} PDFs → Sarvam → FAISS ({embedder_kind}) [incremental]")
    pdf_chunks = split_pdfs(pdf_paths, max_pages=max_pages)
    zip_paths = process_pdf_chunks(pdf_chunks, language=language)
    if not zip_paths:
        return {"status": "error", "detail": "Sarvam processing produced no outputs."}

    new_zip_names = [os.path.basename(z) for z in zip_paths]
    all_text, all_images, all_tables = parse_zips(zip_names=new_zip_names)

    result = add_to_index(embedder_kind, all_text, all_images, all_tables, chunk_size=chunk_size, chunk_overlap=chunk_overlap)
    result["pdfs_processed"] = len(pdf_paths)
    result["sarvam_jobs_completed"] = len(zip_paths)
    return result


if __name__ == "__main__":
    import sys
    mode = sys.argv[1] if len(sys.argv) > 1 else "index"      # "full" | "index"
    kind = sys.argv[2] if len(sys.argv) > 2 else "hf"
    print(run_full_pipeline(kind) if mode == "full" else run_index_rebuild(kind))