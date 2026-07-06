import os
from pypdf import PdfReader, PdfWriter

from app.config import PDF_CHUNKS_DIR


def split_pdf(pdf_path: str, max_pages: int = 10) -> list:
    chunks = []
    reader = PdfReader(pdf_path)
    total_pages = len(reader.pages)
    base_name = os.path.splitext(os.path.basename(pdf_path))[0]
    print(f"Processing PDF: {pdf_path} ({total_pages} pages)")

    if total_pages <= max_pages:
        return [{
            "source_pdf": pdf_path,
            "chunk_path": pdf_path,
            "chunk_num": 1,
            "pdf_name": base_name,
        }]

    for i in range(0, total_pages, max_pages):
        writer = PdfWriter()
        end_page = min(i + max_pages, total_pages)
        for page_no in range(i, end_page):
            writer.add_page(reader.pages[page_no])

        chunk_file = os.path.join(PDF_CHUNKS_DIR, f"{base_name}_part_{(i // max_pages) + 1}.pdf")
        with open(chunk_file, "wb") as f:
            writer.write(f)
        print(f"  Created chunk: {chunk_file} (pages {i+1}-{end_page})")

        chunks.append({
            "source_pdf": pdf_path,
            "chunk_path": chunk_file,
            "chunk_num": i // max_pages + 1,
            "pdf_name": base_name,
        })
    return chunks


def split_pdfs(pdf_paths: list, max_pages: int = 10) -> list:
    all_chunks = []
    for pdf_path in pdf_paths:
        try:
            all_chunks.extend(split_pdf(pdf_path, max_pages=max_pages))
        except Exception as e:
            print(f"Error splitting {pdf_path}: {e}")
    return all_chunks


__all__ = ["split_pdf", "split_pdfs"]