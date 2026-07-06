import os
import re
import base64
import zipfile

from app.config import SARVAM_API_KEY, SARVAM_OUTPUT_DIR, EXTRACT_DIR, IMAGES_DIR

_sarvam_client = None

FOLLOWING_WINDOW = 8000     # chars captured after an image for desc/caption search
FALLBACK_WINDOW = 5000      # used as embedding_text if no desc/caption found

def _get_client():
    global _sarvam_client
    if _sarvam_client is None:
        from sarvamai import SarvamAI
        _sarvam_client = SarvamAI(api_subscription_key=SARVAM_API_KEY)
    return _sarvam_client


def process_pdf_chunk(chunk_info: dict, language: str = "en-IN") -> str | None:
    from sarvamai.core.api_error import ApiError

    chunk_path = chunk_info["chunk_path"]
    pdf_name = chunk_info["pdf_name"]
    chunk_num = chunk_info["chunk_num"]
    client = _get_client()

    try:
        job = client.document_intelligence.create_job(language=language, output_format="md")
        print(f"  Job {job.job_id}: {pdf_name} part {chunk_num}")
        job.upload_file(chunk_path)
        job.start()
        status = job.wait_until_complete()

        if status.job_state == "Completed":
            output_zip = os.path.join(SARVAM_OUTPUT_DIR, f"{pdf_name}_part_{chunk_num}.zip")
            job.download_output(output_zip)
            print(f"  Saved: {output_zip}")
            return output_zip
        print(f"  Unexpected job status: {status.job_state}")
        return None

    except ApiError as e:
        if e.status_code == 403:
            print("  Invalid Sarvam API key")
        elif e.status_code == 429:
            print("  Sarvam rate limit exceeded")
        else:
            print(f"  Sarvam API Error {e.status_code}: {e.body}")
        return None
    except Exception as e:
        print(f"  Error: {e}")
        return None


def process_pdf_chunks(pdf_chunks: list, language: str = "en-IN") -> list:
    zips = []
    for idx, chunk_info in enumerate(pdf_chunks, 1):
        print(f"\n--- Sarvam chunk {idx}/{len(pdf_chunks)} ---")
        z = process_pdf_chunk(chunk_info, language=language)
        if z:
            zips.append(z)
    return zips


def unzip_outputs(zip_dir: str = SARVAM_OUTPUT_DIR, zip_names: list | None = None) -> list:
    extracted_files = []
    zip_files = [f for f in os.listdir(zip_dir) if f.endswith('.zip')]
    if zip_names is not None:
        zip_files = [f for f in zip_files if f in zip_names]
    if not zip_files:
        print(f"No ZIP files found in {zip_dir}")
        return []
    print(f"Found {len(zip_files)} ZIP files")
    for zip_file in zip_files:
        zip_path = os.path.join(zip_dir, zip_file)
        try:
            with zipfile.ZipFile(zip_path, 'r') as zf:
                for md_file in [f for f in zf.namelist() if f.endswith('.md')]:
                    md_name = os.path.basename(md_file)
                    out_path = os.path.join(EXTRACT_DIR, f"{zip_file.replace('.zip','')}_{md_name}")
                    with zf.open(md_file) as src, open(out_path, 'w', encoding='utf-8') as tgt:
                        tgt.write(src.read().decode('utf-8'))
                    extracted_files.append((zip_file, out_path))
        except Exception as e:
            print(f"Error unzipping {zip_file}: {e}")
    return extracted_files


def clean_raw_markdown(markdown_text: str) -> str:
    text = re.sub(r'\b(\w+)(\s+\1\b)+', r'\1', markdown_text)
    text = re.sub(r'\b((?:\w+\s+){1,5}\w+)(\s+\1\b)+', r'\1', text)
    text = re.sub(r'([^\w\s])\1{3,}', r'\1', text)
    return text

_IMAGE_PATTERN = (r'!\[([^\]]*)\]'
                  r'\(data:image\/(?:png|jpeg|jpg);base64,([A-Za-z0-9+/=\n\r]+)\)')
 
_CAPTION_RE = re.compile(
    r'(?:^|\n)\s*\*?\s*(?:FIG(?:URE)?|CHART|TABLE)\.?\s*\d+\s*[.:]\s*(.+?)(?:\*|\n\n|\Z)',
    re.IGNORECASE | re.DOTALL,
)

def parse_markdown(md_path: str) -> dict:
    with open(md_path, 'r', encoding='utf-8') as f:
        raw_text = f.read()
        
    markdown_text = clean_raw_markdown(raw_text)
    md_filename = os.path.basename(md_path)
    content = {"text": [], "images": [], "tables": []}
 
    # TEXT — everything except the base64 image blobs
    text_only = re.sub(_IMAGE_PATTERN, '', markdown_text).strip()
    for para in text_only.split('\n\n'):
        para = para.strip()
        if para:
            content["text"].append({"chunk": para, "source": md_filename})
        
    # IMAGES — next 8000 chars after each image used to find desc/caption;
    # if neither is found, first 5000 chars of that window is the embedding text.
    image_matches = list(re.finditer(_IMAGE_PATTERN, markdown_text))
    for i, match in enumerate(image_matches):
        image_desc, base64_str = match.group(1), match.group(2)
        match_end = match.end()
        following = markdown_text[match_end:match_end + FOLLOWING_WINDOW]

        cap_match = _CAPTION_RE.search(following)
        caption = cap_match.group(1).strip() if cap_match else ""

        desc_match = re.search(r'\*(.+?)\*', following, re.DOTALL)
        description = desc_match.group(1).strip() if desc_match else ""

        if description and caption:
            embedding_text = " ".join(p for p in [description, caption] if p)
        else:
            embedding_text = following[:FALLBACK_WINDOW].strip() or image_desc
 
        content["images"].append({
            "base64": base64_str, "alt_text": image_desc,
            "sarvam_description": description, "caption": caption,
            "embedding_text": embedding_text, "source": md_filename
        })

    # TABLES
    current = []
    for line in markdown_text.split('\n'):
        if line.strip().lstrip('*').strip().startswith('|'):
            current.append(line.strip().lstrip('*').strip())
        elif current:
            content["tables"].append({"markdown": '\n'.join(current), "source": md_filename, "type": "markdown"})
            current = []
    if current:
        content["tables"].append({"markdown": '\n'.join(current), "source": md_filename, "type": "markdown"})
 
    _TABLE_CAP_RE = re.compile(r'\*\s*Table\s+\d+\s*[:.]\s*(.+?)\*', re.IGNORECASE | re.DOTALL)
 
    for m in re.finditer(r'<table[^>]*>.*?</table>', markdown_text, re.DOTALL):
        preceding = markdown_text[max(0, m.start() - FOLLOWING_WINDOW):m.start()]
        cap_match = list(_TABLE_CAP_RE.finditer(preceding))
        if cap_match:
            caption = cap_match[-1].group(1).strip()
        else:
            following = markdown_text[m.end():m.end() + FOLLOWING_WINDOW]
            cap_match = _CAPTION_RE.search(following)
            caption = cap_match.group(1).strip() if cap_match else (following[:FALLBACK_WINDOW].strip() or "HTML Table")
        content["tables"].append({
            "html": m.group(0), "caption": caption,
            "source": md_filename, "type": "html"
        })
 
    return content


def decode_and_save_images(image_blocks: list, output_dir: str = IMAGES_DIR):
    for idx, img in enumerate(image_blocks):
        try:
            source = img["source"].replace(".md", "")
            image_bytes = base64.b64decode(img["base64"])
            path = os.path.join(output_dir, f"{source}_img_{idx}.png")
            img["image_path"] = path
            with open(path, 'wb') as f:
                f.write(image_bytes)
        except Exception as e:
            print(f"Error decoding image {idx}: {e}")
 
 
__all__ = [
    "process_pdf_chunk", "process_pdf_chunks",
    "unzip_outputs", "parse_markdown",
    "clean_raw_markdown", "decode_and_save_images",
]