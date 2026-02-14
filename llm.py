import os
import io
import re
import json
import time
import base64
import hashlib
import tempfile
from pathlib import Path
from typing import Optional, Dict, List, Tuple

import requests
from dotenv import load_dotenv

from googleapiclient.discovery import build
from googleapiclient.http import MediaIoBaseDownload, MediaFileUpload
from google.oauth2.credentials import Credentials
from google_auth_oauthlib.flow import InstalledAppFlow
from google.auth.transport.requests import Request

# ============================================================
# ENV
# ============================================================
load_dotenv()

OPENAI_API_KEY = (os.getenv("OPENAI_API_KEY") or "").strip()
if not OPENAI_API_KEY:
    raise RuntimeError("OPENAI_API_KEY not set. Put it in .env as OPENAI_API_KEY=...")

# Non-interactive slot selection (1-based index)
SLOT_CHOICE = (os.getenv("SLOT_CHOICE") or "").strip()

# Optional model override
MODEL = (os.getenv("OPENAI_MODEL") or "gpt-5").strip()

# Shared drives flag from env (optional)
USE_SHARED_DRIVES = (os.getenv("USE_SHARED_DRIVES") or "").strip().lower() in ("1", "true", "yes", "y")

# Headless auth (optional)
HEADLESS_AUTH = (os.getenv("HEADLESS_AUTH") or "").strip().lower() in ("1", "true", "yes", "y")

# Second pass rerun failed tasks (optional)
RERUN_FAILED_ONCE = (os.getenv("RERUN_FAILED_ONCE") or "true").strip().lower() in ("1", "true", "yes", "y")

# ============================================================
# CONFIG (Drive)
# ============================================================
SCOPES = ["https://www.googleapis.com/auth/drive"]
CREDENTIALS_FILE = Path("credentials.json")
TOKEN_FILE = Path("token.json")
FOLDER_MIME = "application/vnd.google-apps.folder"
GOOGLE_DOC_MIME = "application/vnd.google-apps.document"

ROOT_2026_FOLDER_NAME = "2026"  # can be nested anywhere in My Drive

# ============================================================
# CONFIG (Local Prompt Folder)
# ============================================================
PROMPT_DIR = Path("prompt")
if not PROMPT_DIR.exists():
    raise FileNotFoundError("Local 'prompt/' folder not found. Put all prompts inside ./prompt/")

# ============================================================
# CONFIG (Prompts Mapping: Drive Folder -> Prompt File)
# ============================================================
FOLDER_TO_PROMPT = {
    "3. Introduction Video": "intro-prompt.txt",
    "4. Mock Interview (First Call)": "mock-prompt.txt",
    "5. Project Scenarios": "project-scenario.txt",
    "6. 30 Questions Related to Their Niche": "niche-prompt.txt",
    "7. 50 Questions Related to the Resume": "CV-prompt.txt",
    "8. Tools & Technology Videos": "Tools-Technology-prompt.txt",
    "9. System Design Video (with Draw.io)": "System-design.txt",
    "10. Persona Video": "persona.txt",
    "11. Small Talk": "smalltalk.txt",
    "12. JD Video": "JD-prompt.txt",
}

# Prompts that should include candidate CV as input_file
PROMPT_NEEDS_CV = {
    "project-scenario.txt",
    "niche-prompt.txt",
    "CV-prompt.txt",
    "Tools-Technology-prompt.txt",
    "persona.txt",
    "JD-prompt.txt",
    "System-design.txt",
}

# Prompts that require diagram images from SAME deliverable folder
# (We will attach as input_image base64, not input_file)
PROMPT_NEEDS_PNG = {
    "Tools-Technology-prompt.txt",
    "System-design.txt",
    "JD-prompt.txt",
}

# ============================================================
# CONFIG (Reference PDFs) - now inside local ./pdf folder
# ============================================================
PDF_DIR = Path("pdf")
if not PDF_DIR.exists():
    raise FileNotFoundError("Local 'pdf/' folder not found. Put reference PDFs inside ./pdf/")

NICHE_REFERENCE_PDF = PDF_DIR / "Niche-Questions.pdf"
MOCK_REFERENCE_PDF = PDF_DIR / "31-Questions.pdf"

if not NICHE_REFERENCE_PDF.exists():
    raise FileNotFoundError("pdf/Niche-Questions.pdf not found (required for niche-prompt.txt).")
if not MOCK_REFERENCE_PDF.exists():
    raise FileNotFoundError("pdf/31-Questions.pdf not found (required for mock-prompt.txt).")

# ============================================================
# CONFIG (OpenAI)
# ============================================================
OPENAI_RESPONSES_URL = "https://api.openai.com/v1/responses"
OPENAI_FILES_URL = "https://api.openai.com/v1/files"
OPENAI_FILE_PURPOSE = "user_data"

MAX_TRANSCRIPT_CHARS = 250_000  # safety limit

# ============================================================
# MIME types
# ============================================================
DOCX_MIME = "application/vnd.openxmlformats-officedocument.wordprocessingml.document"
DOC_MIME = "application/msword"

# ============================================================
# Windows-safe local filenames (fix Errno 22)
# ============================================================
_WINDOWS_FORBIDDEN = r'<>:"/\\|?*'
_WINDOWS_RESERVED = {
    "CON", "PRN", "AUX", "NUL",
    *(f"COM{i}" for i in range(1, 10)),
    *(f"LPT{i}" for i in range(1, 10)),
}

def safe_local_filename(name: str, fallback: str = "file.txt") -> str:
    name = (name or "").strip() or fallback
    name = re.sub(f"[{re.escape(_WINDOWS_FORBIDDEN)}]", "_", name)
    name = re.sub(r"\s+", " ", name).strip()
    name = name.rstrip(" .")

    stem, dot, ext = name.partition(".")
    if stem.upper() in _WINDOWS_RESERVED:
        stem = f"_{stem}"
    name = stem + (dot + ext if dot else "")

    return name or fallback

# ============================================================
# Google Drive Auth
# ============================================================
def get_drive_service():
    creds = None
    if TOKEN_FILE.exists():
        creds = Credentials.from_authorized_user_file(str(TOKEN_FILE), SCOPES)

    # If token exists but scopes changed, refresh will fail -> force new login
    if creds and set(creds.scopes or []) != set(SCOPES):
        print("[AUTH] token.json scopes mismatch. Deleting token.json and re-authenticating...")
        TOKEN_FILE.unlink(missing_ok=True)
        creds = None

    if not creds or not creds.valid:
        if creds and creds.expired and creds.refresh_token:
            try:
                creds.refresh(Request())
            except Exception as e:
                print(f"[AUTH] Refresh failed ({e}). Re-authenticating...")
                TOKEN_FILE.unlink(missing_ok=True)
                creds = None

        if not creds or not creds.valid:
            if not CREDENTIALS_FILE.exists():
                raise FileNotFoundError("credentials.json not found next to this script.")
            flow = InstalledAppFlow.from_client_secrets_file(str(CREDENTIALS_FILE), SCOPES)
            if HEADLESS_AUTH:
                creds = flow.run_console()
            else:
                creds = flow.run_local_server(port=0)
            TOKEN_FILE.write_text(creds.to_json(), encoding="utf-8")

    return build("drive", "v3", credentials=creds)

def _list_kwargs():
    if USE_SHARED_DRIVES:
        return {"supportsAllDrives": True, "includeItemsFromAllDrives": True, "corpora": "allDrives"}
    return {}

def _get_media_kwargs():
    if USE_SHARED_DRIVES:
        return {"supportsAllDrives": True}
    return {}

def _write_kwargs():
    if USE_SHARED_DRIVES:
        return {"supportsAllDrives": True}
    return {}

# ============================================================
# Drive helpers
# ============================================================
def _escape_drive_q_value(s: str) -> str:
    return s.replace("'", "''")

def drive_find_child(service, parent_id: str, name: str, mime_type: Optional[str] = None):
    safe_name = _escape_drive_q_value(name)
    q_parts = [f"'{parent_id}' in parents", "trashed = false", f"name = '{safe_name}'"]
    if mime_type:
        q_parts.append(f"mimeType = '{mime_type}'")
    q = " and ".join(q_parts)

    res = service.files().list(
        q=q, fields="files(id,name,mimeType,modifiedTime)", pageSize=50, **_list_kwargs()
    ).execute()
    files = res.get("files", []) or []
    return sorted(files, key=lambda f: f.get("modifiedTime") or "", reverse=True)[0] if files else None

def drive_list_children(service, parent_id: str, mime_type: Optional[str] = None):
    q_parts = [f"'{parent_id}' in parents", "trashed = false"]
    if mime_type:
        q_parts.append(f"mimeType = '{mime_type}'")
    q = " and ".join(q_parts)

    page_token = None
    while True:
        res = service.files().list(
            q=q,
            fields="nextPageToken, files(id,name,mimeType,size,modifiedTime)",
            pageSize=1000,
            pageToken=page_token,
            **_list_kwargs(),
        ).execute()
        for f in res.get("files", []):
            yield f
        page_token = res.get("nextPageToken")
        if not page_token:
            break

def drive_download_file(service, file_id: str, dest_path: Path):
    request = service.files().get_media(fileId=file_id, **_get_media_kwargs())
    with io.FileIO(str(dest_path), "wb") as fh:
        downloader = MediaIoBaseDownload(fh, request, chunksize=1024 * 1024 * 8)
        done = False
        while not done:
            _, done = downloader.next_chunk()

def drive_export_google_doc_to_pdf(service, file_id: str, dest_path: Path):
    request = service.files().export_media(fileId=file_id, mimeType="application/pdf", **_get_media_kwargs())
    with io.FileIO(str(dest_path), "wb") as fh:
        downloader = MediaIoBaseDownload(fh, request, chunksize=1024 * 1024 * 8)
        done = False
        while not done:
            _, done = downloader.next_chunk()

def drive_export_google_doc_to_text(service, file_id: str, dest_path: Path):
    request = service.files().export_media(fileId=file_id, mimeType="text/plain", **_get_media_kwargs())
    with io.FileIO(str(dest_path), "wb") as fh:
        downloader = MediaIoBaseDownload(fh, request, chunksize=1024 * 1024 * 8)
        done = False
        while not done:
            _, done = downloader.next_chunk()

def drive_download_or_export_text(service, f: dict, dest_path: Path):
    mt = f.get("mimeType")
    if mt == GOOGLE_DOC_MIME:
        drive_export_google_doc_to_text(service, f["id"], dest_path)
    else:
        drive_download_file(service, f["id"], dest_path)

def drive_upload_text(service, parent_id: str, filename: str, local_path: Path):
    existing = drive_find_child(service, parent_id, filename, None)
    media = MediaFileUpload(str(local_path), mimetype="text/plain", resumable=True)
    if existing:
        service.files().update(fileId=existing["id"], media_body=media, **_write_kwargs()).execute()
    else:
        meta = {"name": filename, "parents": [parent_id]}
        service.files().create(body=meta, media_body=media, fields="id", **_write_kwargs()).execute()

def drive_search_folder_anywhere(service, folder_name: str) -> List[dict]:
    safe_name = _escape_drive_q_value(folder_name)
    q = f"name = '{safe_name}' and mimeType = '{FOLDER_MIME}' and trashed=false"
    res = service.files().list(
        q=q, fields="files(id,name,parents,modifiedTime)", pageSize=200, **_list_kwargs()
    ).execute()
    return res.get("files", []) or []

def pick_best_named_folder(candidates: List[dict]) -> dict:
    return sorted(candidates, key=lambda c: c.get("modifiedTime") or "", reverse=True)[0]

# ============================================================
# SLOT SELECTION (supports SLOT_CHOICE env)
# ============================================================
def list_slot_folders(service, slots_parent_id: str) -> List[dict]:
    return sorted(list(drive_list_children(service, slots_parent_id, FOLDER_MIME)), key=lambda x: x["name"].lower())

def choose_slot(service, slots_parent_id: str) -> dict:
    slots = list_slot_folders(service, slots_parent_id)
    if not slots:
        raise RuntimeError("No slot folders found under 2026.")

    if SLOT_CHOICE.isdigit():
        idx = int(SLOT_CHOICE)
        if 1 <= idx <= len(slots):
            chosen = slots[idx - 1]
            print(f"[AUTO] Using SLOT_CHOICE={idx}: {chosen['name']}")
            return chosen
        raise RuntimeError(f"SLOT_CHOICE='{SLOT_CHOICE}' out of range (1..{len(slots)}).")

    print("\n" + "=" * 80)
    print("SELECT SLOT TO PROCESS")
    print("=" * 80)
    for i, s in enumerate(slots, start=1):
        print(f"  {i:2}. {s['name']}")
    print("  EXIT - Exit\n")

    while True:
        choice = input("Choose slot number (e.g. 1) or EXIT: ").strip().lower()
        if choice == "exit":
            raise SystemExit(0)
        if choice.isdigit():
            idx = int(choice)
            if 1 <= idx <= len(slots):
                return slots[idx - 1]
        print(" Invalid choice. Try again.")

# ============================================================
# Convert Office (DOCX/DOC) -> PDF via Google (no LibreOffice)
# ============================================================
def drive_convert_office_to_pdf_via_google(service, office_file_id: str, dest_pdf_path: Path) -> None:
    body = {"name": f"__TEMP_CONVERT__{office_file_id}", "mimeType": GOOGLE_DOC_MIME}
    temp = service.files().copy(fileId=office_file_id, body=body, fields="id", **_write_kwargs()).execute()
    temp_id = temp["id"]

    try:
        drive_export_google_doc_to_pdf(service, temp_id, dest_pdf_path)
    finally:
        try:
            service.files().delete(fileId=temp_id, **_write_kwargs()).execute()
        except Exception:
            pass

# ============================================================
# Resume discovery (in 2026/<Slot>/<Person>/)
# ============================================================
def is_resume_like(name: str) -> bool:
    n = name.lower()
    return ("resume" in n) or ("cv" in n)

def find_candidate_resume_file(service, person_folder_id: str) -> Optional[dict]:
    files = list(drive_list_children(service, person_folder_id, None))

    pdfs = [f for f in files if is_resume_like(f["name"]) and f.get("mimeType") == "application/pdf"]
    if pdfs:
        return sorted(pdfs, key=lambda x: x["name"])[0]

    gdocs = [f for f in files if is_resume_like(f["name"]) and f.get("mimeType") == GOOGLE_DOC_MIME]
    if gdocs:
        return sorted(gdocs, key=lambda x: x["name"])[0]

    word_docs = [f for f in files if is_resume_like(f["name"]) and f.get("mimeType") in (DOCX_MIME, DOC_MIME)]
    if word_docs:
        return sorted(word_docs, key=lambda x: x["name"])[0]

    anypdf = [f for f in files if f.get("mimeType") == "application/pdf"]
    if len(anypdf) == 1:
        return anypdf[0]

    anyword = [f for f in files if f.get("mimeType") in (DOCX_MIME, DOC_MIME)]
    if len(anyword) == 1:
        return anyword[0]

    return None

# ============================================================
# Transcript aggregation
#   - Default: only *_transcripts.txt
#   - JD folder: include *_transcripts.txt + any other .txt JD files (exclude LLM_OUTPUT__)
# ============================================================
def read_all_transcripts_in_folder(service, transcript_folder_id: str, folder_name: str) -> str:
    txts: List[dict] = []

    for f in drive_list_children(service, transcript_folder_id, None):
        if f.get("mimeType") == FOLDER_MIME:
            continue

        name = (f.get("name") or "")
        lname = name.lower()

        if not lname.endswith(".txt"):
            continue

        # Always ignore model outputs
        if lname.startswith("llm_output__"):
            continue

        if folder_name == "12. JD Video":
            # JD folder: include ALL .txt (both transcripts + JD text files)
            txts.append(f)
        else:
            # Other folders: include only transcripts produced by transcription script
            if lname.endswith("_transcripts.txt"):
                txts.append(f)

    txts = sorted(txts, key=lambda x: (x.get("name") or "").lower())

    combined_parts: List[str] = []
    with tempfile.TemporaryDirectory() as td2:
        td2 = Path(td2)
        for f in txts:
            drive_name = f.get("name") or "file.txt"
            safe_name = safe_local_filename(drive_name, "file.txt")
            local = td2 / f"{f['id']}__{safe_name}"

            drive_download_or_export_text(service, f, local)

            content = local.read_text(encoding="utf-8", errors="ignore").strip()
            if not content:
                continue
            combined_parts.append(f"===== {drive_name} =====\n{content}\n")

    combined = "\n".join(combined_parts).strip()
    if len(combined) > MAX_TRANSCRIPT_CHARS:
        combined = combined[:MAX_TRANSCRIPT_CHARS] + "\n\n[TRUNCATED DUE TO SIZE LIMIT]\n"
    return combined

def download_pngs_in_folder(service, folder_id: str, temp_dir: Path) -> List[Path]:
    png_paths: List[Path] = []
    for f in drive_list_children(service, folder_id, None):
        if f.get("mimeType") == FOLDER_MIME:
            continue
        name = (f.get("name") or "").lower()
        if not name.endswith(".png"):
            continue

        safe_name = safe_local_filename(f.get("name") or "diagram.png", "diagram.png")
        local_path = temp_dir / f"{f['id']}__{safe_name}"
        drive_download_file(service, f["id"], local_path)

        if local_path.exists() and local_path.stat().st_size > 0:
            png_paths.append(local_path)

    png_paths.sort(key=lambda p: p.name.lower())
    return png_paths

# ============================================================
# OpenAI helpers
# ============================================================
def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()

class OpenAIFileCache:
    def __init__(self, cache_path: Path = Path(".openai_file_cache.json")):
        self.cache_path = cache_path
        self.data: Dict[str, str] = {}
        if cache_path.exists():
            try:
                self.data = json.loads(cache_path.read_text(encoding="utf-8"))
            except Exception:
                self.data = {}

    def get(self, digest: str) -> Optional[str]:
        return self.data.get(digest)

    def set(self, digest: str, file_id: str):
        self.data[digest] = file_id
        self.cache_path.write_text(json.dumps(self.data, indent=2), encoding="utf-8")

def request_with_retries(method, url, *, max_tries=6, base_sleep=1.0, **kwargs):
    last_exc = None
    for attempt in range(1, max_tries + 1):
        try:
            r = method(url, **kwargs)
        except (requests.Timeout, requests.ConnectionError) as e:
            last_exc = e
            r = None

        if r is not None and r.status_code in (200, 201):
            return r

        retryable = False
        retry_after = None

        if r is None:
            retryable = True
        else:
            retryable = r.status_code in (429, 500, 502, 503, 504)
            retry_after = r.headers.get("retry-after")

        if not retryable or attempt == max_tries:
            if r is None and last_exc:
                raise last_exc
            return r

        sleep_s = base_sleep * (2 ** (attempt - 1))
        if retry_after and str(retry_after).isdigit():
            sleep_s = max(sleep_s, float(retry_after))

        time.sleep(sleep_s)

    if last_exc:
        raise last_exc
    raise RuntimeError("request_with_retries: exhausted retries with unknown error")

def openai_upload_file(path: Path, mime: str, cache: OpenAIFileCache) -> str:
    digest = sha256_file(path)
    cached = cache.get(digest)
    if cached:
        return cached

    headers = {"Authorization": f"Bearer {OPENAI_API_KEY}"}
    files = {"file": (path.name, open(path, "rb"), mime)}
    data = {"purpose": "user_data"}

    r = request_with_retries(requests.post, OPENAI_FILES_URL, headers=headers, files=files, data=data, timeout=600)
    if r.status_code not in (200, 201):
        raise RuntimeError(f"OpenAI Files upload failed: HTTP {r.status_code}: {r.text}")

    file_id = r.json()["id"]
    cache.set(digest, file_id)
    return file_id

def openai_upload_pdf(path: Path, cache: OpenAIFileCache) -> str:
    return openai_upload_file(path, "application/pdf", cache)

def png_to_input_image_part(png_path: Path) -> dict:
    b = png_path.read_bytes()
    b64 = base64.b64encode(b).decode("utf-8")
    data_url = f"data:image/png;base64,{b64}"
    return {"type": "input_image", "image_url": {"url": data_url}}

def openai_run_prompt(
    prompt_text: str,
    transcript_text: str,
    cv_pdf_file_id: Optional[str],
    niche_pdf_file_id: Optional[str],
    mock_pdf_file_id: Optional[str],
    image_paths: Optional[List[Path]] = None,
) -> str:
    headers = {"Authorization": f"Bearer {OPENAI_API_KEY}", "Content-Type": "application/json"}

    content_parts = [{"type": "input_text", "text": prompt_text}]

    if mock_pdf_file_id:
        content_parts.append({"type": "input_file", "file_id": mock_pdf_file_id})
    if niche_pdf_file_id:
        content_parts.append({"type": "input_file", "file_id": niche_pdf_file_id})
    if cv_pdf_file_id:
        content_parts.append({"type": "input_file", "file_id": cv_pdf_file_id})

    if image_paths:
        for p in image_paths:
            content_parts.append(png_to_input_image_part(p))

    content_parts.append({"type": "input_text", "text": f"\n\nTRANSCRIPT:\n{transcript_text}\n"})

    payload = {"model": MODEL, "input": [{"role": "user", "content": content_parts}]}

    r = request_with_retries(requests.post, OPENAI_RESPONSES_URL, headers=headers, json=payload, timeout=600)
    if r.status_code != 200:
        raise RuntimeError(f"OpenAI Responses failed: HTTP {r.status_code}: {r.text}")

    resp = r.json()

    out_texts = []
    for item in resp.get("output", []):
        for c in item.get("content", []):
            if c.get("type") == "output_text" and c.get("text"):
                out_texts.append(c["text"])

    final = "\n".join(t.strip() for t in out_texts if t and t.strip()).strip()
    if not final:
        final = json.dumps(resp, indent=2)
    return final

# ============================================================
# MAIN
# ============================================================
def main():
    service = get_drive_service()

    candidates = drive_search_folder_anywhere(service, ROOT_2026_FOLDER_NAME)
    if not candidates:
        raise RuntimeError(f"Could not find folder '{ROOT_2026_FOLDER_NAME}' anywhere in Drive.")

    base_2026 = pick_best_named_folder(candidates)
    slots_parent_id = base_2026["id"]

    slot = choose_slot(service, slots_parent_id)

    file_cache = OpenAIFileCache()

    # Upload reference PDFs once (cached)
    niche_pdf_file_id = openai_upload_pdf(NICHE_REFERENCE_PDF, file_cache)
    mock_pdf_file_id = openai_upload_pdf(MOCK_REFERENCE_PDF, file_cache)

    people = sorted(list(drive_list_children(service, slot["id"], FOLDER_MIME)), key=lambda x: x["name"])

    failed_tasks: List[Tuple[str, str, str, str]] = []
    # (person_name, folder_name, prompt_filename, out_name)

    for person in people:
        print("\n" + "=" * 90)
        print(f"[PERSON] {slot['name']} / {person['name']}")

        folder_nodes = sorted(list(drive_list_children(service, person["id"], FOLDER_MIME)), key=lambda x: x["name"])
        person_needs_cv = any((FOLDER_TO_PROMPT.get(fn["name"]) in PROMPT_NEEDS_CV) for fn in folder_nodes)

        resume_file = find_candidate_resume_file(service, person["id"])
        cv_pdf_file_id = None

        with tempfile.TemporaryDirectory() as td_str:
            td = Path(td_str)

            # --- CV upload (only if needed) ---
            if person_needs_cv:
                if resume_file:
                    local_cv_pdf = td / "candidate_cv.pdf"
                    mt = resume_file.get("mimeType")

                    try:
                        if mt == "application/pdf":
                            drive_download_file(service, resume_file["id"], local_cv_pdf)
                        elif mt == GOOGLE_DOC_MIME:
                            drive_export_google_doc_to_pdf(service, resume_file["id"], local_cv_pdf)
                        elif mt in (DOCX_MIME, DOC_MIME):
                            drive_convert_office_to_pdf_via_google(service, resume_file["id"], local_cv_pdf)
                        else:
                            local_any = td / safe_local_filename(resume_file.get("name") or "resume", "resume")
                            drive_download_file(service, resume_file["id"], local_any)
                            if local_any.suffix.lower() in (".docx", ".doc"):
                                drive_convert_office_to_pdf_via_google(service, resume_file["id"], local_cv_pdf)
                            else:
                                print(f"[WARN] Unsupported resume type for {person['name']}: {mt}")
                                local_cv_pdf = None

                        if local_cv_pdf and local_cv_pdf.exists():
                            cv_pdf_file_id = openai_upload_pdf(local_cv_pdf, file_cache)
                            print("[CV] Uploaded / cached.")
                    except Exception as e:
                        print(f"[WARN] CV processing/upload failed: {e}")
                        cv_pdf_file_id = None
                else:
                    print("[CV] Needed but no resume/CV found (continuing).")
            else:
                print("[CV] Not needed for this person (no CV-dependent prompts).")

            # --- Run deliverable prompts ---
            for folder_node in folder_nodes:
                folder_name = folder_node["name"]
                prompt_filename = FOLDER_TO_PROMPT.get(folder_name)
                if not prompt_filename:
                    continue

                prompt_path = PROMPT_DIR / prompt_filename
                if not prompt_path.exists():
                    print(f"[WARN] Prompt missing: {prompt_filename} (skip {person['name']}/{folder_name})")
                    continue

                out_name = f"LLM_OUTPUT__{prompt_filename.replace('.txt', '')}.txt"
                existing = drive_find_child(service, folder_node["id"], out_name, None)
                if existing:
                    print(f"[SKIP] Output exists: {person['name']}/{folder_name}/{out_name}")
                    continue

                transcript_text = read_all_transcripts_in_folder(service, folder_node["id"], folder_name)
                if not transcript_text.strip():
                    print(f"[SKIP] No transcripts: {person['name']}/{folder_name}")
                    continue

                prompt_text = prompt_path.read_text(encoding="utf-8", errors="ignore").strip()

                needs_cv = prompt_filename in PROMPT_NEEDS_CV
                use_cv_id = cv_pdf_file_id if needs_cv else None

                use_niche_id = niche_pdf_file_id if prompt_filename == "niche-prompt.txt" else None
                use_mock_id = mock_pdf_file_id if prompt_filename == "mock-prompt.txt" else None

                image_paths: List[Path] = []
                if prompt_filename in PROMPT_NEEDS_PNG:
                    image_paths = download_pngs_in_folder(service, folder_node["id"], td)
                    if not image_paths:
                        print(f"[WARN] No .png found in {person['name']}/{folder_name} but prompt expects diagram(s).")

                print(f"[RUN ] {person['name']}/{folder_name} -> {out_name}")

                try:
                    result = openai_run_prompt(
                        prompt_text=prompt_text,
                        transcript_text=transcript_text,
                        cv_pdf_file_id=use_cv_id,
                        niche_pdf_file_id=use_niche_id,
                        mock_pdf_file_id=use_mock_id,
                        image_paths=image_paths if image_paths else None,
                    )

                    local_out = td / safe_local_filename(out_name, out_name)
                    local_out.write_text(result.strip() + "\n", encoding="utf-8")
                    drive_upload_text(service, folder_node["id"], out_name, local_out)
                    print("[OK  ] uploaded")

                except Exception as e:
                    print(f"[FAIL] {person['name']}/{folder_name}: {type(e).__name__}: {e}")
                    failed_tasks.append((person["name"], folder_name, prompt_filename, out_name))

                time.sleep(0.5)

    # ============================================================
    # SECOND PASS: rerun failed tasks once
    # ============================================================
    if RERUN_FAILED_ONCE and failed_tasks:
        print("\n" + "=" * 90)
        print(f"[RERUN] Second pass for failed tasks: {len(failed_tasks)}")
        print("=" * 90)

        slot_people = {
            p["name"]: p
            for p in sorted(list(drive_list_children(service, slot["id"], FOLDER_MIME)), key=lambda x: x["name"])
        }

        for person_name, folder_name, prompt_filename, out_name in failed_tasks:
            try:
                person = slot_people.get(person_name)
                if not person:
                    print(f"[RERUN][SKIP] Person missing: {person_name}")
                    continue

                person_folders = {f["name"]: f for f in drive_list_children(service, person["id"], FOLDER_MIME)}
                folder_node = person_folders.get(folder_name)
                if not folder_node:
                    print(f"[RERUN][SKIP] Folder missing: {person_name}/{folder_name}")
                    continue

                existing = drive_find_child(service, folder_node["id"], out_name, None)
                if existing:
                    print(f"[RERUN][SKIP] Output exists now: {person_name}/{folder_name}/{out_name}")
                    continue

                prompt_path = PROMPT_DIR / prompt_filename
                if not prompt_path.exists():
                    print(f"[RERUN][SKIP] Prompt missing: {prompt_filename}")
                    continue

                transcript_text = read_all_transcripts_in_folder(service, folder_node["id"], folder_name)
                if not transcript_text.strip():
                    print(f"[RERUN][SKIP] No transcripts: {person_name}/{folder_name}")
                    continue

                prompt_text = prompt_path.read_text(encoding="utf-8", errors="ignore").strip()

                needs_cv = prompt_filename in PROMPT_NEEDS_CV
                cv_pdf_file_id_rerun = None
                if needs_cv:
                    resume_file = find_candidate_resume_file(service, person["id"])
                    if resume_file:
                        with tempfile.TemporaryDirectory() as td_str:
                            td = Path(td_str)
                            local_cv_pdf = td / "candidate_cv.pdf"
                            mt = resume_file.get("mimeType")

                            if mt == "application/pdf":
                                drive_download_file(service, resume_file["id"], local_cv_pdf)
                            elif mt == GOOGLE_DOC_MIME:
                                drive_export_google_doc_to_pdf(service, resume_file["id"], local_cv_pdf)
                            elif mt in (DOCX_MIME, DOC_MIME):
                                drive_convert_office_to_pdf_via_google(service, resume_file["id"], local_cv_pdf)

                            if local_cv_pdf.exists():
                                cv_pdf_file_id_rerun = openai_upload_pdf(local_cv_pdf, file_cache)

                use_niche_id = niche_pdf_file_id if prompt_filename == "niche-prompt.txt" else None
                use_mock_id = mock_pdf_file_id if prompt_filename == "mock-prompt.txt" else None

                image_paths = []
                if prompt_filename in PROMPT_NEEDS_PNG:
                    with tempfile.TemporaryDirectory() as td_str:
                        td = Path(td_str)
                        image_paths = download_pngs_in_folder(service, folder_node["id"], td)

                print(f"[RERUN][RUN ] {person_name}/{folder_name} -> {out_name}")

                result = openai_run_prompt(
                    prompt_text=prompt_text,
                    transcript_text=transcript_text,
                    cv_pdf_file_id=cv_pdf_file_id_rerun if needs_cv else None,
                    niche_pdf_file_id=use_niche_id,
                    mock_pdf_file_id=use_mock_id,
                    image_paths=image_paths if image_paths else None,
                )

                with tempfile.TemporaryDirectory() as td_str:
                    td = Path(td_str)
                    local_out = td / safe_local_filename(out_name, out_name)
                    local_out.write_text(result.strip() + "\n", encoding="utf-8")
                    drive_upload_text(service, folder_node["id"], out_name, local_out)

                print("[RERUN][OK  ] uploaded")

            except Exception as e:
                print(f"[RERUN][FAIL] {person_name}/{folder_name}: {type(e).__name__}: {e}")

if __name__ == "__main__":
    main()
