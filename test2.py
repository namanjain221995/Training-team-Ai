import os
import io
import json
import time
import hashlib
import tempfile
from pathlib import Path
from typing import Optional, Dict, List

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

# Non-interactive slot selection
# Example: SLOT_CHOICE=2 in .env -> picks 2nd slot in sorted list
SLOT_CHOICE = (os.getenv("SLOT_CHOICE") or "").strip()

# Optional model override
MODEL = (os.getenv("OPENAI_MODEL") or "gpt-5").strip()

# Shared drives flag from env (optional)
USE_SHARED_DRIVES = (os.getenv("USE_SHARED_DRIVES") or "").strip().lower() in ("1", "true", "yes", "y")

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
}

# Prompts that require PNG diagram(s) from the SAME Drive deliverable folder (Option A)
PROMPT_NEEDS_PNG = {
    "Tools-Technology-prompt.txt",  # Day3 Team Structure .png is stored inside Tools folder
    "System-design.txt",            # Day6 diagrams .png are stored inside System Design folder
}

# ============================================================
# CONFIG (Reference PDFs) - now inside local ./pdf folder
# ============================================================
PDF_DIR = Path("pdf")
if not PDF_DIR.exists():
    raise FileNotFoundError("Local 'pdf/' folder not found. Put reference PDFs inside ./pdf/")

NICHE_REFERENCE_PDF = PDF_DIR / "Niche-Questions.pdf"  # used for niche-prompt.txt
MOCK_REFERENCE_PDF = PDF_DIR / "31-Questions.pdf"      # used for mock-prompt.txt

if not NICHE_REFERENCE_PDF.exists():
    raise FileNotFoundError("pdf/Niche-Questions.pdf not found (required for niche-prompt.txt).")
if not MOCK_REFERENCE_PDF.exists():
    raise FileNotFoundError("pdf/31-Questions.pdf not found (required for mock-prompt.txt).")

# ============================================================
# CONFIG (OpenAI)
# ============================================================
OPENAI_RESPONSES_URL = "https://api.openai.com/v1/responses"
OPENAI_FILES_URL = "https://api.openai.com/v1/files"
OPENAI_FILE_PURPOSE = "user_data"  # recommended for model inputs

MAX_TRANSCRIPT_CHARS = 250_000  # safety limit

# ============================================================
# MIME types
# ============================================================
DOCX_MIME = "application/vnd.openxmlformats-officedocument.wordprocessingml.document"
DOC_MIME = "application/msword"


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
            # NOTE: In Docker/headless, consider flow.run_console()
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
    with io.FileIO(dest_path, "wb") as fh:
        downloader = MediaIoBaseDownload(fh, request, chunksize=1024 * 1024 * 8)
        done = False
        while not done:
            _, done = downloader.next_chunk()


def drive_export_google_doc_to_pdf(service, file_id: str, dest_path: Path):
    request = service.files().export_media(fileId=file_id, mimeType="application/pdf", **_get_media_kwargs())
    with io.FileIO(dest_path, "wb") as fh:
        downloader = MediaIoBaseDownload(fh, request, chunksize=1024 * 1024 * 8)
        done = False
        while not done:
            _, done = downloader.next_chunk()


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
# Transcript aggregation (in 2026/<Slot>/<Person>/<FolderName>/)
# ============================================================
def read_all_transcripts_in_folder(service, transcript_folder_id: str) -> str:
    txts = []
    for f in drive_list_children(service, transcript_folder_id, None):
        if f.get("mimeType") == FOLDER_MIME:
            continue
        name = (f.get("name") or "").lower()
        if not name.endswith(".txt"):
            continue
        if name.startswith("llm_output__"):
            continue
        txts.append(f)

    txts = sorted(txts, key=lambda x: (x.get("name") or "").lower())

    combined_parts = []
    with tempfile.TemporaryDirectory() as td2:
        td2 = Path(td2)
        for f in txts:
            local = td2 / (f.get("name") or "file.txt")
            drive_download_file(service, f["id"], local)
            content = local.read_text(encoding="utf-8", errors="ignore").strip()
            if not content:
                continue
            combined_parts.append(f"===== {f['name']} =====\n{content}\n")

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

        local_path = temp_dir / (f.get("name") or "diagram.png")
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


def openai_upload_file(path: Path, mime: str, cache: OpenAIFileCache) -> str:
    digest = sha256_file(path)
    cached = cache.get(digest)
    if cached:
        return cached

    headers = {"Authorization": f"Bearer {OPENAI_API_KEY}"}
    files = {"file": (path.name, open(path, "rb"), mime)}
    data = {"purpose": OPENAI_FILE_PURPOSE}

    r = requests.post(OPENAI_FILES_URL, headers=headers, files=files, data=data, timeout=600)
    if r.status_code not in (200, 201):
        raise RuntimeError(f"OpenAI Files upload failed: HTTP {r.status_code}: {r.text}")

    file_id = r.json()["id"]
    cache.set(digest, file_id)
    return file_id


def openai_upload_pdf(path: Path, cache: OpenAIFileCache) -> str:
    return openai_upload_file(path, "application/pdf", cache)


def openai_upload_png(path: Path, cache: OpenAIFileCache) -> str:
    return openai_upload_file(path, "image/png", cache)


def openai_run_prompt(
    prompt_text: str,
    transcript_text: str,
    cv_pdf_file_id: Optional[str],
    niche_pdf_file_id: Optional[str],
    mock_pdf_file_id: Optional[str],
    extra_file_ids: Optional[List[str]] = None,
) -> str:
    headers = {"Authorization": f"Bearer {OPENAI_API_KEY}", "Content-Type": "application/json"}

    content_parts = [{"type": "input_text", "text": prompt_text}]

    if mock_pdf_file_id:
        content_parts.append({"type": "input_file", "file_id": mock_pdf_file_id})
    if niche_pdf_file_id:
        content_parts.append({"type": "input_file", "file_id": niche_pdf_file_id})
    if cv_pdf_file_id:
        content_parts.append({"type": "input_file", "file_id": cv_pdf_file_id})

    if extra_file_ids:
        for fid in extra_file_ids:
            content_parts.append({"type": "input_file", "file_id": fid})

    content_parts.append({"type": "input_text", "text": f"\n\nTRANSCRIPT:\n{transcript_text}\n"})

    payload = {"model": MODEL, "input": [{"role": "user", "content": content_parts}]}

    r = requests.post(OPENAI_RESPONSES_URL, headers=headers, json=payload, timeout=600)
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

    for person in people:
        print("\n" + "=" * 90)
        print(f"[PERSON] {slot['name']} / {person['name']}")

        resume_file = find_candidate_resume_file(service, person["id"])
        cv_pdf_file_id = None

        with tempfile.TemporaryDirectory() as td_str:
            td = Path(td_str)

            # --- CV upload (only if needed later) ---
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
                        local_any = td / (resume_file.get("name") or "resume")
                        drive_download_file(service, resume_file["id"], local_any)
                        if local_any.suffix.lower() in (".docx", ".doc"):
                            drive_convert_office_to_pdf_via_google(service, resume_file["id"], local_cv_pdf)
                        else:
                            print(
                                f"[WARN] Unsupported resume type for {slot['name']}/{person['name']}: "
                                f"{mt} ({resume_file.get('name')})"
                            )
                            local_cv_pdf = None

                    if local_cv_pdf and local_cv_pdf.exists():
                        cv_pdf_file_id = openai_upload_pdf(local_cv_pdf, file_cache)
                        print("[CV] Uploaded / cached.")
                except Exception as e:
                    print(f"[WARN] CV processing/upload failed: {e}")
                    cv_pdf_file_id = None
            else:
                print("[CV] No resume/CV file found (continuing).")

            # --- Run deliverable prompts ---
            folder_nodes = sorted(list(drive_list_children(service, person["id"], FOLDER_MIME)), key=lambda x: x["name"])
            for folder_node in folder_nodes:
                folder_name = folder_node["name"]
                prompt_filename = FOLDER_TO_PROMPT.get(folder_name)
                if not prompt_filename:
                    continue

                prompt_path = PROMPT_DIR / prompt_filename
                if not prompt_path.exists():
                    print(f"[WARN] Prompt missing: {prompt_filename} (skip {person['name']}/{folder_name})")
                    continue

                transcript_text = read_all_transcripts_in_folder(service, folder_node["id"])
                if not transcript_text.strip():
                    print(f"[SKIP] No transcripts: {person['name']}/{folder_name}")
                    continue

                prompt_text = prompt_path.read_text(encoding="utf-8", errors="ignore").strip()

                needs_cv = prompt_filename in PROMPT_NEEDS_CV
                use_cv_id = cv_pdf_file_id if needs_cv else None

                use_niche_id = niche_pdf_file_id if prompt_filename == "niche-prompt.txt" else None
                use_mock_id = mock_pdf_file_id if prompt_filename == "mock-prompt.txt" else None

                extra_file_ids: List[str] = []
                if prompt_filename in PROMPT_NEEDS_PNG:
                    png_paths = download_pngs_in_folder(service, folder_node["id"], td)
                    if not png_paths:
                        print(f"[WARN] No .png found in {person['name']}/{folder_name} but prompt expects diagram(s).")
                    for p in png_paths:
                        try:
                            extra_file_ids.append(openai_upload_png(p, file_cache))
                        except Exception as e:
                            print(f"[WARN] PNG upload failed ({p.name}): {e}")

                out_name = f"LLM_OUTPUT__{prompt_filename.replace('.txt', '')}.txt"
                existing = drive_find_child(service, folder_node["id"], out_name, None)
                if existing:
                    print(f"[SKIP] Output exists: {person['name']}/{folder_name}/{out_name}")
                    continue

                print(f"[RUN ] {person['name']}/{folder_name} -> {out_name}")

                try:
                    result = openai_run_prompt(
                        prompt_text=prompt_text,
                        transcript_text=transcript_text,
                        cv_pdf_file_id=use_cv_id,
                        niche_pdf_file_id=use_niche_id,
                        mock_pdf_file_id=use_mock_id,
                        extra_file_ids=extra_file_ids if extra_file_ids else None,
                    )

                    local_out = td / out_name
                    local_out.write_text(result.strip() + "\n", encoding="utf-8")
                    drive_upload_text(service, folder_node["id"], out_name, local_out)
                    print("[OK  ] uploaded")

                except Exception as e:
                    print(f"[FAIL] {person['name']}/{folder_name}: {type(e).__name__}: {e}")

                time.sleep(0.5)


if __name__ == "__main__":
    main()
