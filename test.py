import os
import io
import math
import time
import random
import tempfile
import subprocess
import contextlib
import wave
from pathlib import Path
from typing import Optional, List

import requests
from dotenv import load_dotenv

from googleapiclient.discovery import build
from googleapiclient.http import MediaIoBaseDownload, MediaFileUpload
from googleapiclient.errors import HttpError
from google.oauth2.credentials import Credentials
from google_auth_oauthlib.flow import InstalledAppFlow
from google.auth.transport.requests import Request

# =========================
# LOAD ENV (.env)
# =========================
load_dotenv()

# =========================
# CONFIG
# =========================
ROOT_2026_FOLDER_NAME = "2026"  # the folder containing Slot folders (can be nested anywhere in My Drive)

VIDEO_EXTS = {".mp4", ".mkv", ".mov", ".avi", ".webm", ".m4v"}

FOLDER_NAMES_TO_PROCESS = [
    "3. Introduction Video",
    "4. Mock Interview (First Call)",
    "5. Project Scenarios",
    "6. 30 Questions Related to Their Niche",
    "7. 50 Questions Related to the Resume",
    "8. Tools & Technology Videos",
    "9. System Design Video (with Draw.io)",
    "10. Persona Video",
    "11. Small Talk",
    "12. JD Video",
]

# OpenAI Transcription
MODEL = "gpt-4o-transcribe-diarize"
LANGUAGE = "en"  # set None for auto-detect
FORCE_RETRANSCRIBE = False
INCLUDE_SPEAKER = False  # True => "m:ss: Speaker X: text"
CHUNKING_STRATEGY = "auto"
CHUNK_SECONDS = 540  # 9 minutes
OVERLAP_SECONDS = 2
API_URL = "https://api.openai.com/v1/audio/transcriptions"

# Google Drive OAuth
SCOPES = ["https://www.googleapis.com/auth/drive"]
CREDENTIALS_FILE = Path("credentials.json")
TOKEN_FILE = Path("token.json")
FOLDER_MIME = "application/vnd.google-apps.folder"

USE_SHARED_DRIVES = (os.getenv("USE_SHARED_DRIVES") or "").strip().lower() in ("1", "true", "yes", "y")
SLOT_CHOICE_ENV = (os.getenv("SLOT_CHOICE") or "").strip()

# =========================
# AUTH: OpenAI
# =========================
api_key = (os.getenv("OPENAI_API_KEY") or "").strip()
if not api_key:
    raise RuntimeError("OPENAI_API_KEY not set. Put it in .env as OPENAI_API_KEY=...")

# =========================
# RETRY WRAPPER (Drive API)
# =========================
def execute_with_retries(request, *, max_retries: int = 8, base_sleep: float = 1.0):
    """
    Retries transient Google API errors: 429, 500, 502, 503, 504.
    Exponential backoff + jitter.
    """
    for attempt in range(max_retries):
        try:
            return request.execute()
        except HttpError as e:
            status = getattr(e.resp, "status", None)
            if status in (429, 500, 502, 503, 504):
                if attempt == max_retries - 1:
                    raise
                sleep = (base_sleep * (2 ** attempt)) + random.random()
                print(f"[WARN] Drive API transient error HTTP {status}. Retrying in {sleep:.1f}s...")
                time.sleep(sleep)
                continue
            raise

# =========================
# Google Drive Auth
# =========================
def get_drive_service():
    creds = None
    if TOKEN_FILE.exists():
        creds = Credentials.from_authorized_user_file(str(TOKEN_FILE), SCOPES)

    if not creds or not creds.valid:
        if creds and creds.expired and creds.refresh_token:
            creds.refresh(Request())
        else:
            if not CREDENTIALS_FILE.exists():
                raise FileNotFoundError("credentials.json not found next to this script.")
            flow = InstalledAppFlow.from_client_secrets_file(str(CREDENTIALS_FILE), SCOPES)
            # NOTE: In Docker/headless, consider changing to: flow.run_console()
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

# =========================
# Drive helpers
# =========================
def _escape_drive_q_value(s: str) -> str:
    """
    Escape a value embedded inside Drive API 'q' string single quotes.

    Drive query string literals are wrapped with single quotes. A single quote inside
    must be backslash-escaped: \'
    Also escape backslashes for safety.
    """
    return s.replace("\\", "\\\\").replace("'", "\\'")

def drive_list_children(service, parent_id: str, mime_type: Optional[str] = None):
    q_parts = [f"'{parent_id}' in parents", "trashed = false"]
    if mime_type:
        q_parts.append(f"mimeType = '{_escape_drive_q_value(mime_type)}'")
    q = " and ".join(q_parts)

    page_token = None
    while True:
        res = execute_with_retries(
            service.files().list(
                q=q,
                fields="nextPageToken, files(id,name,mimeType,size,modifiedTime)",
                pageSize=1000,
                pageToken=page_token,
                **_list_kwargs(),
            )
        )
        for f in res.get("files", []):
            yield f
        page_token = res.get("nextPageToken")
        if not page_token:
            break

def drive_find_child(service, parent_id: str, name: str, mime_type: Optional[str] = None):
    """
    Robust lookup by name that avoids Drive q-escaping pitfalls:
    - lists children (optionally filtered by mimeType)
    - matches the exact name in Python
    """
    for f in drive_list_children(service, parent_id, mime_type):
        if f.get("name") == name:
            return f
    return None

def drive_download_file(service, file_id: str, dest_path: Path):
    request = service.files().get_media(fileId=file_id, **_get_media_kwargs())
    with io.FileIO(dest_path, "wb") as fh:
        downloader = MediaIoBaseDownload(fh, request, chunksize=1024 * 1024 * 8)
        done = False
        while not done:
            try:
                status, done = downloader.next_chunk()
            except HttpError as e:
                status_code = getattr(e.resp, "status", None)
                if status_code in (429, 500, 502, 503, 504):
                    sleep = 1.0 + random.random()
                    print(f"[WARN] Download transient error HTTP {status_code}. Retrying in {sleep:.1f}s...")
                    time.sleep(sleep)
                    continue
                raise
            if status:
                print(f"      [DL  ] {int(status.progress() * 100)}%")

def drive_upload_text(service, parent_id: str, filename: str, local_path: Path):
    existing = drive_find_child(service, parent_id, filename, None)
    media = MediaFileUpload(str(local_path), mimetype="text/plain", resumable=True)

    if existing:
        execute_with_retries(
            service.files().update(fileId=existing["id"], media_body=media, **_list_kwargs())
        )
    else:
        meta = {"name": filename, "parents": [parent_id]}
        execute_with_retries(
            service.files().create(body=meta, media_body=media, fields="id", **_list_kwargs())
        )

def debug_list_root_folders(service, limit=200):
    res = execute_with_retries(
        service.files().list(
            q="('root' in parents) and trashed=false and mimeType='application/vnd.google-apps.folder'",
            fields="files(id,name)",
            pageSize=limit,
            **_list_kwargs(),
        )
    )
    print("\n[DEBUG] Top-level folders in My Drive:")
    files = sorted(res.get("files", []), key=lambda x: x["name"].lower())
    for f in files:
        print(" -", f["name"])

def drive_search_folder_anywhere_in_my_drive(service, folder_name: str) -> List[dict]:
    """
    Search ALL folders matching name. Paginated. Returns list of folder dicts.
    """
    safe_name = _escape_drive_q_value(folder_name)
    q = f"name = '{safe_name}' and mimeType = '{FOLDER_MIME}' and trashed=false"

    out = []
    page_token = None
    while True:
        res = execute_with_retries(
            service.files().list(
                q=q,
                fields="nextPageToken, files(id,name,parents,modifiedTime)",
                pageSize=1000,
                pageToken=page_token,
                **_list_kwargs(),
            )
        )
        out.extend(res.get("files", []))
        page_token = res.get("nextPageToken")
        if not page_token:
            break
    return out

# =========================
# SLOT SELECTION
# =========================
def list_slot_folders(service, slots_parent_id: str) -> List[dict]:
    return sorted(
        list(drive_list_children(service, slots_parent_id, FOLDER_MIME)),
        key=lambda x: x["name"].lower(),
    )

def choose_slot(service, slots_parent_id: str) -> dict:
    slots = list_slot_folders(service, slots_parent_id)
    if not slots:
        raise RuntimeError("No slot folders found under 2026.")

    if SLOT_CHOICE_ENV.isdigit():
        idx = int(SLOT_CHOICE_ENV)
        if 1 <= idx <= len(slots):
            chosen = slots[idx - 1]
            print(f"[AUTO] Using SLOT_CHOICE={idx}: {chosen['name']}")
            return chosen
        raise RuntimeError(f"SLOT_CHOICE '{SLOT_CHOICE_ENV}' is out of range (1..{len(slots)}).")

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

# =========================
# Audio helpers (ffmpeg)
# =========================
def is_video_name(name: str) -> bool:
    p = Path(name)

    if p.name.startswith("."):
        return False

    if p.suffix.lower() not in VIDEO_EXTS:
        return False

    # Skip eye-tracker generated/processed videos
    if "__EYE_" in p.name:
        return False

    return True

def extract_audio_wav(video_path: Path, wav_path: Path):
    proc = subprocess.run(
        ["ffmpeg", "-y", "-i", str(video_path), "-vn", "-ac", "1", "-ar", "16000", str(wav_path)],
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
    )
    if proc.returncode != 0:
        raise RuntimeError(f"ffmpeg failed:\n{proc.stderr}")

def wav_duration_seconds(wav_path: Path) -> float:
    with contextlib.closing(wave.open(str(wav_path), "rb")) as wf:
        return wf.getnframes() / float(wf.getframerate())

def split_audio(wav_path: Path, chunk_seconds: int, overlap_seconds: int):
    total_seconds = wav_duration_seconds(wav_path)
    chunks = []
    idx = 0
    start = 0.0
    while start < total_seconds:
        ss = max(0.0, start - (overlap_seconds if idx > 0 else 0.0))
        duration = chunk_seconds + (overlap_seconds if idx > 0 else 0.0)
        out = wav_path.with_name(f"{wav_path.stem}_part{idx}.wav")

        proc = subprocess.run(
            ["ffmpeg", "-y", "-i", str(wav_path), "-ss", str(ss), "-t", str(duration), str(out)],
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
        )
        if proc.returncode != 0:
            raise RuntimeError(f"ffmpeg split failed:\n{proc.stderr}")

        chunks.append((out, ss, start))  # chunk_path, actual_start, logical_start
        start += chunk_seconds
        idx += 1

    return chunks

def fmt_ts(seconds: float) -> str:
    s = int(math.floor(seconds or 0.0))
    m, s = divmod(s, 60)
    return f"{m}:{s:02d}"

# =========================
# OpenAI helpers
# =========================
def pick_chunks(resp_json: dict):
    return resp_json.get("segments") or resp_json.get("chunks") or resp_json.get("results") or []

def to_timestamp_lines(chunks, include_speaker=False):
    out = []
    for c in chunks or []:
        start = float(c.get("start", 0.0) or 0.0)
        text = (c.get("text") or "").strip()
        speaker = c.get("speaker")
        if not text:
            continue
        prefix = f"{fmt_ts(start)}: "
        if include_speaker and speaker:
            prefix += f"{speaker}: "
        out.append((start, prefix + text))
    return out

def transcribe_diarize_with_http(wav_path: Path, max_retries: int = 6) -> dict:
    headers = {"Authorization": f"Bearer {api_key}"}
    data = {
        "model": MODEL,
        "response_format": "diarized_json",
        "chunking_strategy": CHUNKING_STRATEGY,
    }
    if LANGUAGE:
        data["language"] = LANGUAGE

    for attempt in range(max_retries):
        try:
            with open(wav_path, "rb") as f:
                files = {"file": (wav_path.name, f, "audio/wav")}
                r = requests.post(API_URL, headers=headers, data=data, files=files, timeout=600)
        except requests.RequestException:
            if attempt == max_retries - 1:
                raise
            time.sleep((2 ** attempt) + random.random())
            continue

        if r.status_code == 200:
            return r.json()

        if r.status_code in (429, 500, 502, 503, 504):
            if attempt == max_retries - 1:
                raise RuntimeError(f"HTTP {r.status_code}: {r.text}")
            retry_after = r.headers.get("Retry-After")
            if retry_after and retry_after.isdigit():
                time.sleep(int(retry_after))
            else:
                time.sleep((2 ** attempt) + random.random())
            continue

        raise RuntimeError(f"HTTP {r.status_code}: {r.text}")

# =========================
# MAIN
# =========================
def main():
    # Ensure ffmpeg exists
    try:
        subprocess.run(
            ["ffmpeg", "-version"],
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
            check=True,
        )
    except Exception:
        raise RuntimeError("ffmpeg not found in PATH. Install ffmpeg and restart terminal.")

    service = get_drive_service()

    # Find your "2026" folder ANYWHERE
    candidates = drive_search_folder_anywhere_in_my_drive(service, ROOT_2026_FOLDER_NAME)
    if not candidates:
        debug_list_root_folders(service)
        raise RuntimeError(
            f"Could not find folder '{ROOT_2026_FOLDER_NAME}'.\n"
            "It may be named differently or inside another folder.\n"
            "Check [DEBUG] list above and update ROOT_2026_FOLDER_NAME."
        )

    # If multiple, pick most recently modified (stable default)
    candidates.sort(key=lambda x: x.get("modifiedTime", ""), reverse=True)
    if len(candidates) > 1:
        print(f"\n[WARN] Multiple folders named: {ROOT_2026_FOLDER_NAME}")
        for c in candidates[:10]:
            print(" - id:", c["id"], "modified:", c.get("modifiedTime"), "parents:", c.get("parents"))
        print("[WARN] Using the most recently modified one.\n")

    root_2026_id = candidates[0]["id"]
    slots_parent_id = root_2026_id

    # Pick slot (auto via SLOT_CHOICE or interactive)
    slot = choose_slot(service, slots_parent_id)

    total = done = skipped = failed = 0

    people = sorted(list(drive_list_children(service, slot["id"], FOLDER_MIME)), key=lambda x: x["name"])

    for person in people:
        for folder_name in FOLDER_NAMES_TO_PROCESS:
            target = drive_find_child(service, person["id"], folder_name, FOLDER_MIME)
            if not target:
                continue

            # Upload transcripts directly into the SAME deliverable folder
            out_folder_id = target["id"]

            children = list(drive_list_children(service, target["id"], None))
            videos = sorted(
                [f for f in children if f.get("mimeType") != FOLDER_MIME and is_video_name(f["name"])],
                key=lambda x: x["name"],
            )
            if not videos:
                continue

            print("\n" + "=" * 100)
            print("Slot  :", slot["name"])
            print("Person:", person["name"])
            print("Folder:", folder_name)
            print("Videos:", len(videos))

            for vid in videos:
                total += 1
                out_txt_name = f"{Path(vid['name']).stem}.txt"

                existing_out = drive_find_child(service, out_folder_id, out_txt_name, None)
                if existing_out and not FORCE_RETRANSCRIBE:
                    skipped += 1
                    print(f"  [SKIP] {vid['name']} -> transcript exists")
                    continue

                try:
                    with tempfile.TemporaryDirectory() as td:
                        td = Path(td)
                        local_video = td / vid["name"]
                        local_wav = td / f"{Path(vid['name']).stem}__tmp.wav"
                        local_txt = td / out_txt_name

                        print(f"  [DL  ] {vid['name']} -> downloading")
                        drive_download_file(service, vid["id"], local_video)

                        print("  [RUN ] extracting audio")
                        extract_audio_wav(local_video, local_wav)

                        dur = wav_duration_seconds(local_wav)
                        print(f"  [INFO] Audio duration: {dur/60:.1f} min")

                        parts = split_audio(local_wav, CHUNK_SECONDS, OVERLAP_SECONDS)

                        merged = []
                        for i, (chunk_path, actual_ss, logical_start) in enumerate(parts):
                            try:
                                print(f"    [CHUNK] part{i} ss={actual_ss:.1f}s (logical={logical_start:.1f}s)")
                                resp = transcribe_diarize_with_http(chunk_path)
                                lines = to_timestamp_lines(pick_chunks(resp), include_speaker=INCLUDE_SPEAKER)

                                for rel_start, line in lines:
                                    abs_start = rel_start + actual_ss
                                    # Drop overlap lines that belong to the previous chunk
                                    if i > 0 and abs_start < logical_start:
                                        continue
                                    merged.append(
                                        (
                                            abs_start,
                                            line.replace(f"{fmt_ts(rel_start)}:", f"{fmt_ts(abs_start)}:", 1),
                                        )
                                    )
                            finally:
                                chunk_path.unlink(missing_ok=True)

                        merged.sort(key=lambda x: x[0])

                        # Light dedupe (0.5s buckets)
                        final_lines = []
                        seen = set()
                        for abs_start, line in merged:
                            bucket = int(abs_start * 2)  # 0.5-second buckets
                            tail = line.split(":", 1)[-1].strip().lower()
                            key = (bucket, tail)
                            if key in seen:
                                continue
                            seen.add(key)
                            final_lines.append(line)

                        local_txt.write_text("\n".join(final_lines).strip() + "\n", encoding="utf-8")

                        print(f"  [UP  ] uploading -> {slot['name']}/{person['name']}/{folder_name}/{out_txt_name}")
                        drive_upload_text(service, out_folder_id, out_txt_name, local_txt)

                        done += 1
                        print("  [OK  ] done")

                except Exception as e:
                    failed += 1
                    print(f"  [FAIL] {vid['name']} -> {type(e).__name__}: {e}")

    print("\nSUMMARY")
    print("Total:", total, "Done:", done, "Skipped:", skipped, "Failed:", failed)
    print("Output root folder:", ROOT_2026_FOLDER_NAME)

if __name__ == "__main__":
    main()
