import os
import io
import math
import time
import random
import tempfile
import subprocess
import contextlib
import wave
import re
from pathlib import Path
from typing import Optional, List, Tuple

from dotenv import load_dotenv

from googleapiclient.discovery import build
from googleapiclient.http import MediaIoBaseDownload, MediaFileUpload
from googleapiclient.errors import HttpError
from google.oauth2.credentials import Credentials
from google_auth_oauthlib.flow import InstalledAppFlow
from google.auth.transport.requests import Request

load_dotenv()

# ========================
# CONFIG
# =========================
ROOT_2026_FOLDER_NAME = "2026"
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

TRANSCRIPT_SUFFIX = "_transcripts.txt"

CHUNK_SECONDS = int((os.getenv("CHUNK_SECONDS") or "540").strip())
OVERLAP_SECONDS = int((os.getenv("OVERLAP_SECONDS") or "2").strip())
FORCE_RETRANSCRIBE = (os.getenv("FORCE_RETRANSCRIBE") or "").strip().lower() in ("1", "true", "yes", "y")

# IMPORTANT: default to small for t2.micro safety (can override via env)
WHISPER_MODEL = (os.getenv("WHISPER_MODEL") or "medium").strip()
DEVICE = (os.getenv("DEVICE") or "cpu").strip()
COMPUTE_TYPE = (os.getenv("COMPUTE_TYPE") or ("float16" if DEVICE == "cuda" else "int8")).strip()
LANGUAGE = (os.getenv("LANGUAGE") or "en").strip()
USE_VAD = (os.getenv("USE_VAD") or "true").strip().lower() in ("1", "true", "yes", "y")

SCOPES = ["https://www.googleapis.com/auth/drive"]
FOLDER_MIME = "application/vnd.google-apps.folder"
USE_SHARED_DRIVES = (os.getenv("USE_SHARED_DRIVES") or "").strip().lower() in ("1", "true", "yes", "y")
SLOT_CHOICE_ENV = (os.getenv("SLOT_CHOICE") or "4").strip()

HEADLESS_AUTH = (os.getenv("HEADLESS_AUTH") or "").strip().lower() in ("1", "true", "yes", "y")
CREDENTIALS_FILE = Path(os.getenv("CREDENTIALS_FILE") or "credentials.json")
TOKEN_FILE = Path(os.getenv("TOKEN_FILE") or "token.json")

VIDEO_FILE_ID = (os.getenv("VIDEO_FILE_ID") or "").strip()
TARGET_FOLDER_ID = (os.getenv("TARGET_FOLDER_ID") or "").strip()
VIDEO_NAME_ENV = (os.getenv("VIDEO_NAME") or "").strip()

# =========================
# WINDOWS-SAFE FILENAMES
# =========================
_WINDOWS_FORBIDDEN = r'<>:"/\\|?*'
_WINDOWS_RESERVED = {
    "CON", "PRN", "AUX", "NUL",
    *(f"COM{i}" for i in range(1, 10)),
    *(f"LPT{i}" for i in range(1, 10)),
}

def safe_local_filename(name: str, fallback: str = "file.bin") -> str:
    name = (name or "").strip() or fallback
    name = re.sub(f"[{re.escape(_WINDOWS_FORBIDDEN)}]", "_", name)
    name = re.sub(r"\s+", " ", name).strip()
    name = name.rstrip(" .")
    stem, dot, ext = name.partition(".")
    if stem.upper() in _WINDOWS_RESERVED:
        stem = f"_{stem}"
    name = stem + (dot + ext if dot else "")
    return name or fallback

def transcript_output_name(video_name: str) -> str:
    return f"{Path(video_name).stem}{TRANSCRIPT_SUFFIX}"

# =========================
# RETRY WRAPPER
# =========================
def execute_with_retries(request, *, max_retries: int = 8, base_sleep: float = 1.0):
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
# DRIVE AUTH (headless-safe)
# =========================
def get_drive_service():
    creds = None
    if TOKEN_FILE.exists() and TOKEN_FILE.is_file():
        creds = Credentials.from_authorized_user_file(str(TOKEN_FILE), SCOPES)

    if not creds or not creds.valid:
        if creds and creds.expired and creds.refresh_token:
            creds.refresh(Request())
            try:
                TOKEN_FILE.parent.mkdir(parents=True, exist_ok=True)
                TOKEN_FILE.write_text(creds.to_json(), encoding="utf-8")
            except Exception:
                pass
        else:
            if not CREDENTIALS_FILE.exists():
                raise FileNotFoundError(f"{CREDENTIALS_FILE} not found.")
            if HEADLESS_AUTH:
                # Workers / CI should never do interactive auth
                raise RuntimeError(
                    "Drive token is missing/invalid in this environment. "
                    "Update /transcription/worker/token_json in SSM Parameter Store with a valid token.json."
                )

            flow = InstalledAppFlow.from_client_secrets_file(str(CREDENTIALS_FILE), SCOPES)
            creds = flow.run_local_server(port=0)
            TOKEN_FILE.parent.mkdir(parents=True, exist_ok=True)
            TOKEN_FILE.write_text(creds.to_json(), encoding="utf-8")

    return build("drive", "v3", credentials=creds)

def _list_kwargs():
    if USE_SHARED_DRIVES:
        return {"supportsAllDrives": True, "includeItemsFromAllDrives": True, "corpora": "allDrives"}
    return {}

def _read_kwargs():
    if USE_SHARED_DRIVES:
        return {"supportsAllDrives": True}
    return {}

def _write_kwargs():
    if USE_SHARED_DRIVES:
        return {"supportsAllDrives": True}
    return {}

# =========================
# DRIVE HELPERS
# =========================
def _escape_drive_q_value(s: str) -> str:
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
    for f in drive_list_children(service, parent_id, mime_type):
        if f.get("name") == name:
            return f
    return None

def drive_get_file_meta(service, file_id: str):
    return execute_with_retries(
        service.files().get(
            fileId=file_id,
            fields="id,name,mimeType,parents,modifiedTime",
            **_read_kwargs(),
        )
    )

def drive_download_file(service, file_id: str, dest_path: Path):
    request = service.files().get_media(fileId=file_id, **_read_kwargs())
    with io.FileIO(str(dest_path), "wb") as fh:
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
        execute_with_retries(service.files().update(fileId=existing["id"], media_body=media, **_write_kwargs()))
    else:
        meta = {"name": filename, "parents": [parent_id]}
        execute_with_retries(service.files().create(body=meta, media_body=media, fields="id", **_write_kwargs()))

def drive_search_folder_anywhere_in_my_drive(service, folder_name: str) -> List[dict]:
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
    return sorted(list(drive_list_children(service, slots_parent_id, FOLDER_MIME)), key=lambda x: x["name"].lower())

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

    for i, s in enumerate(slots, start=1):
        print(f"  {i:2}. {s['name']}")
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
# AUDIO HELPERS
# =========================
def is_video_name(name: str) -> bool:
    p = Path(name)
    if p.name.startswith("."):
        return False
    if p.suffix.lower() not in VIDEO_EXTS:
        return False
    if "__EYE_" in p.name:
        return False
    return True

def extract_audio_wav(video_path: Path, wav_path: Path):
    proc = subprocess.run(
        ["ffmpeg", "-y", "-loglevel", "error", "-i", str(video_path), "-vn", "-ac", "1", "-ar", "16000", str(wav_path)],
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
            ["ffmpeg", "-y", "-loglevel", "error", "-i", str(wav_path), "-ss", str(ss), "-t", str(duration), str(out)],
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
        )
        if proc.returncode != 0:
            raise RuntimeError(f"ffmpeg split failed:\n{proc.stderr}")

        chunks.append((out, ss, start))
        start += chunk_seconds
        idx += 1
    return chunks

def fmt_ts(seconds: float) -> str:
    s = int(math.floor(seconds or 0.0))
    m, s = divmod(s, 60)
    return f"{m}:{s:02d}"

# =========================
# WHISPER
# =========================
def get_whisper_model():
    from faster_whisper import WhisperModel
    try:
        return WhisperModel(WHISPER_MODEL, device=DEVICE, compute_type=COMPUTE_TYPE)
    except RuntimeError as e:
        if "cuda" in str(e).lower():
            print(f"[WARN] CUDA failed ({e}). Falling back to CPU int8.")
            return WhisperModel(WHISPER_MODEL, device="cpu", compute_type="int8")
        raise

def transcribe_with_faster_whisper(model, wav_path: Path) -> List[Tuple[float, str]]:
    lang = LANGUAGE if LANGUAGE else None
    segments, _info = model.transcribe(
        str(wav_path),
        language=lang,
        vad_filter=USE_VAD,
        word_timestamps=False,
        beam_size=5,
    )
    out: List[Tuple[float, str]] = []
    for seg in segments:
        start = float(seg.start or 0.0)
        text = (seg.text or "").strip()
        if text:
            out.append((start, text))
    return out

# =========================
# PROCESS ONE VIDEO
# =========================
def process_one_video(service, whisper_model, video_file_id: str, target_folder_id: str, video_name: str = ""):
    meta = drive_get_file_meta(service, video_file_id)
    video_name = video_name or meta["name"]
    out_txt_name = transcript_output_name(video_name)

    existing_out = drive_find_child(service, target_folder_id, out_txt_name, None)
    if existing_out and not FORCE_RETRANSCRIBE:
        print(f"[SKIP] transcript exists: {out_txt_name}")
        return

    with tempfile.TemporaryDirectory() as td:
        td = Path(td)
        safe_vid_name = safe_local_filename(video_name, "video.mp4")
        safe_txt_name = safe_local_filename(out_txt_name, out_txt_name)

        local_video = td / safe_vid_name
        local_wav = td / f"{Path(safe_vid_name).stem}__tmp.wav"
        local_txt = td / safe_txt_name

        print(f"[DL] {video_name}")
        drive_download_file(service, video_file_id, local_video)

        print("[RUN] extract audio")
        extract_audio_wav(local_video, local_wav)

        dur = wav_duration_seconds(local_wav)
        print(f"[INFO] Audio duration: {dur/60:.1f} min")

        parts = [(local_wav, 0.0, 0.0)] if dur <= (CHUNK_SECONDS + 5) else split_audio(local_wav, CHUNK_SECONDS, OVERLAP_SECONDS)

        merged: List[Tuple[float, str]] = []
        for i, (chunk_path, actual_ss, logical_start) in enumerate(parts):
            segs = transcribe_with_faster_whisper(whisper_model, chunk_path)
            for rel_start, text in segs:
                abs_start = rel_start + actual_ss
                if i > 0 and abs_start < logical_start:
                    continue
                merged.append((abs_start, text))
            if chunk_path != local_wav:
                chunk_path.unlink(missing_ok=True)

        merged.sort(key=lambda x: x[0])
        lines = [f"{fmt_ts(t)}: {txt}" for (t, txt) in merged]
        local_txt.write_text("\n".join(lines).strip() + "\n", encoding="utf-8")

        print(f"[UP] uploading transcript -> {out_txt_name}")
        drive_upload_text(service, target_folder_id, out_txt_name, local_txt)
        print("[OK] done")

# =========================
# FULL SCAN MODE
# =========================
def main_full_scan():
    service = get_drive_service()
    whisper_model = get_whisper_model()

    candidates = drive_search_folder_anywhere_in_my_drive(service, ROOT_2026_FOLDER_NAME)
    if not candidates:
        raise RuntimeError(f"Could not find folder '{ROOT_2026_FOLDER_NAME}'")

    candidates.sort(key=lambda x: x.get("modifiedTime", ""), reverse=True)
    root_2026_id = candidates[0]["id"]
    slot = choose_slot(service, root_2026_id)

    people = sorted(list(drive_list_children(service, slot["id"], FOLDER_MIME)), key=lambda x: x["name"])
    for person in people:
        for folder_name in FOLDER_NAMES_TO_PROCESS:
            target = drive_find_child(service, person["id"], folder_name, FOLDER_MIME)
            if not target:
                continue
            folder_id = target["id"]
            children = list(drive_list_children(service, folder_id, None))
            videos = [f for f in children if f.get("mimeType") != FOLDER_MIME and is_video_name(f["name"])]
            for vid in videos:
                process_one_video(service, whisper_model, vid["id"], folder_id, vid["name"])

# =========================
# MAIN
# =========================
def main():
    subprocess.run(["ffmpeg", "-version"], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, check=True)

    # Validate single-video env vars
    if (VIDEO_FILE_ID and not TARGET_FOLDER_ID) or (TARGET_FOLDER_ID and not VIDEO_FILE_ID):
        raise RuntimeError("Both VIDEO_FILE_ID and TARGET_FOLDER_ID must be set for single-video mode.")

    if VIDEO_FILE_ID and TARGET_FOLDER_ID:
        print("[MODE] Single-video worker mode")
        print(f"[INFO] TOKEN_FILE={TOKEN_FILE} exists={TOKEN_FILE.exists()} is_file={TOKEN_FILE.is_file()}")
        print(f"[INFO] CREDENTIALS_FILE={CREDENTIALS_FILE} exists={CREDENTIALS_FILE.exists()} is_file={CREDENTIALS_FILE.is_file()}")

        service = get_drive_service()
        whisper_model = get_whisper_model()
        process_one_video(service, whisper_model, VIDEO_FILE_ID, TARGET_FOLDER_ID, VIDEO_NAME_ENV)
        return

    main_full_scan()

if __name__ == "__main__":
    main()