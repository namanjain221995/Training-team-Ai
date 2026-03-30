#!/usr/bin/env python3

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
import ssl
import shutil
import socket
from pathlib import Path
from typing import Optional, List, Tuple, Generator

from dotenv import load_dotenv
from google.oauth2 import service_account
from googleapiclient.discovery import build
from googleapiclient.http import MediaIoBaseDownload, MediaFileUpload
from googleapiclient.errors import HttpError

load_dotenv()

SERVICE_ACCOUNT_FILE = Path(os.getenv("SERVICE_ACCOUNT_FILE") or "service-account.json")
SCOPES = ["https://www.googleapis.com/auth/drive"]

ROOT_2026_FOLDER_NAME = (os.getenv("ROOT_2026_FOLDER_NAME") or "2026").strip()
SHARED_DRIVE_NAME = (os.getenv("SHARED_DRIVE_NAME") or "2026_Shared_Drive").strip()
FOLDER_MIME = "application/vnd.google-apps.folder"

USE_SHARED_DRIVES = (os.getenv("USE_SHARED_DRIVES") or "").strip().lower() in ("1", "true", "yes", "y")
USE_DELEGATION = (os.getenv("USE_DELEGATION") or "").strip().lower() in ("1", "true", "yes", "y")
DELEGATED_USER_EMAIL = (os.getenv("DELEGATED_USER_EMAIL") or "").strip()

SLOT_CHOICE_ENV = (os.getenv("SLOT_CHOICE") or "6").strip()
VIDEO_FILE_ID = (os.getenv("VIDEO_FILE_ID") or "").strip()
TARGET_FOLDER_ID = (os.getenv("TARGET_FOLDER_ID") or "").strip()
VIDEO_NAME_ENV = (os.getenv("VIDEO_NAME") or "").strip()

VIDEO_EXTS = {".mp4", ".mkv", ".mov", ".avi", ".webm", ".m4v"}
TRANSCRIPT_SUFFIX = "_transcripts.txt"

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

CHUNK_SECONDS = int((os.getenv("CHUNK_SECONDS") or "540").strip())
OVERLAP_SECONDS = int((os.getenv("OVERLAP_SECONDS") or "2").strip())
FORCE_RETRANSCRIBE = (os.getenv("FORCE_RETRANSCRIBE") or "").strip().lower() in ("1", "true", "yes", "y")

WHISPER_MODEL = (os.getenv("WHISPER_MODEL") or "large-v3").strip()
DEVICE = (os.getenv("DEVICE") or "cpu").strip()
COMPUTE_TYPE = (os.getenv("COMPUTE_TYPE") or ("float16" if DEVICE == "cuda" else "int8")).strip()
LANGUAGE = (os.getenv("LANGUAGE") or "en").strip()
USE_VAD = (os.getenv("USE_VAD") or "true").strip().lower() in ("1", "true", "yes", "y")

MAX_RETRIES = int((os.getenv("MAX_RETRIES") or "8").strip())
BASE_SLEEP = float((os.getenv("BASE_SLEEP") or "1.0").strip())

_WINDOWS_FORBIDDEN = r'<>:"/\\|?*'
_WINDOWS_RESERVED = {
    "CON", "PRN", "AUX", "NUL",
    *(f"COM{i}" for i in range(1, 10)),
    *(f"LPT{i}" for i in range(1, 10)),
}

_SLOT_PREFIX_RE = re.compile(r"^\s*(\d+)\.\s*")


def safe_local_filename(name: str, fallback: str = "file.bin") -> str:
    name = (name or "").strip() or fallback
    name = re.sub(f"[{re.escape(_WINDOWS_FORBIDDEN)}]", "_", name)
    name = re.sub(r"\s+", " ", name).strip()
    name = name.rstrip(" .")

    if "." in name:
        stem, ext = name.rsplit(".", 1)
        if stem.upper() in _WINDOWS_RESERVED:
            stem = f"_{stem}"
        name = f"{stem}.{ext}"
    else:
        if name.upper() in _WINDOWS_RESERVED:
            name = f"_{name}"

    return name or fallback


def transcript_output_name(video_name: str) -> str:
    return f"{Path(video_name).stem}{TRANSCRIPT_SUFFIX}"


def validate_env():
    if not SERVICE_ACCOUNT_FILE.exists():
        raise FileNotFoundError(
            f"\n❌ Service account file not found: {SERVICE_ACCOUNT_FILE}\n"
            "Setup instructions:\n"
            "1. Download service-account.json from Google Cloud Console\n"
            "2. Place it in the same directory as this script\n"
            "3. Or set SERVICE_ACCOUNT_FILE to the full path\n"
        )

    if not USE_SHARED_DRIVES and not USE_DELEGATION:
        raise RuntimeError(
            "⚠️  WARNING: Neither USE_SHARED_DRIVES nor USE_DELEGATION is enabled!\n"
            "Raw Service Accounts cannot upload to My Drive.\n\n"
            "CHOOSE ONE:\n"
            "  Option 1 (Recommended): USE_SHARED_DRIVES=true\n"
            "    - Shared Drive must exist and service account must be a member\n\n"
            "  Option 2: USE_DELEGATION=true + DELEGATED_USER_EMAIL=<user@domain.com>\n"
            "    - Workspace admin must authorize domain-wide delegation\n"
        )

    if USE_DELEGATION and not DELEGATED_USER_EMAIL:
        raise RuntimeError(
            "USE_DELEGATION=true but DELEGATED_USER_EMAIL is not set.\n"
            "Example:\n"
            "DELEGATED_USER_EMAIL=techsphere@techsarasolutions.com"
        )


def get_drive_service(user_email: Optional[str] = None):
    print("\n" + "=" * 80)
    print("AUTHENTICATION")
    print("=" * 80)

    creds = service_account.Credentials.from_service_account_file(
        str(SERVICE_ACCOUNT_FILE),
        scopes=SCOPES,
    )

    print(f"[INFO] Service Account Email: {creds.service_account_email}")

    if user_email:
        creds = creds.with_subject(user_email)
        print("[INFO] Mode: Domain-Wide Delegation")
        print(f"[INFO] Impersonating User: {user_email}")
    else:
        print("[INFO] Mode: Raw Service Account")

    if USE_SHARED_DRIVES:
        print(f"[INFO] Using Shared Drive: {SHARED_DRIVE_NAME}")
        print("[INFO] Service account MUST be a member of this Shared Drive")
    else:
        print("[INFO] Using My Drive (via delegation)")

    service = build("drive", "v3", credentials=creds)
    print("[✓] Drive authentication successful\n")
    return service


def is_storage_quota_error(e: Exception) -> bool:
    if isinstance(e, HttpError):
        msg = str(e).lower()
        return ("storage quota" in msg) or ("storagequotaexceeded" in msg)
    return False


def is_transient_error(e: Exception) -> bool:
    if isinstance(e, HttpError):
        status = getattr(e.resp, "status", None)
        if status in (429, 500, 502, 503, 504):
            return True
        if status == 403:
            msg = str(e).lower()
            if any(x in msg for x in ("rate limit", "quota exceeded", "user rate limit", "temporarily")):
                return True
        return False

    if isinstance(e, (ssl.SSLEOFError, ssl.SSLError, TimeoutError, ConnectionError)):
        return True

    if isinstance(e, socket.timeout):
        return True

    if isinstance(e, OSError):
        msg = str(e).lower()
        if any(x in msg for x in (
            "connection reset",
            "broken pipe",
            "connection aborted",
            "timed out",
            "temporary failure",
        )):
            return True

    msg = str(e).lower()
    if any(x in msg for x in (
        "connection reset",
        "eof occurred",
        "broken pipe",
        "connection aborted",
        "timed out",
        "ssl",
        "temporary failure",
    )):
        return True

    return False


def execute_with_retries(request, *, max_retries: int = MAX_RETRIES, base_sleep: float = BASE_SLEEP):
    for attempt in range(max_retries):
        try:
            return request.execute()
        except Exception as e:
            if is_storage_quota_error(e):
                print("\n[ERROR] Storage Quota Error!")
                print("[ERROR] This means uploads are going to My Drive instead of Shared Drive")
                print("[ERROR] Fix: Ensure 2026 folder is INSIDE the Shared Drive, not in My Drive")
                print("[ERROR] Also verify: USE_SHARED_DRIVES=true in .env")
                raise

            if is_transient_error(e):
                if attempt == max_retries - 1:
                    print(f"[ERROR] Max retries reached ({max_retries}).")
                    raise

                sleep = (base_sleep * (2 ** attempt)) + random.random()
                print(f"[WARN] Transient error: {type(e).__name__}. Retrying in {sleep:.1f}s... "
                      f"(Attempt {attempt + 1}/{max_retries})")
                time.sleep(sleep)
                continue

            raise


class SafeTemporaryDirectory:
    def __init__(self):
        self.path = Path(tempfile.mkdtemp())
        print(f"[INFO] Created temp directory: {self.path}")

    def cleanup(self):
        if not self.path.exists():
            return

        try:
            shutil.rmtree(str(self.path))
            print("[INFO] Cleaned up temp directory")
        except PermissionError:
            print("[WARN] Files locked in temp directory, retrying...")
            time.sleep(0.5)
            try:
                shutil.rmtree(str(self.path), ignore_errors=True)
                print("[INFO] Cleaned up temp directory (forced)")
            except Exception:
                print(f"[WARN] Could not fully clean temp directory: {self.path}")

    def __enter__(self):
        return self.path

    def __exit__(self, exc_type, exc, tb):
        self.cleanup()


def _escape_drive_q_value(s: str) -> str:
    return s.replace("\\", "\\\\").replace("'", "\\'")


def _list_kwargs():
    return {
        "supportsAllDrives": True,
        "includeItemsFromAllDrives": True,
        "corpora": "allDrives",
    }


def _read_kwargs():
    return {"supportsAllDrives": True}


def _write_kwargs():
    return {"supportsAllDrives": True}


def drive_list_children(service, parent_id: str, mime_type: Optional[str] = None) -> Generator[dict, None, None]:
    q_parts = [f"'{parent_id}' in parents", "trashed = false"]
    if mime_type:
        q_parts.append(f"mimeType = '{_escape_drive_q_value(mime_type)}'")
    q = " and ".join(q_parts)

    page_token = None
    while True:
        res = execute_with_retries(
            service.files().list(
                q=q,
                fields="nextPageToken, files(id,name,mimeType,size,modifiedTime,parents)",
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
    q_parts = [
        f"'{parent_id}' in parents",
        f"name = '{_escape_drive_q_value(name)}'",
        "trashed = false",
    ]
    if mime_type:
        q_parts.append(f"mimeType = '{_escape_drive_q_value(mime_type)}'")
    q = " and ".join(q_parts)

    res = execute_with_retries(
        service.files().list(
            q=q,
            fields="files(id,name,mimeType,size,modifiedTime,parents)",
            pageSize=10,
            **_list_kwargs(),
        )
    )
    files = res.get("files", [])
    return files[0] if files else None


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
        chunk_attempt = 0

        while not done:
            try:
                status, done = downloader.next_chunk()
                chunk_attempt = 0
            except Exception as e:
                if is_transient_error(e):
                    chunk_attempt += 1
                    if chunk_attempt > 3:
                        raise
                    sleep = (BASE_SLEEP * (2 ** chunk_attempt)) + random.random()
                    print(f"[WARN] Download error, retrying in {sleep:.1f}s...")
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
            service.files().update(
                fileId=existing["id"],
                media_body=media,
                **_write_kwargs(),
            )
        )
        print(f"[UP] Updated: {filename}")
    else:
        meta = {"name": filename, "parents": [parent_id]}
        execute_with_retries(
            service.files().create(
                body=meta,
                media_body=media,
                fields="id",
                **_write_kwargs(),
            )
        )
        print(f"[UP] Uploaded: {filename}")


def find_shared_drive(service, drive_name: str) -> Optional[dict]:
    page_token = None
    while True:
        res = execute_with_retries(
            service.drives().list(
                q=f"name = '{_escape_drive_q_value(drive_name)}'",
                fields="nextPageToken, drives(id,name)",
                pageSize=100,
                pageToken=page_token,
            )
        )
        for drive in res.get("drives", []):
            return drive

        page_token = res.get("nextPageToken")
        if not page_token:
            break

    return None


def drive_search_folder_anywhere(service, folder_name: str) -> List[dict]:
    safe_name = _escape_drive_q_value(folder_name)
    q = f"name = '{safe_name}' and mimeType = '{FOLDER_MIME}' and trashed = false"

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


def extract_slot_prefix(name: str) -> Optional[int]:
    m = _SLOT_PREFIX_RE.match(name or "")
    return int(m.group(1)) if m else None


def list_slot_folders(service, slots_parent_id: str) -> List[dict]:
    folders = list(drive_list_children(service, slots_parent_id, FOLDER_MIME))
    out = []

    for f in folders:
        slot_no = extract_slot_prefix(f["name"])
        if slot_no is not None:
            item = dict(f)
            item["_slot_no"] = slot_no
            out.append(item)

    return sorted(out, key=lambda x: (x["_slot_no"], x["name"].lower()))


def choose_slot(service, slots_parent_id: str) -> dict:
    slots = list_slot_folders(service, slots_parent_id)
    if not slots:
        raise RuntimeError("No numbered slot folders found under the root folder.")

    if SLOT_CHOICE_ENV.isdigit():
        wanted_slot_no = int(SLOT_CHOICE_ENV)
        for s in slots:
            if s["_slot_no"] == wanted_slot_no:
                print(f"[AUTO] Using SLOT_CHOICE={wanted_slot_no}: {s['name']}")
                return s

        available = ", ".join(str(s["_slot_no"]) for s in slots)
        raise RuntimeError(
            f"SLOT_CHOICE '{wanted_slot_no}' not found. "
            f"Available folder numbers: {available}"
        )

    print("\nAvailable slots:")
    for s in slots:
        print(f"  {s['_slot_no']:2}. {s['name']}")

    while True:
        choice = input("Choose slot number (e.g. 8 or 9) or EXIT: ").strip().lower()
        if choice == "exit":
            raise SystemExit(0)

        if choice.isdigit():
            wanted_slot_no = int(choice)
            for s in slots:
                if s["_slot_no"] == wanted_slot_no:
                    return s

        print("Invalid choice. Try again.")


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
        raise RuntimeError(f"ffmpeg failed while extracting audio:\n{proc.stderr}")


def wav_duration_seconds(wav_path: Path) -> float:
    with contextlib.closing(wave.open(str(wav_path), "rb")) as wf:
        return wf.getnframes() / float(wf.getframerate())


def split_audio(wav_path: Path, chunk_seconds: int, overlap_seconds: int) -> List[Tuple[Path, float, float]]:
    total_seconds = wav_duration_seconds(wav_path)
    chunks: List[Tuple[Path, float, float]] = []

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


def get_whisper_model():
    from faster_whisper import WhisperModel

    try:
        return WhisperModel(WHISPER_MODEL, device=DEVICE, compute_type=COMPUTE_TYPE)
    except RuntimeError as e:
        if "cuda" in str(e).lower():
            print("[WARN] CUDA load failed. Falling back to CPU int8.")
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


def process_one_video(service, whisper_model, video_file_id: str, target_folder_id: str, video_name: str = ""):
    print("\n" + "=" * 80)

    meta = drive_get_file_meta(service, video_file_id)
    video_name = video_name or meta["name"]
    out_txt_name = transcript_output_name(video_name)

    existing_out = drive_find_child(service, target_folder_id, out_txt_name, None)
    if existing_out and not FORCE_RETRANSCRIBE:
        print(f"[SKIP] Transcript exists: {out_txt_name}")
        print("=" * 80)
        return

    with SafeTemporaryDirectory() as td:
        td = Path(td)

        safe_vid_name = safe_local_filename(video_name, "video.mp4")
        safe_txt_name = safe_local_filename(out_txt_name, "transcript.txt")

        local_video = td / safe_vid_name
        local_wav = td / f"{Path(safe_vid_name).stem}__tmp.wav"
        local_txt = td / safe_txt_name

        print(f"[DL] Downloading: {video_name}")
        drive_download_file(service, video_file_id, local_video)

        print("[RUN] Extracting audio...")
        extract_audio_wav(local_video, local_wav)

        dur = wav_duration_seconds(local_wav)
        print(f"[INFO] Audio duration: {dur / 60:.1f} minutes")

        if dur <= (CHUNK_SECONDS + 5):
            parts = [(local_wav, 0.0, 0.0)]
        else:
            parts = split_audio(local_wav, CHUNK_SECONDS, OVERLAP_SECONDS)

        print(f"[INFO] Processing in {len(parts)} part(s)")

        merged: List[Tuple[float, str]] = []

        for i, (chunk_path, actual_ss, logical_start) in enumerate(parts):
            print(f"[TRANSCRIBE] Part {i + 1}/{len(parts)}...")
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

        print(f"[UP] Uploading transcript: {out_txt_name}")
        drive_upload_text(service, target_folder_id, out_txt_name, local_txt)

        print("[OK] ✓ Transcription complete!")
        print("=" * 80)


def resolve_service():
    delegated_user = DELEGATED_USER_EMAIL if USE_DELEGATION else None
    return get_drive_service(delegated_user)


def main_full_scan():
    service = resolve_service()
    whisper_model = get_whisper_model()

    print("\n" + "=" * 80)
    print("FULL SCAN MODE - PROCESSING ALL VIDEOS")
    print("=" * 80)

    if USE_SHARED_DRIVES:
        print(f"\n[SEARCH] Looking for Shared Drive: '{SHARED_DRIVE_NAME}'")
        shared_drive = find_shared_drive(service, SHARED_DRIVE_NAME)
        if not shared_drive:
            raise RuntimeError(
                f"Could not find Shared Drive '{SHARED_DRIVE_NAME}'\n"
                f"Make sure:\n"
                f"  1. The Shared Drive exists\n"
                f"  2. The service account is a member\n"
                f"  3. SHARED_DRIVE_NAME matches exactly\n"
            )
        root_id = shared_drive["id"]
        print(f"[FOUND] Shared Drive: {shared_drive['name']} ({root_id})")
    else:
        print(f"\n[SEARCH] Looking for root folder: '{ROOT_2026_FOLDER_NAME}'")
        candidates = drive_search_folder_anywhere(service, ROOT_2026_FOLDER_NAME)
        if not candidates:
            raise RuntimeError(f"Could not find folder '{ROOT_2026_FOLDER_NAME}'")
        candidates.sort(key=lambda x: x.get("modifiedTime", ""), reverse=True)
        root_folder = candidates[0]
        root_id = root_folder["id"]
        print(f"[FOUND] Root folder: {root_folder['name']} ({root_id})")

    root_2026 = drive_find_child(service, root_id, ROOT_2026_FOLDER_NAME, FOLDER_MIME)
    if not root_2026:
        raise RuntimeError(
            f"Could not find '{ROOT_2026_FOLDER_NAME}' folder inside the root\n"
            f"Make sure the folder structure exists in the Shared Drive"
        )
    root_2026_id = root_2026["id"]
    print(f"[FOUND] 2026 folder: {root_2026['name']} ({root_2026_id})")

    slot = choose_slot(service, root_2026_id)
    print(f"[SELECTED] Slot: {slot['name']}")

    people = sorted(
        list(drive_list_children(service, slot["id"], FOLDER_MIME)),
        key=lambda x: x["name"].lower(),
    )
    print(f"[INFO] Found {len(people)} candidate folder(s)\n")

    for person in people:
        print(f"\n{'=' * 80}")
        print(f"PROCESSING CANDIDATE: {person['name']}")
        print(f"{'=' * 80}")

        for folder_name in FOLDER_NAMES_TO_PROCESS:
            target = drive_find_child(service, person["id"], folder_name, FOLDER_MIME)
            if not target:
                continue

            print(f"\n[FOLDER] {folder_name}")
            folder_id = target["id"]

            children = list(drive_list_children(service, folder_id, None))
            videos = [
                f for f in children
                if f.get("mimeType") != FOLDER_MIME and is_video_name(f["name"])
            ]

            if not videos:
                print("[INFO] No videos found")
                continue

            print(f"[INFO] Found {len(videos)} video(s)")

            for vid in videos:
                process_one_video(service, whisper_model, vid["id"], folder_id, vid["name"])


def main():
    validate_env()

    print("\n" + "╔" + "=" * 78 + "╗")
    print("║" + " " * 78 + "║")
    print("║" + "  FULLY FIXED - Transcription with Shared Drive Support".center(78) + "║")
    print("║" + "  Service Account + Shared Drive Uploads".center(78) + "║")
    print("║" + "  TechSara Solutions".center(78) + "║")
    print("║" + " " * 78 + "║")
    print("╚" + "=" * 78 + "╝")

    try:
        subprocess.run(
            ["ffmpeg", "-version"],
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
            check=True,
        )
    except Exception:
        raise RuntimeError("ffmpeg is not installed or not available in PATH.")

    if VIDEO_FILE_ID and TARGET_FOLDER_ID:
        print("\n[MODE] Single-Video Worker Mode")
        service = resolve_service()
        whisper_model = get_whisper_model()
        process_one_video(service, whisper_model, VIDEO_FILE_ID, TARGET_FOLDER_ID, VIDEO_NAME_ENV)
        return

    main_full_scan()

    print("\n" + "=" * 80)
    print("✓ ALL TRANSCRIPTIONS COMPLETE!")
    print("=" * 80 + "\n")


if __name__ == "__main__":
    main()