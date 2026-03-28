import os
import io
import re
import time
import random
from pathlib import Path
from typing import Optional, List

from dotenv import load_dotenv

from googleapiclient.discovery import build
from googleapiclient.http import MediaIoBaseDownload, MediaIoBaseUpload
from googleapiclient.errors import HttpError
from google.oauth2 import service_account

# =========================
# ENV
# =========================
load_dotenv()

SLOT_CHOICE = (os.getenv("SLOT_CHOICE") or "").strip()

SKIP_PERSON_FOLDERS = {
    x.strip() for x in (os.getenv("SKIP_PERSON_FOLDERS") or "1. Format").split(",") if x.strip()
}

AUTH_MODE = (os.getenv("AUTH_MODE") or "service_account").strip().lower()
SERVICE_ACCOUNT_FILE = Path((os.getenv("SERVICE_ACCOUNT_FILE") or "service-account.json").strip())
USE_DELEGATION = (os.getenv("USE_DELEGATION") or "").strip().lower() in ("1", "true", "yes", "y")
DELEGATED_USER_EMAIL = (os.getenv("DELEGATED_USER_EMAIL") or "").strip()

SHARED_DRIVE_NAME = (os.getenv("SHARED_DRIVE_NAME") or "").strip()
ROOT_2026_FOLDER_NAME = (os.getenv("ROOT_2026_FOLDER_NAME") or "2026").strip()

# Optional: exact root folder id inside selected Shared Drive
ROOT_2026_FOLDER_ID = (os.getenv("ROOT_2026_FOLDER_ID") or "").strip()

# =========================
# CONFIG
# =========================
SCOPES = ["https://www.googleapis.com/auth/drive"]

FOLDER_MIME = "application/vnd.google-apps.folder"
GDOC_MIME = "application/vnd.google-apps.document"

LLM_OUTPUT_REGEX = re.compile(r"^LLM_OUTPUT__.*\.txt$", re.IGNORECASE)
DOC_NAME = "Deliverables Analysis"

RETRYABLE_STATUS_CODES = {429, 500, 502, 503, 504}
MAX_RETRIES = 6

# =========================
# Retry helpers
# =========================
def drive_execute(req, label: str = "Drive API call"):
    for attempt in range(1, MAX_RETRIES + 1):
        try:
            return req.execute()
        except HttpError as e:
            status = getattr(e.resp, "status", None)
            if status in RETRYABLE_STATUS_CODES and attempt < MAX_RETRIES:
                sleep_s = min(2 ** (attempt - 1), 20) + random.uniform(0, 1)
                print(
                    f"[RETRY] {label} failed with HTTP {status}. "
                    f"Attempt {attempt}/{MAX_RETRIES}. Sleeping {sleep_s:.1f}s..."
                )
                time.sleep(sleep_s)
                continue
            raise
        except Exception as e:
            if attempt < MAX_RETRIES:
                sleep_s = min(2 ** (attempt - 1), 20) + random.uniform(0, 1)
                print(
                    f"[RETRY] {label} failed ({e}). "
                    f"Attempt {attempt}/{MAX_RETRIES}. Sleeping {sleep_s:.1f}s..."
                )
                time.sleep(sleep_s)
                continue
            raise

def is_retryable_http_error(e: Exception) -> bool:
    if not isinstance(e, HttpError):
        return False
    status = getattr(e.resp, "status", None)
    return status in RETRYABLE_STATUS_CODES

# =========================
# Drive auth
# =========================
def get_drive_service():
    if AUTH_MODE != "service_account":
        raise RuntimeError(f"Unsupported AUTH_MODE='{AUTH_MODE}'. Use AUTH_MODE=service_account")

    if not SERVICE_ACCOUNT_FILE.exists():
        raise FileNotFoundError(
            f"Service account file not found: {SERVICE_ACCOUNT_FILE}\n"
            f"Set SERVICE_ACCOUNT_FILE in .env or place service-account.json next to this script."
        )

    creds = service_account.Credentials.from_service_account_file(
        str(SERVICE_ACCOUNT_FILE),
        scopes=SCOPES,
    )

    if USE_DELEGATION:
        if not DELEGATED_USER_EMAIL:
            raise RuntimeError("USE_DELEGATION=true but DELEGATED_USER_EMAIL is empty.")
        creds = creds.with_subject(DELEGATED_USER_EMAIL)
        print(f"[AUTH] Service account with delegation -> {DELEGATED_USER_EMAIL}")
    else:
        print("[AUTH] Service account without delegation")

    return build("drive", "v3", credentials=creds)

# =========================
# Drive kwargs
# =========================
def _list_kwargs_for_drive(drive_id: Optional[str] = None):
    kwargs = {
        "supportsAllDrives": True,
        "includeItemsFromAllDrives": True,
    }
    if drive_id:
        kwargs["corpora"] = "drive"
        kwargs["driveId"] = drive_id
    else:
        kwargs["corpora"] = "allDrives"
    return kwargs

def _get_kwargs():
    return {"supportsAllDrives": True}

def _write_kwargs():
    return {"supportsAllDrives": True}

# =========================
# Drive helpers
# =========================
def _escape_drive_q_value(s: str) -> str:
    return s.replace("\\", "\\\\").replace("'", "\\'")

def _slot_sort_key(name: str):
    m = re.search(r"(\d+)", name or "")
    return (int(m.group(1)) if m else float("inf"), (name or "").lower())

def list_shared_drives(service) -> List[dict]:
    out = []
    page_token = None
    while True:
        res = drive_execute(
            service.drives().list(
                pageSize=100,
                pageToken=page_token,
            ),
            label="list shared drives",
        )
        out.extend(res.get("drives", []))
        page_token = res.get("nextPageToken")
        if not page_token:
            break
    return out

def get_shared_drive_by_name(service, drive_name: str) -> dict:
    drives = list_shared_drives(service)
    matches = [d for d in drives if d.get("name") == drive_name]

    if not matches:
        available = sorted([d.get("name", "") for d in drives])
        raise RuntimeError(
            f"Shared Drive '{drive_name}' not found.\n"
            f"Visible Shared Drives: {available}"
        )

    if len(matches) > 1:
        print(f"[WARN] Multiple Shared Drives named '{drive_name}'. Using first match.")

    return matches[0]

def drive_get_file(service, file_id: str) -> dict:
    return drive_execute(
        service.files().get(
            fileId=file_id,
            fields="id,name,parents,modifiedTime,mimeType,driveId",
            **_get_kwargs(),
        ),
        label=f"get file {file_id}",
    )

def drive_find_child(
    service,
    parent_id: str,
    name: str,
    mime_type: Optional[str] = None,
    drive_id: Optional[str] = None,
):
    safe_name = _escape_drive_q_value(name)
    q_parts = [f"'{parent_id}' in parents", "trashed = false", f"name = '{safe_name}'"]
    if mime_type:
        q_parts.append(f"mimeType = '{mime_type}'")
    q = " and ".join(q_parts)

    res = drive_execute(
        service.files().list(
            q=q,
            fields="files(id,name,mimeType,modifiedTime,driveId)",
            pageSize=50,
            **_list_kwargs_for_drive(drive_id),
        ),
        label=f"find child '{name}' in parent {parent_id}",
    )

    files = res.get("files", []) or []
    return sorted(files, key=lambda f: f.get("modifiedTime") or "", reverse=True)[0] if files else None

def drive_list_children(service, parent_id: str, mime_type: Optional[str] = None, drive_id: Optional[str] = None):
    q_parts = [f"'{parent_id}' in parents", "trashed = false"]
    if mime_type:
        q_parts.append(f"mimeType = '{mime_type}'")
    q = " and ".join(q_parts)

    page_token = None
    while True:
        res = drive_execute(
            service.files().list(
                q=q,
                fields="nextPageToken, files(id,name,mimeType,modifiedTime,size,driveId)",
                pageSize=1000,
                pageToken=page_token,
                **_list_kwargs_for_drive(drive_id),
            ),
            label=f"list children of parent {parent_id}",
        )

        for f in res.get("files", []):
            yield f

        page_token = res.get("nextPageToken")
        if not page_token:
            break

def drive_download_text(service, file_id: str) -> str:
    for attempt in range(1, MAX_RETRIES + 1):
        try:
            request = service.files().get_media(fileId=file_id, **_get_kwargs())
            fh = io.BytesIO()
            downloader = MediaIoBaseDownload(fh, request, chunksize=1024 * 1024 * 4)

            done = False
            while not done:
                _, done = downloader.next_chunk()

            fh.seek(0)
            return fh.read().decode("utf-8", errors="ignore")

        except Exception as e:
            retryable = is_retryable_http_error(e)
            if retryable and attempt < MAX_RETRIES:
                status = getattr(getattr(e, "resp", None), "status", None)
                sleep_s = min(2 ** (attempt - 1), 20) + random.uniform(0, 1)
                print(
                    f"[RETRY] download file {file_id} failed with HTTP {status}. "
                    f"Attempt {attempt}/{MAX_RETRIES}. Sleeping {sleep_s:.1f}s..."
                )
                time.sleep(sleep_s)
                continue
            raise

def drive_create_or_replace_gdoc_from_text(
    service,
    parent_id: str,
    doc_name: str,
    text: str,
    drive_id: Optional[str] = None,
):
    existing = drive_find_child(service, parent_id, doc_name, GDOC_MIME, drive_id)
    if existing:
        drive_execute(
            service.files().delete(fileId=existing["id"], **_write_kwargs()),
            label=f"delete existing doc '{doc_name}'",
        )

    media = MediaIoBaseUpload(
        io.BytesIO(text.encode("utf-8")),
        mimetype="text/plain",
        resumable=False,
    )
    meta = {"name": doc_name, "mimeType": GDOC_MIME, "parents": [parent_id]}

    created = drive_execute(
        service.files().create(
            body=meta,
            media_body=media,
            fields="id",
            **_write_kwargs(),
        ),
        label=f"create gdoc '{doc_name}'",
    )
    return created["id"]

def drive_search_folder_anywhere_in_shared_drive(service, folder_name: str, drive_id: str) -> List[dict]:
    safe_name = _escape_drive_q_value(folder_name)
    q = f"name = '{safe_name}' and mimeType = '{FOLDER_MIME}' and trashed = false"

    out = []
    page_token = None
    while True:
        res = drive_execute(
            service.files().list(
                q=q,
                fields="nextPageToken, files(id,name,parents,modifiedTime,driveId,mimeType)",
                pageSize=1000,
                pageToken=page_token,
                **_list_kwargs_for_drive(drive_id),
            ),
            label=f"search folder '{folder_name}' in shared drive {drive_id}",
        )
        out.extend(res.get("files", []))
        page_token = res.get("nextPageToken")
        if not page_token:
            break
    return out

def pick_best_named_folder(candidates: List[dict]) -> dict:
    return sorted(candidates, key=lambda c: c.get("modifiedTime") or "", reverse=True)[0]

# =========================
# SLOT SELECTION
# =========================
def list_slot_folders(service, slots_parent_id: str, drive_id: str) -> List[dict]:
    return sorted(
        list(drive_list_children(service, slots_parent_id, FOLDER_MIME, drive_id)),
        key=lambda x: _slot_sort_key(x.get("name") or ""),
    )

def choose_slot(service, slots_parent_id: str, drive_id: str) -> dict:
    slots = list_slot_folders(service, slots_parent_id, drive_id)
    if not slots:
        raise RuntimeError("No slot folders found under 2026 inside the selected Shared Drive.")

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
        print("Invalid choice. Try again.")

# =========================
# Merge logic
# =========================
def build_person_doc_content(
    service,
    slot_name: str,
    person_name: str,
    person_folder_id: str,
    drive_id: str,
) -> str:
    folder_nodes = sorted(
        list(drive_list_children(service, person_folder_id, FOLDER_MIME, drive_id)),
        key=lambda x: (x.get("name") or "").lower(),
    )

    sections: List[str] = []
    header = (
        f"{person_name}\n"
        f"Slot: {slot_name}\n"
        f"Generated: {time.strftime('%Y-%m-%d %H:%M:%S')}\n"
    )
    sections.append(header)
    sections.append("=" * 90)

    total_files = 0

    for folder_node in folder_nodes:
        folder_name = folder_node["name"]

        try:
            files = list(drive_list_children(service, folder_node["id"], None, drive_id))
        except Exception as e:
            sections.append(f"\n\n## {folder_name}\n" + "-" * 90)
            sections.append(f"\n[ERROR READING FOLDER] {e}")
            print(f"[WARN] Could not read folder '{folder_name}' for '{person_name}': {e}")
            continue

        llm_txts = [
            f for f in files
            if f.get("mimeType") != FOLDER_MIME
            and LLM_OUTPUT_REGEX.match((f.get("name") or ""))
        ]

        if not llm_txts:
            continue

        llm_txts = sorted(llm_txts, key=lambda x: (x.get("name") or "").lower())
        sections.append(f"\n\n## {folder_name}\n" + "-" * 90)

        for f in llm_txts:
            total_files += 1
            try:
                content = drive_download_text(service, f["id"]).strip()
            except Exception as e:
                print(f"[WARN] Could not download file '{f.get('name')}' for '{person_name}': {e}")
                sections.append(f"\n\n### {f['name']}\n")
                sections.append(f"[ERROR DOWNLOADING FILE] {e}")
                continue

            sections.append(f"\n\n### {f['name']}\n")
            sections.append(content if content else "[EMPTY OUTPUT]")

    if total_files == 0:
        sections.append("\n\nNo LLM output files found (expected: LLM_OUTPUT__*.txt).")

    sections.append("\n")
    return "\n".join(sections)

# =========================
# MAIN
# =========================
def main():
    if not SHARED_DRIVE_NAME:
        raise RuntimeError("SHARED_DRIVE_NAME is required. Example: SHARED_DRIVE_NAME=2026_Shared_Drive")

    service = get_drive_service()

    print("[INFO] Shared-Drive-only mode enabled")
    print(f"[INFO] SHARED_DRIVE_NAME={SHARED_DRIVE_NAME}")
    print(f"[INFO] ROOT_2026_FOLDER_NAME={ROOT_2026_FOLDER_NAME}")

    shared_drive = get_shared_drive_by_name(service, SHARED_DRIVE_NAME)
    shared_drive_id = shared_drive["id"]

    print(f"[INFO] Using Shared Drive: {shared_drive['name']} ({shared_drive_id})")

    if ROOT_2026_FOLDER_ID:
        print(f"[ROOT] Using ROOT_2026_FOLDER_ID={ROOT_2026_FOLDER_ID}")
        base_2026 = drive_get_file(service, ROOT_2026_FOLDER_ID)

        if base_2026.get("mimeType") != FOLDER_MIME:
            raise RuntimeError(f"ROOT_2026_FOLDER_ID={ROOT_2026_FOLDER_ID} is not a folder.")

        file_drive_id = base_2026.get("driveId")
        if file_drive_id and file_drive_id != shared_drive_id:
            raise RuntimeError(
                f"ROOT_2026_FOLDER_ID belongs to driveId={file_drive_id}, "
                f"but SHARED_DRIVE_NAME resolved to driveId={shared_drive_id}."
            )
    else:
        candidates = drive_search_folder_anywhere_in_shared_drive(service, ROOT_2026_FOLDER_NAME, shared_drive_id)
        if not candidates:
            raise RuntimeError(
                f"Could not find folder '{ROOT_2026_FOLDER_NAME}' inside Shared Drive '{SHARED_DRIVE_NAME}'."
            )

        if len(candidates) > 1:
            print(f"[WARN] Multiple folders named '{ROOT_2026_FOLDER_NAME}' inside Shared Drive '{SHARED_DRIVE_NAME}'.")
            for c in candidates[:10]:
                print(
                    " - id:", c["id"],
                    "modified:", c.get("modifiedTime"),
                    "parents:", c.get("parents"),
                    "driveId:", c.get("driveId"),
                )
            print("[WARN] Using the most recently modified one inside this Shared Drive.")

        base_2026 = pick_best_named_folder(candidates)
        print(f"[ROOT] Using folder: {base_2026['name']} (id={base_2026['id']})")

    slot = choose_slot(service, base_2026["id"], shared_drive_id)

    all_people = sorted(
        list(drive_list_children(service, slot["id"], FOLDER_MIME, shared_drive_id)),
        key=lambda x: (x.get("name") or "").lower(),
    )

    people = [
        p for p in all_people
        if (p.get("name") or "").strip() not in SKIP_PERSON_FOLDERS
    ]

    print(f"\n[INFO] Slot selected: {slot['name']}")
    print(f"[INFO] Total candidate folders to process: {len(people)}")
    if SKIP_PERSON_FOLDERS:
        print(f"[INFO] Skipping folders: {sorted(SKIP_PERSON_FOLDERS)}")

    ok_count = 0
    fail_count = 0

    for person in people:
        print(f"\n[BUILD] Slot={slot['name']}  Person={person['name']}")
        try:
            combined_text = build_person_doc_content(
                service=service,
                slot_name=slot["name"],
                person_name=person["name"],
                person_folder_id=person["id"],
                drive_id=shared_drive_id,
            )

            doc_id = drive_create_or_replace_gdoc_from_text(
                service=service,
                parent_id=person["id"],
                doc_name=DOC_NAME,
                text=combined_text,
                drive_id=shared_drive_id,
            )

            print(
                f"[OK] Created Google Doc: {DOC_NAME} (id={doc_id}) "
                f"in 2026/{slot['name']}/{person['name']}"
            )
            ok_count += 1

        except Exception as e:
            fail_count += 1
            print(f"[ERROR] Failed for person '{person['name']}': {e}")

        time.sleep(0.25)

    print("\n" + "=" * 80)
    print("DONE")
    print("=" * 80)
    print(f"[SUMMARY] Success: {ok_count}")
    print(f"[SUMMARY] Failed : {fail_count}")

if __name__ == "__main__":
    main()