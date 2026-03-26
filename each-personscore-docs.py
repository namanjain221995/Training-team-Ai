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
from google.oauth2.credentials import Credentials
from google_auth_oauthlib.flow import InstalledAppFlow
from google.auth.transport.requests import Request

# =========================
# ENV
# =========================
load_dotenv()

# Non-interactive slot selection
# Example in .env: SLOT_CHOICE=2
SLOT_CHOICE = (os.getenv("SLOT_CHOICE") or "").strip()

# Shared drives flag from env
USE_SHARED_DRIVES = (os.getenv("USE_SHARED_DRIVES") or "").strip().lower() in ("1", "true", "yes", "y")

# Optional: safer than searching by folder name
ROOT_2026_FOLDER_ID = (os.getenv("ROOT_2026_FOLDER_ID") or "").strip()

# Optional skip list (comma-separated)
# Example: SKIP_PERSON_FOLDERS=1. Format,Archive
SKIP_PERSON_FOLDERS = {
    x.strip() for x in (os.getenv("SKIP_PERSON_FOLDERS") or "1. Format").split(",") if x.strip()
}

# Optional: headless auth support
HEADLESS_AUTH = (os.getenv("HEADLESS_AUTH") or "").strip().lower() in ("1", "true", "yes", "y")

# =========================
# CONFIG
# =========================
SCOPES = ["https://www.googleapis.com/auth/drive"]
CREDENTIALS_FILE = Path("credentials.json")
TOKEN_FILE = Path("token.json")

ROOT_2026_FOLDER_NAME = "2026"  # used if ROOT_2026_FOLDER_ID not provided

FOLDER_MIME = "application/vnd.google-apps.folder"
GDOC_MIME = "application/vnd.google-apps.document"

# Only merge LLM output files matching this pattern
LLM_OUTPUT_REGEX = re.compile(r"^LLM_OUTPUT__.*\.txt$", re.IGNORECASE)

# Name for the Google Doc created in each person folder
DOC_NAME = "Deliverables Analysis"

# Retry config
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
    creds = None

    if TOKEN_FILE.exists():
        creds = Credentials.from_authorized_user_file(str(TOKEN_FILE), SCOPES)

    # If token exists but scopes changed, refresh will fail. Force new login.
    if creds and set(creds.scopes or []) != set(SCOPES):
        print("[AUTH] token.json scopes mismatch. Deleting token.json and re-authenticating...")
        TOKEN_FILE.unlink(missing_ok=True)
        creds = None

    if not creds or not creds.valid:
        if creds and creds.expired and creds.refresh_token:
            try:
                creds.refresh(Request())
                TOKEN_FILE.write_text(creds.to_json(), encoding="utf-8")
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
        return {
            "supportsAllDrives": True,
            "includeItemsFromAllDrives": True,
            "corpora": "allDrives",
        }
    return {}


def _get_media_kwargs():
    if USE_SHARED_DRIVES:
        return {"supportsAllDrives": True}
    return {}


def _write_kwargs():
    if USE_SHARED_DRIVES:
        return {"supportsAllDrives": True}
    return {}


# =========================
# Drive helpers
# =========================
def _escape_drive_q_value(s: str) -> str:
    return s.replace("'", "''")


def _slot_sort_key(name: str):
    m = re.search(r"(\d+)", name or "")
    return (int(m.group(1)) if m else float("inf"), (name or "").lower())


def drive_find_child(service, parent_id: str, name: str, mime_type: Optional[str] = None):
    safe_name = _escape_drive_q_value(name)
    q_parts = [f"'{parent_id}' in parents", "trashed = false", f"name = '{safe_name}'"]
    if mime_type:
        q_parts.append(f"mimeType = '{mime_type}'")
    q = " and ".join(q_parts)

    res = drive_execute(
        service.files().list(
            q=q,
            fields="files(id,name,mimeType,modifiedTime)",
            pageSize=50,
            **_list_kwargs(),
        ),
        label=f"find child '{name}' in parent {parent_id}",
    )

    files = res.get("files", []) or []
    return sorted(files, key=lambda f: f.get("modifiedTime") or "", reverse=True)[0] if files else None


def drive_list_children(service, parent_id: str, mime_type: Optional[str] = None):
    q_parts = [f"'{parent_id}' in parents", "trashed = false"]
    if mime_type:
        q_parts.append(f"mimeType = '{mime_type}'")
    q = " and ".join(q_parts)

    page_token = None
    while True:
        res = drive_execute(
            service.files().list(
                q=q,
                fields="nextPageToken, files(id,name,mimeType,modifiedTime,size)",
                pageSize=1000,
                pageToken=page_token,
                **_list_kwargs(),
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
            request = service.files().get_media(fileId=file_id, **_get_media_kwargs())
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


def drive_create_or_replace_gdoc_from_text(service, parent_id: str, doc_name: str, text: str):
    """
    Creates a Google Doc in parent folder.
    If exists, deletes and recreates it.
    """
    existing = drive_find_child(service, parent_id, doc_name, GDOC_MIME)
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


def drive_search_folder_anywhere(service, folder_name: str) -> List[dict]:
    safe_name = _escape_drive_q_value(folder_name)
    q = f"name = '{safe_name}' and mimeType = '{FOLDER_MIME}' and trashed = false"

    res = drive_execute(
        service.files().list(
            q=q,
            fields="files(id,name,parents,modifiedTime)",
            pageSize=200,
            **_list_kwargs(),
        ),
        label=f"search folder '{folder_name}' anywhere",
    )
    return res.get("files", []) or []


def drive_get_file(service, file_id: str) -> dict:
    return drive_execute(
        service.files().get(
            fileId=file_id,
            fields="id,name,parents,modifiedTime,mimeType",
            **_get_media_kwargs(),
        ),
        label=f"get file {file_id}",
    )


def pick_best_named_folder(candidates: List[dict]) -> dict:
    return sorted(candidates, key=lambda c: c.get("modifiedTime") or "", reverse=True)[0]


# =========================
# SLOT SELECTION
# =========================
def list_slot_folders(service, slots_parent_id: str) -> List[dict]:
    return sorted(
        list(drive_list_children(service, slots_parent_id, FOLDER_MIME)),
        key=lambda x: _slot_sort_key(x.get("name") or ""),
    )


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
        print("Invalid choice. Try again.")


# =========================
# Merge logic
# =========================
def build_person_doc_content(service, slot_name: str, person_name: str, person_folder_id: str) -> str:
    """
    Looks inside 2026/<Slot>/<Person>/<FolderName>
    Collects LLM_OUTPUT__*.txt from each FolderName.
    Builds one combined text.
    """
    folder_nodes = sorted(
        list(drive_list_children(service, person_folder_id, FOLDER_MIME)),
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
            files = list(drive_list_children(service, folder_node["id"], None))
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
    service = get_drive_service()

    if ROOT_2026_FOLDER_ID:
        print(f"[ROOT] Using ROOT_2026_FOLDER_ID={ROOT_2026_FOLDER_ID}")
        base_2026 = drive_get_file(service, ROOT_2026_FOLDER_ID)
        if base_2026.get("mimeType") != FOLDER_MIME:
            raise RuntimeError(f"ROOT_2026_FOLDER_ID={ROOT_2026_FOLDER_ID} is not a folder.")
    else:
        candidates = drive_search_folder_anywhere(service, ROOT_2026_FOLDER_NAME)
        if not candidates:
            raise RuntimeError(f"Could not find folder '{ROOT_2026_FOLDER_NAME}' anywhere in Drive.")
        base_2026 = pick_best_named_folder(candidates)
        print(f"[ROOT] Found folder by name: {base_2026['name']} (id={base_2026['id']})")

    slots_parent_id = base_2026["id"]

    slot = choose_slot(service, slots_parent_id)

    all_people = sorted(
        list(drive_list_children(service, slot["id"], FOLDER_MIME)),
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
            )

            doc_id = drive_create_or_replace_gdoc_from_text(
                service=service,
                parent_id=person["id"],
                doc_name=DOC_NAME,
                text=combined_text,
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