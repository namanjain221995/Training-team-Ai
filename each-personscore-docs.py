import os
import io
import re
import time
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

# Non-interactive slot selection (Option B)
# Example in .env: SLOT_CHOICE=2
SLOT_CHOICE = (os.getenv("SLOT_CHOICE") or "").strip()

# Shared drives flag from env (optional)
USE_SHARED_DRIVES = (os.getenv("USE_SHARED_DRIVES") or "").strip().lower() in ("1", "true", "yes", "y")

# =========================
# CONFIG
# =========================
SCOPES = ["https://www.googleapis.com/auth/drive"]
CREDENTIALS_FILE = Path("credentials.json")
TOKEN_FILE = Path("token.json")

ROOT_2026_FOLDER_NAME = "2026"  # can be nested anywhere in My Drive

FOLDER_MIME = "application/vnd.google-apps.folder"
GDOC_MIME = "application/vnd.google-apps.document"

# Only merge LLM output files matching this pattern
LLM_OUTPUT_REGEX = re.compile(r"^LLM_OUTPUT__.*\.txt$", re.IGNORECASE)

# Name for the Google Doc created in each person folder
DOC_NAME = "Deliverables Analysis"


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
            except Exception as e:
                print(f"[AUTH] Refresh failed ({e}). Re-authenticating...")
                TOKEN_FILE.unlink(missing_ok=True)
                creds = None

        if not creds or not creds.valid:
            if not CREDENTIALS_FILE.exists():
                raise FileNotFoundError("credentials.json not found next to this script.")
            flow = InstalledAppFlow.from_client_secrets_file(str(CREDENTIALS_FILE), SCOPES)
            # NOTE: In Docker/headless you may want flow.run_console()
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


# =========================
# Drive helpers
# =========================
def _escape_drive_q_value(s: str) -> str:
    # Drive query string escaping: single quote is escaped by doubling it
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
            fields="nextPageToken, files(id,name,mimeType,modifiedTime,size)",
            pageSize=1000,
            pageToken=page_token,
            **_list_kwargs(),
        ).execute()

        for f in res.get("files", []):
            yield f

        page_token = res.get("nextPageToken")
        if not page_token:
            break


def drive_download_text(service, file_id: str) -> str:
    request = service.files().get_media(fileId=file_id, **_get_media_kwargs())
    fh = io.BytesIO()
    downloader = MediaIoBaseDownload(fh, request, chunksize=1024 * 1024 * 4)
    done = False
    while not done:
        _, done = downloader.next_chunk()
    fh.seek(0)
    return fh.read().decode("utf-8", errors="ignore")


def drive_create_or_replace_gdoc_from_text(service, parent_id: str, doc_name: str, text: str):
    """
    Creates a Google Doc in parent folder.
    If exists, deletes+recreates (simpler than Docs API updates).
    """
    existing = drive_find_child(service, parent_id, doc_name, GDOC_MIME)
    if existing:
        service.files().delete(fileId=existing["id"], **_write_kwargs()).execute()

    media = MediaIoBaseUpload(io.BytesIO(text.encode("utf-8")), mimetype="text/plain", resumable=False)
    meta = {"name": doc_name, "mimeType": GDOC_MIME, "parents": [parent_id]}
    created = service.files().create(body=meta, media_body=media, fields="id", **_write_kwargs()).execute()
    return created["id"]


def drive_search_folder_anywhere(service, folder_name: str) -> List[dict]:
    safe_name = _escape_drive_q_value(folder_name)
    q = f"name = '{safe_name}' and mimeType = '{FOLDER_MIME}' and trashed=false"
    res = service.files().list(
        q=q,
        fields="files(id,name,parents,modifiedTime)",
        pageSize=200,
        **_list_kwargs(),
    ).execute()
    return res.get("files", []) or []


def pick_best_named_folder(candidates: List[dict]) -> dict:
    return sorted(candidates, key=lambda c: c.get("modifiedTime") or "", reverse=True)[0]


# =========================
# SLOT SELECTION (supports SLOT_CHOICE env)
# =========================
def list_slot_folders(service, slots_parent_id: str) -> List[dict]:
    return sorted(
        list(drive_list_children(service, slots_parent_id, FOLDER_MIME)),
        key=lambda x: (x.get("name") or "").lower(),
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

    # interactive fallback
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
    header = f"{person_name}\nSlot: {slot_name}\nGenerated: {time.strftime('%Y-%m-%d %H:%M:%S')}\n"
    sections.append(header)
    sections.append("=" * 90)

    total_files = 0

    for folder_node in folder_nodes:
        folder_name = folder_node["name"]
        files = list(drive_list_children(service, folder_node["id"], None))

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
            content = drive_download_text(service, f["id"]).strip()
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

    candidates = drive_search_folder_anywhere(service, ROOT_2026_FOLDER_NAME)
    if not candidates:
        raise RuntimeError(f"Could not find folder '{ROOT_2026_FOLDER_NAME}' anywhere in Drive.")

    base_2026 = pick_best_named_folder(candidates)
    slots_parent_id = base_2026["id"]

    # Pick slot (AUTO if SLOT_CHOICE provided)
    slot = choose_slot(service, slots_parent_id)

    people = sorted(
        list(drive_list_children(service, slot["id"], FOLDER_MIME)),
        key=lambda x: (x.get("name") or "").lower(),
    )

    for person in people:
        print(f"\n[BUILD] Slot={slot['name']}  Person={person['name']}")
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
        print(f"[OK] Created Google Doc: {DOC_NAME} (id={doc_id}) in 2026/{slot['name']}/{person['name']}")

        time.sleep(0.25)


if __name__ == "__main__":
    main()
