import os
import time
import random
import re
from pathlib import Path
from typing import Optional, List, Dict, Any

from dotenv import load_dotenv

from googleapiclient.discovery import build
from googleapiclient.errors import HttpError
from google.oauth2 import service_account

# =========================
# ENV
# =========================
load_dotenv()

SLOT_CHOICE = (os.getenv("SLOT_CHOICE") or "").strip()

AUTH_MODE = (os.getenv("AUTH_MODE") or "service_account").strip().lower()
SERVICE_ACCOUNT_FILE = Path((os.getenv("SERVICE_ACCOUNT_FILE") or "service-account.json").strip())
USE_DELEGATION = (os.getenv("USE_DELEGATION") or "").strip().lower() in ("1", "true", "yes", "y")
DELEGATED_USER_EMAIL = (os.getenv("DELEGATED_USER_EMAIL") or "").strip()

SHARED_DRIVE_NAME = (os.getenv("SHARED_DRIVE_NAME") or "").strip()
ROOT_2026_FOLDER_NAME = (os.getenv("ROOT_2026_FOLDER_NAME") or "2026").strip()
OUTPUT_ROOT_FOLDER_NAME = (os.getenv("OUTPUT_ROOT_FOLDER_NAME") or "Candidate Result").strip()

# Optional exact IDs inside selected Shared Drive
ROOT_2026_FOLDER_ID = (os.getenv("ROOT_2026_FOLDER_ID") or "").strip()
OUTPUT_ROOT_FOLDER_ID = (os.getenv("OUTPUT_ROOT_FOLDER_ID") or "").strip()

SKIP_PERSON_FOLDERS = {
    x.strip() for x in (os.getenv("SKIP_PERSON_FOLDERS") or "1. Format").split(",") if x.strip()
}

# =========================
# CONFIG
# =========================
SCOPES = [
    "https://www.googleapis.com/auth/drive",
    "https://www.googleapis.com/auth/documents",
]

FOLDER_MIME = "application/vnd.google-apps.folder"
GDOC_MIME = "application/vnd.google-apps.document"

PERSON_DOC_NAME = "Deliverables Analysis"
SLOT_DOC_NAME = "All Deliverables Analysis"

EDITOR_EMAILS = [
    "rajvi.patel@techsarasolutions.com",
    "sahil.patel@techsarasolutions.com",
    "soham.piprotar@techsarasolutions.com",
]

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

def docs_execute(req, label: str = "Docs API call"):
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

# =========================
# Auth
# =========================
def get_service_account_creds():
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

    return creds

def get_drive_service(creds):
    return build("drive", "v3", credentials=creds)

def get_docs_service(creds):
    return build("docs", "v1", credentials=creds)

# =========================
# Drive kwargs
# =========================
def _list_kwargs_for_drive(drive_id: Optional[str] = None) -> Dict[str, Any]:
    kwargs: Dict[str, Any] = {
        "supportsAllDrives": True,
        "includeItemsFromAllDrives": True,
    }
    if drive_id:
        kwargs["corpora"] = "drive"
        kwargs["driveId"] = drive_id
    else:
        kwargs["corpora"] = "allDrives"
    return kwargs

def _get_kwargs() -> Dict[str, Any]:
    return {"supportsAllDrives": True}

def _write_kwargs() -> Dict[str, Any]:
    return {"supportsAllDrives": True}

# =========================
# Drive helpers
# =========================
def _escape_drive_q_value(s: str) -> str:
    return s.replace("\\", "\\\\").replace("'", "\\'")

def _slot_sort_key(name: str):
    m = re.search(r"(\d+)", name or "")
    return (int(m.group(1)) if m else float("inf"), (name or "").lower())

def list_shared_drives(drive) -> List[dict]:
    out = []
    page_token = None
    while True:
        res = drive_execute(
            drive.drives().list(
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

def get_shared_drive_by_name(drive, drive_name: str) -> dict:
    drives = list_shared_drives(drive)
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

def drive_get_file(drive, file_id: str) -> dict:
    return drive_execute(
        drive.files().get(
            fileId=file_id,
            fields="id,name,parents,modifiedTime,mimeType,driveId",
            **_get_kwargs(),
        ),
        label=f"get file {file_id}",
    )

def drive_search_folder_anywhere_in_shared_drive(drive, folder_name: str, drive_id: str) -> List[dict]:
    safe = _escape_drive_q_value(folder_name)
    q = f"name = '{safe}' and mimeType = '{FOLDER_MIME}' and trashed = false"

    out = []
    page_token = None
    while True:
        res = drive_execute(
            drive.files().list(
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

def drive_list_children(drive, parent_id: str, mime_type: Optional[str] = None, drive_id: Optional[str] = None):
    q_parts = [f"'{parent_id}' in parents", "trashed = false"]
    if mime_type:
        q_parts.append(f"mimeType = '{mime_type}'")
    q = " and ".join(q_parts)

    page_token = None
    while True:
        res = drive_execute(
            drive.files().list(
                q=q,
                fields="nextPageToken, files(id,name,mimeType,modifiedTime,driveId)",
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

def drive_find_children(
    drive,
    parent_id: str,
    name: str,
    mime_type: Optional[str] = None,
    drive_id: Optional[str] = None,
) -> List[dict]:
    safe = _escape_drive_q_value(name)
    q_parts = [f"'{parent_id}' in parents", "trashed = false", f"name = '{safe}'"]
    if mime_type:
        q_parts.append(f"mimeType = '{mime_type}'")
    q = " and ".join(q_parts)

    res = drive_execute(
        drive.files().list(
            q=q,
            fields="files(id,name,mimeType,parents,modifiedTime,driveId)",
            pageSize=200,
            **_list_kwargs_for_drive(drive_id),
        ),
        label=f"find children named '{name}' in parent {parent_id}",
    )
    return res.get("files", []) or []

def drive_find_child(
    drive,
    parent_id: str,
    name: str,
    mime_type: Optional[str] = None,
    drive_id: Optional[str] = None,
) -> Optional[dict]:
    files = drive_find_children(drive, parent_id, name, mime_type, drive_id)
    if not files:
        return None
    return sorted(files, key=lambda f: f.get("modifiedTime") or "", reverse=True)[0]

def drive_delete_file(drive, file_id: str):
    drive_execute(
        drive.files().delete(fileId=file_id, **_write_kwargs()),
        label=f"delete file {file_id}",
    )

def drive_delete_all_named(
    drive,
    parent_id: str,
    name: str,
    mime_type: Optional[str] = None,
    drive_id: Optional[str] = None,
) -> int:
    matches = drive_find_children(drive, parent_id, name, mime_type, drive_id)
    for f in matches:
        drive_delete_file(drive, f["id"])
    return len(matches)

def drive_create_folder(drive, parent_id: str, name: str) -> str:
    meta = {"name": name, "mimeType": FOLDER_MIME, "parents": [parent_id]}
    created = drive_execute(
        drive.files().create(
            body=meta,
            fields="id",
            **_write_kwargs(),
        ),
        label=f"create folder '{name}'",
    )
    return created["id"]

def drive_create_gdoc(drive, parent_id: str, name: str) -> str:
    meta = {"name": name, "mimeType": GDOC_MIME, "parents": [parent_id]}
    created = drive_execute(
        drive.files().create(
            body=meta,
            fields="id",
            **_write_kwargs(),
        ),
        label=f"create gdoc '{name}'",
    )
    return created["id"]

def drive_grant_editor_access(drive, file_id: str, emails: List[str]):
    for email in emails:
        perm = {
            "type": "user",
            "role": "writer",
            "emailAddress": email,
        }
        try:
            drive_execute(
                drive.permissions().create(
                    fileId=file_id,
                    body=perm,
                    sendNotificationEmail=False,
                    **_write_kwargs(),
                ),
                label=f"grant editor access to {email}",
            )
            print(f"  [PERM] Editor added: {email}")
        except HttpError as e:
            msg = (e.content or b"").decode("utf-8", errors="ignore").lower()
            if "alreadyexists" in msg or "already exists" in msg or "duplicate" in msg:
                print(f"  [PERM] Already has access: {email}")
            else:
                print(f"  [PERM] Failed for {email}: {e}")

# =========================
# SLOT SELECTION
# =========================
def list_slot_folders(drive, slots_parent_id: str, drive_id: str) -> List[dict]:
    return sorted(
        list(drive_list_children(drive, slots_parent_id, FOLDER_MIME, drive_id)),
        key=lambda x: _slot_sort_key(x.get("name") or ""),
    )

def choose_slot(drive, slots_parent_id: str, drive_id: str) -> dict:
    slots = list_slot_folders(drive, slots_parent_id, drive_id)
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
# Docs helpers (tabs + text)
# =========================
def docs_get_document(docs, document_id: str) -> Dict[str, Any]:
    return docs_execute(
        docs.documents().get(documentId=document_id, includeTabsContent=True),
        label=f"get docs document {document_id}",
    )

def flatten_tabs(doc: Dict[str, Any]) -> List[Dict[str, Any]]:
    out: List[Dict[str, Any]] = []

    def rec(tab: Dict[str, Any]):
        out.append(tab)
        for child in tab.get("childTabs", []) or []:
            rec(child)

    for t in doc.get("tabs", []) or []:
        rec(t)
    return out

def read_structural_elements(elements: List[Dict[str, Any]]) -> str:
    parts: List[str] = []
    for el in elements or []:
        if "paragraph" in el:
            for pe in el["paragraph"].get("elements", []) or []:
                tr = pe.get("textRun")
                if tr and "content" in tr:
                    parts.append(tr["content"])
        elif "table" in el:
            table = el["table"]
            for row in table.get("tableRows", []) or []:
                for cell in row.get("tableCells", []) or []:
                    parts.append(read_structural_elements(cell.get("content", [])))
                parts.append("\n")
        elif "tableOfContents" in el:
            toc = el["tableOfContents"]
            parts.append(read_structural_elements(toc.get("content", [])))
    return "".join(parts)

def extract_all_text_from_doc(docs, document_id: str) -> str:
    doc = docs_get_document(docs, document_id)
    tabs = flatten_tabs(doc)
    if not tabs:
        return ""

    if len(tabs) == 1:
        dt = tabs[0].get("documentTab") or {}
        body = (dt.get("body") or {}).get("content", [])
        return read_structural_elements(body).strip()

    chunks: List[str] = []
    for t in tabs:
        props = t.get("tabProperties") or {}
        title = props.get("title") or "Untitled Tab"
        dt = t.get("documentTab") or {}
        body = (dt.get("body") or {}).get("content", [])
        txt = read_structural_elements(body).strip()
        chunks.append(f"## {title}\n\n{txt}\n")
    return "\n\n".join(chunks).strip()

def find_tab_id_by_title(doc: Dict[str, Any], title: str) -> Optional[str]:
    for t in flatten_tabs(doc):
        props = t.get("tabProperties") or {}
        if (props.get("title") or "") == title:
            return props.get("tabId")
    return None

def get_first_tab_id(doc: Dict[str, Any]) -> str:
    tabs = flatten_tabs(doc)
    if not tabs:
        raise RuntimeError("Destination doc has no tabs (unexpected).")
    tab_id = (tabs[0].get("tabProperties") or {}).get("tabId")
    if not tab_id:
        raise RuntimeError("First tab has no tabId (unexpected).")
    return tab_id

def docs_batch_update(docs, document_id: str, requests: List[Dict[str, Any]]):
    return docs_execute(
        docs.documents().batchUpdate(
            documentId=document_id,
            body={"requests": requests},
        ),
        label=f"batch update doc {document_id}",
    )

def set_tab_title(docs, document_id: str, tab_id: str, title: str):
    docs_batch_update(
        docs,
        document_id,
        [
            {
                "updateDocumentTabProperties": {
                    "tabProperties": {"tabId": tab_id, "title": title},
                    "fields": "title",
                }
            }
        ],
    )

def add_tab(docs, document_id: str, title: str) -> str:
    docs_batch_update(
        docs,
        document_id,
        [{"addDocumentTab": {"tabProperties": {"title": title}}}],
    )
    doc = docs_get_document(docs, document_id)
    tab_id = find_tab_id_by_title(doc, title)
    if not tab_id:
        raise RuntimeError(f"Created tab '{title}' but couldn't find its tabId.")
    return tab_id

def insert_text_into_tab(docs, document_id: str, tab_id: str, text: str):
    docs_batch_update(
        docs,
        document_id,
        [
            {
                "insertText": {
                    "location": {"index": 1, "tabId": tab_id},
                    "text": text,
                }
            }
        ],
    )

# =========================
# MAIN
# =========================
def main():
    if not SHARED_DRIVE_NAME:
        raise RuntimeError("SHARED_DRIVE_NAME is required. Example: SHARED_DRIVE_NAME=2026_Shared_Drive")

    creds = get_service_account_creds()
    drive = get_drive_service(creds)
    docs = get_docs_service(creds)

    print("[INFO] Shared-Drive-only mode enabled")
    print(f"[INFO] SHARED_DRIVE_NAME={SHARED_DRIVE_NAME}")
    print(f"[INFO] ROOT_2026_FOLDER_NAME={ROOT_2026_FOLDER_NAME}")
    print(f"[INFO] OUTPUT_ROOT_FOLDER_NAME={OUTPUT_ROOT_FOLDER_NAME}")

    shared_drive = get_shared_drive_by_name(drive, SHARED_DRIVE_NAME)
    shared_drive_id = shared_drive["id"]

    print(f"[INFO] Using Shared Drive: {shared_drive['name']} ({shared_drive_id})")

    # --- SOURCE ROOT ---
    if ROOT_2026_FOLDER_ID:
        print(f"[ROOT] Using ROOT_2026_FOLDER_ID={ROOT_2026_FOLDER_ID}")
        base_2026 = drive_get_file(drive, ROOT_2026_FOLDER_ID)
        if base_2026.get("mimeType") != FOLDER_MIME:
            raise RuntimeError(f"ROOT_2026_FOLDER_ID={ROOT_2026_FOLDER_ID} is not a folder.")
        file_drive_id = base_2026.get("driveId")
        if file_drive_id and file_drive_id != shared_drive_id:
            raise RuntimeError(
                f"ROOT_2026_FOLDER_ID belongs to driveId={file_drive_id}, "
                f"but SHARED_DRIVE_NAME resolved to driveId={shared_drive_id}."
            )
    else:
        candidates_2026 = drive_search_folder_anywhere_in_shared_drive(drive, ROOT_2026_FOLDER_NAME, shared_drive_id)
        if not candidates_2026:
            raise RuntimeError(
                f"Could not find folder '{ROOT_2026_FOLDER_NAME}' inside Shared Drive '{SHARED_DRIVE_NAME}'."
            )
        if len(candidates_2026) > 1:
            print(f"[WARN] Multiple folders named '{ROOT_2026_FOLDER_NAME}' inside Shared Drive '{SHARED_DRIVE_NAME}'.")
            for c in candidates_2026[:10]:
                print(
                    " - id:", c["id"],
                    "modified:", c.get("modifiedTime"),
                    "parents:", c.get("parents"),
                    "driveId:", c.get("driveId"),
                )
            print("[WARN] Using the most recently modified one inside this Shared Drive.")
        base_2026 = pick_best_named_folder(candidates_2026)
        print(f"[ROOT] Using source folder: {base_2026['name']} (id={base_2026['id']})")

    # --- OUTPUT ROOT ---
    if OUTPUT_ROOT_FOLDER_ID:
        print(f"[ROOT] Using OUTPUT_ROOT_FOLDER_ID={OUTPUT_ROOT_FOLDER_ID}")
        output_root = drive_get_file(drive, OUTPUT_ROOT_FOLDER_ID)
        if output_root.get("mimeType") != FOLDER_MIME:
            raise RuntimeError(f"OUTPUT_ROOT_FOLDER_ID={OUTPUT_ROOT_FOLDER_ID} is not a folder.")
        file_drive_id = output_root.get("driveId")
        if file_drive_id and file_drive_id != shared_drive_id:
            raise RuntimeError(
                f"OUTPUT_ROOT_FOLDER_ID belongs to driveId={file_drive_id}, "
                f"but SHARED_DRIVE_NAME resolved to driveId={shared_drive_id}."
            )
    else:
        candidates_out = drive_search_folder_anywhere_in_shared_drive(drive, OUTPUT_ROOT_FOLDER_NAME, shared_drive_id)
        if not candidates_out:
            raise RuntimeError(
                f"Could not find output folder '{OUTPUT_ROOT_FOLDER_NAME}' inside Shared Drive '{SHARED_DRIVE_NAME}'. "
                f"Create it and run again."
            )
        if len(candidates_out) > 1:
            print(f"[WARN] Multiple folders named '{OUTPUT_ROOT_FOLDER_NAME}' inside Shared Drive '{SHARED_DRIVE_NAME}'.")
            for c in candidates_out[:10]:
                print(
                    " - id:", c["id"],
                    "modified:", c.get("modifiedTime"),
                    "parents:", c.get("parents"),
                    "driveId:", c.get("driveId"),
                )
            print("[WARN] Using the most recently modified one inside this Shared Drive.")
        output_root = pick_best_named_folder(candidates_out)
        print(f"[ROOT] Using output folder: {output_root['name']} (id={output_root['id']})")

    # --- SLOT ---
    slot = choose_slot(drive, base_2026["id"], shared_drive_id)
    slot_name = slot["name"]
    slot_id = slot["id"]

    # --- Candidate Result/<SlotName> ---
    slot_output_folder = drive_find_child(drive, output_root["id"], slot_name, FOLDER_MIME, shared_drive_id)
    if not slot_output_folder:
        slot_output_folder_id = drive_create_folder(drive, output_root["id"], slot_name)
        print(f"[OUTPUT] Created folder: {OUTPUT_ROOT_FOLDER_NAME}/{slot_name}")
    else:
        slot_output_folder_id = slot_output_folder["id"]

    print("[PERM] Setting folder editors...")
    drive_grant_editor_access(drive, slot_output_folder_id, EDITOR_EMAILS)

    # --- Slot-level doc ---
    deleted = drive_delete_all_named(
        drive,
        slot_output_folder_id,
        SLOT_DOC_NAME,
        GDOC_MIME,
        shared_drive_id,
    )
    if deleted:
        print(f"[OUTPUT] Deleted {deleted} existing '{SLOT_DOC_NAME}' doc(s) in {OUTPUT_ROOT_FOLDER_NAME}/{slot_name}")

    slot_doc_id = drive_create_gdoc(drive, slot_output_folder_id, SLOT_DOC_NAME)
    print(f"\n[OUTPUT] Created '{SLOT_DOC_NAME}' in {OUTPUT_ROOT_FOLDER_NAME}/{slot_name} ({slot_doc_id})")

    print("[PERM] Setting doc editors...")
    drive_grant_editor_access(drive, slot_doc_id, EDITOR_EMAILS)

    try:
        dest_doc = docs_get_document(docs, slot_doc_id)
    except HttpError as e:
        content = (e.content or b"").decode("utf-8", errors="ignore")
        if e.resp.status == 403 and ("SERVICE_DISABLED" in content or "docs.googleapis.com" in content):
            print("\n[ERROR] Google Docs API is disabled for your Google Cloud project.")
            print("Enable it here:")
            print("https://console.developers.google.com/apis/api/docs.googleapis.com/overview")
            print("Then wait 1-5 minutes and run again.\n")
            raise
        raise

    first_tab_id = get_first_tab_id(dest_doc)
    used_first_tab = False
    found_any = False

    people = sorted(
        [
            f for f in drive_list_children(drive, slot_id, FOLDER_MIME, shared_drive_id)
            if (f.get("name") or "").strip() not in SKIP_PERSON_FOLDERS
        ],
        key=lambda x: (x.get("name") or "").lower(),
    )

    for person in people:
        person_name = person["name"]
        person_folder_id = person["id"]

        person_doc = drive_find_child(drive, person_folder_id, PERSON_DOC_NAME, GDOC_MIME, shared_drive_id)
        if not person_doc:
            print(f"  [SKIP] {person_name}: '{PERSON_DOC_NAME}' not found")
            continue

        found_any = True
        person_text = extract_all_text_from_doc(docs, person_doc["id"]).strip()
        if not person_text:
            person_text = "[EMPTY DOCUMENT]"

        if not used_first_tab:
            tab_id = first_tab_id
            set_tab_title(docs, slot_doc_id, tab_id, person_name)
            used_first_tab = True
        else:
            tab_id = add_tab(docs, slot_doc_id, person_name)

        payload = (
            f"{person_name}\n"
            f"Slot: {slot_name}\n"
            f"Source Doc: {PERSON_DOC_NAME}\n"
            f"Generated: {time.strftime('%Y-%m-%d %H:%M:%S')}\n"
            + ("=" * 90)
            + "\n\n"
            + person_text
            + "\n"
        )

        insert_text_into_tab(docs, slot_doc_id, tab_id, payload)
        print(f"  [OK] Added tab + content for {person_name}")
        time.sleep(0.2)

    if not found_any:
        insert_text_into_tab(
            docs,
            slot_doc_id,
            first_tab_id,
            "No person-level 'Deliverables Analysis' docs were found inside this slot.\n",
        )
        print("  [NOTE] No person docs found; wrote placeholder note.")

    print(f"[DONE] Slot '{slot_name}' updated.")
    print("Done.")

if __name__ == "__main__":
    main()