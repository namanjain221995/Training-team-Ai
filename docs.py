import os
import time
from pathlib import Path
from typing import Optional, List, Dict, Any

from dotenv import load_dotenv

from googleapiclient.discovery import build
from googleapiclient.errors import HttpError
from google.oauth2.credentials import Credentials
from google_auth_oauthlib.flow import InstalledAppFlow
from google.auth.transport.requests import Request

# =========================
# ENV
# =========================
load_dotenv()

# Non-interactive slot selection (Option B)
# Example: SLOT_CHOICE=2
SLOT_CHOICE = (os.getenv("SLOT_CHOICE") or "").strip()

# Shared drives flag from env (optional)
USE_SHARED_DRIVES = (os.getenv("USE_SHARED_DRIVES") or "").strip().lower() in ("1", "true", "yes", "y")

# =========================
# CONFIG
# =========================
SCOPES = [
    "https://www.googleapis.com/auth/drive",
    "https://www.googleapis.com/auth/documents",
]
CREDENTIALS_FILE = Path("credentials.json")
TOKEN_FILE = Path("token.json")

# SOURCE: contains slot folders (read from here)
ROOT_2026_FOLDER_NAME = "2026"

# DESTINATION: must exist anywhere in Drive. Script will NOT create it.
OUTPUT_ROOT_FOLDER_NAME = "Candidate Result"

FOLDER_MIME = "application/vnd.google-apps.folder"
GDOC_MIME = "application/vnd.google-apps.document"

PERSON_DOC_NAME = "Deliverables Analysis"
SLOT_DOC_NAME = "All Deliverables Analysis"

# Skip folders under 2026/<Slot>/ that are NOT people folders
SKIP_PERSON_FOLDERS = {"1. Format"}

# Give editor access to these emails (folder + doc)
EDITOR_EMAILS = [
    "rajvi.patel@techsarasolutions.com",
    "sahil.patel@techsarasolutions.com",
    "soham.piprotar@techsarasolutions.com",
]


# =========================
# Auth
# =========================
def get_creds() -> Credentials:
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

    return creds


def get_drive_service(creds: Credentials):
    return build("drive", "v3", credentials=creds)


def get_docs_service(creds: Credentials):
    return build("docs", "v1", credentials=creds)


def _list_kwargs() -> Dict[str, Any]:
    """
    Add shared-drive flags to every Drive API call when USE_SHARED_DRIVES=True.
    Safe to pass to create/list/delete/permissions.
    """
    if USE_SHARED_DRIVES:
        return {
            "supportsAllDrives": True,
            "includeItemsFromAllDrives": True,
            "corpora": "allDrives",
        }
    return {}


# =========================
# Drive helpers
# =========================
def _escape_drive_q_value(s: str) -> str:
    # Drive query string escaping: single quote is escaped by doubling it
    return s.replace("'", "''")


def drive_search_folder_anywhere(drive, folder_name: str) -> List[dict]:
    safe = _escape_drive_q_value(folder_name)
    q = f"name = '{safe}' and mimeType = '{FOLDER_MIME}' and trashed=false"
    res = drive.files().list(
        q=q,
        fields="files(id,name,parents,modifiedTime)",
        pageSize=200,
        **_list_kwargs(),
    ).execute()
    return res.get("files", []) or []


def pick_best_named_folder(candidates: List[dict]) -> dict:
    return sorted(candidates, key=lambda c: c.get("modifiedTime") or "", reverse=True)[0]


def drive_list_children(drive, parent_id: str, mime_type: Optional[str] = None):
    q_parts = [f"'{parent_id}' in parents", "trashed = false"]
    if mime_type:
        q_parts.append(f"mimeType = '{mime_type}'")
    q = " and ".join(q_parts)

    page_token = None
    while True:
        res = drive.files().list(
            q=q,
            fields="nextPageToken, files(id,name,mimeType,modifiedTime)",
            pageSize=1000,
            pageToken=page_token,
            **_list_kwargs(),
        ).execute()

        for f in res.get("files", []):
            yield f

        page_token = res.get("nextPageToken")
        if not page_token:
            break


def drive_find_children(drive, parent_id: str, name: str, mime_type: Optional[str] = None) -> List[dict]:
    safe = _escape_drive_q_value(name)
    q_parts = [f"'{parent_id}' in parents", "trashed = false", f"name = '{safe}'"]
    if mime_type:
        q_parts.append(f"mimeType = '{mime_type}'")
    q = " and ".join(q_parts)

    res = drive.files().list(
        q=q,
        fields="files(id,name,mimeType,parents,modifiedTime)",
        pageSize=200,
        **_list_kwargs(),
    ).execute()
    return res.get("files", []) or []


def drive_find_child(drive, parent_id: str, name: str, mime_type: Optional[str] = None) -> Optional[dict]:
    files = drive_find_children(drive, parent_id, name, mime_type)
    if not files:
        return None
    return sorted(files, key=lambda f: f.get("modifiedTime") or "", reverse=True)[0]


def drive_delete_file(drive, file_id: str):
    drive.files().delete(fileId=file_id, **_list_kwargs()).execute()


def drive_delete_all_named(drive, parent_id: str, name: str, mime_type: Optional[str] = None) -> int:
    matches = drive_find_children(drive, parent_id, name, mime_type)
    for f in matches:
        drive_delete_file(drive, f["id"])
    return len(matches)


def drive_create_folder(drive, parent_id: str, name: str) -> str:
    meta = {"name": name, "mimeType": FOLDER_MIME, "parents": [parent_id]}
    created = drive.files().create(body=meta, fields="id", **_list_kwargs()).execute()
    return created["id"]


def drive_create_gdoc(drive, parent_id: str, name: str) -> str:
    meta = {"name": name, "mimeType": GDOC_MIME, "parents": [parent_id]}
    created = drive.files().create(body=meta, fields="id", **_list_kwargs()).execute()
    return created["id"]


def drive_grant_editor_access(drive, file_id: str, emails: List[str]):
    """
    Grants editor permission (writer) to given emails for a Drive file/folder.
    Does NOT send notification emails.
    """
    for email in emails:
        perm = {
            "type": "user",
            "role": "writer",
            "emailAddress": email,
        }
        try:
            drive.permissions().create(
                fileId=file_id,
                body=perm,
                sendNotificationEmail=False,
                **_list_kwargs(),
            ).execute()
            print(f"  [PERM] Editor added: {email}")
        except HttpError as e:
            msg = (e.content or b"").decode("utf-8", errors="ignore").lower()
            if "alreadyexists" in msg or "already exists" in msg or "duplicate" in msg:
                print(f"  [PERM] Already has access: {email}")
            else:
                print(f"  [PERM] Failed for {email}: {e}")


# =========================
# SLOT SELECTION (supports SLOT_CHOICE env)
# =========================
def list_slot_folders(drive, slots_parent_id: str) -> List[dict]:
    return sorted(
        list(drive_list_children(drive, slots_parent_id, FOLDER_MIME)),
        key=lambda x: (x.get("name") or "").lower(),
    )


def choose_slot(drive, slots_parent_id: str) -> dict:
    slots = list_slot_folders(drive, slots_parent_id)
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
# Docs helpers (tabs + text)
# =========================
def docs_get_document(docs, document_id: str) -> Dict[str, Any]:
    return docs.documents().get(documentId=document_id, includeTabsContent=True).execute()


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
    return docs.documents().batchUpdate(
        documentId=document_id,
        body={"requests": requests},
    ).execute()


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
    docs_batch_update(docs, document_id, [{"addDocumentTab": {"tabProperties": {"title": title}}}])
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
    creds = get_creds()
    drive = get_drive_service(creds)
    docs = get_docs_service(creds)

    # --- Find SOURCE 2026 folder (contains slots) ---
    candidates_2026 = drive_search_folder_anywhere(drive, ROOT_2026_FOLDER_NAME)
    if not candidates_2026:
        raise RuntimeError(f"Could not find folder '{ROOT_2026_FOLDER_NAME}' anywhere in Drive.")
    base_2026 = pick_best_named_folder(candidates_2026)
    base_2026_id = base_2026["id"]

    # --- Find OUTPUT ROOT folder in Drive (Candidate Result) ---
    candidates_out = drive_search_folder_anywhere(drive, OUTPUT_ROOT_FOLDER_NAME)
    if not candidates_out:
        raise RuntimeError(
            f"Could not find output folder '{OUTPUT_ROOT_FOLDER_NAME}' anywhere in Drive. "
            f"Create it and run again."
        )
    output_root = pick_best_named_folder(candidates_out)
    output_root_id = output_root["id"]

    # --- Choose which slot to process (AUTO if SLOT_CHOICE provided) ---
    slot = choose_slot(drive, base_2026_id)
    slot_name = slot["name"]
    slot_id = slot["id"]

    # --- Create/Use per-slot output folder under Candidate Result ---
    slot_output_folder = drive_find_child(drive, output_root_id, slot_name, FOLDER_MIME)
    if not slot_output_folder:
        slot_output_folder_id = drive_create_folder(drive, output_root_id, slot_name)
        print(f"[OUTPUT] Created folder: {OUTPUT_ROOT_FOLDER_NAME}/{slot_name}")
    else:
        slot_output_folder_id = slot_output_folder["id"]

    # Give editor access on the per-slot output folder
    print("[PERM] Setting folder editors...")
    drive_grant_editor_access(drive, slot_output_folder_id, EDITOR_EMAILS)

    # --- Create slot-level doc inside Candidate Result/<SlotName>/ ---
    deleted = drive_delete_all_named(drive, slot_output_folder_id, SLOT_DOC_NAME, GDOC_MIME)
    if deleted:
        print(f"[OUTPUT] Deleted {deleted} existing '{SLOT_DOC_NAME}' doc(s) in {OUTPUT_ROOT_FOLDER_NAME}/{slot_name}")

    slot_doc_id = drive_create_gdoc(drive, slot_output_folder_id, SLOT_DOC_NAME)
    print(f"\n[OUTPUT] Created '{SLOT_DOC_NAME}' in {OUTPUT_ROOT_FOLDER_NAME}/{slot_name} ({slot_doc_id})")

    # Give editor access on the doc
    print("[PERM] Setting doc editors...")
    drive_grant_editor_access(drive, slot_doc_id, EDITOR_EMAILS)

    # ---- Docs API sanity check + friendly disabled message ----
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

    # --- Read people folders from SOURCE slot folder 2026/<SlotName>/ ---
    people = sorted(
        [
            f
            for f in drive_list_children(drive, slot_id, FOLDER_MIME)
            if (f.get("name") or "").strip() not in SKIP_PERSON_FOLDERS
        ],
        key=lambda x: (x.get("name") or "").lower(),
    )

    for person in people:
        person_name = person["name"]
        person_folder_id = person["id"]

        person_doc = drive_find_child(drive, person_folder_id, PERSON_DOC_NAME, GDOC_MIME)
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
