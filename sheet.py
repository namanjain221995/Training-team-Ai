import io
import re
import time
import json
import tempfile
import os
from pathlib import Path
from typing import Optional, Dict, List, Tuple, Any

from dotenv import load_dotenv

from openpyxl import Workbook
from openpyxl.styles import Font, Alignment, PatternFill
from openpyxl.utils import get_column_letter

from googleapiclient.discovery import build
from googleapiclient.http import MediaIoBaseDownload, MediaFileUpload
from googleapiclient.errors import HttpError
from google.oauth2.credentials import Credentials
from google_auth_oauthlib.flow import InstalledAppFlow
from google.auth.transport.requests import Request

from openai import OpenAI

# =========================
# ENV
# =========================
load_dotenv()

SLOT_CHOICE = (os.getenv("SLOT_CHOICE") or "").strip()
USE_SHARED_DRIVES = (os.getenv("USE_SHARED_DRIVES") or "").strip().lower() in ("1", "true", "yes", "y")
DEFAULT_OPENAI_MODEL = (os.getenv("OPENAI_MODEL") or "").strip() or "gpt-5.2-mini"

# Headless auth for servers/docker (copy/paste code)
HEADLESS_AUTH = (os.getenv("HEADLESS_AUTH") or "").strip().lower() in ("1", "true", "yes", "y")

# =========================
# CONFIG
# =========================
SCOPES = ["https://www.googleapis.com/auth/drive"]
CREDENTIALS_FILE = Path("credentials.json")
TOKEN_FILE = Path("token.json")

ROOT_SLOTS_FOLDER_NAME = "2026"               # SOURCE root (contains slot folders)
OUTPUT_ROOT_FOLDER_NAME = "Candidate Result" # DEST root (must exist)

FOLDER_MIME = "application/vnd.google-apps.folder"
XLSX_MIME = "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"

# Update these names to match your Drive deliverable folder names EXACTLY
FOLDER_NAMES = [
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

AVG_HEADER = "Average % (available)"
OUTPUT_XLSX_NAME = "Deliverables Analysis Sheet.xlsx"

SKIP_PERSON_FOLDERS = {"1. Format"}
SKIP_DELIVERABLE_FOLDERS = {"1. Format"}

LLM_OUTPUT_FILE_RE = re.compile(r"^LLM_OUTPUT__.*(\.txt)?$", re.I)

FAIL_PASS_PCT_RX = re.compile(
    r"final\s*overall\s*score\s*[:\-]\s*(pass|fail)\s*(?:\(\s*([0-9]+(?:\.[0-9]+)?)\s*%\s*\)|[:\-]?\s*([0-9]+(?:\.[0-9]+)?)\s*%?)",
    re.I,
)

SCORE_REGEXES = [
    re.compile(r"final\s*overall\s*score\s*[:\-]\s*([0-9]+(?:\.[0-9]+)?)\s*(/10|\/\s*10|out\s*of\s*10|%)?", re.I),
    re.compile(r"final\s*score\s*[:\-]\s*([0-9]+(?:\.[0-9]+)?)\s*(/10|\/\s*10|out\s*of\s*10|%)?", re.I),
]

MAX_CHARS_TO_MODEL = 80_000
PCT_MIN = 0.0
PCT_MAX = 100.0

SLEEP_BETWEEN_UPLOADS_SEC = 0.25

EDITOR_EMAILS = [
    "rajvi.patel@techsarasolutions.com",
    "sahil.patel@techsarasolutions.com",
    "soham.piprotar@techsarasolutions.com",
]

# =========================
# ✅ Thresholds (PERCENT)
# =========================
THRESHOLDS_PCT: Dict[str, float] = {
    "3. Introduction Video": 75.0,
    "4. Mock Interview (First Call)": 90.0,
    "5. Project Scenarios": 80.0,
    "6. 30 Questions Related to Their Niche": 60.0,
    "7. 50 Questions Related to the Resume": 80.0,
    "8. Tools & Technology Videos": 70.0,
    "9. System Design Video (with Draw.io)": 80.0,
    "10. Persona Video": 70.0,
    "11. Small Talk": 75.0,
    "12. JD Video": 75.0,
    AVG_HEADER: 75.0,
}

GREEN_FILL = PatternFill(start_color="C6EFCE", end_color="C6EFCE", fill_type="solid")

# =========================
# OpenAI
# =========================
def init_openai_client() -> OpenAI:
    api_key = (os.getenv("OPENAI_API_KEY") or "").strip()
    if not api_key:
        raise RuntimeError("OPENAI_API_KEY not found in .env or environment variables.")
    return OpenAI(api_key=api_key)

def safe_response_output_text(resp) -> str:
    # Works across SDK variations
    if hasattr(resp, "output_text") and resp.output_text:
        return resp.output_text
    out = getattr(resp, "output", None) or []
    parts = []
    for item in out:
        for c in (getattr(item, "content", None) or []):
            t = getattr(c, "text", None)
            if t:
                parts.append(t)
    return "\n".join(parts).strip()

# =========================
# Drive auth
# =========================
def get_drive_service():
    creds = None
    if TOKEN_FILE.exists():
        creds = Credentials.from_authorized_user_file(str(TOKEN_FILE), SCOPES)

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

            # Server/Docker safe: copy/paste auth code
            if HEADLESS_AUTH or not os.environ.get("DISPLAY"):
                creds = flow.run_console()
            else:
                creds = flow.run_local_server(port=0)

            TOKEN_FILE.write_text(creds.to_json(), encoding="utf-8")

    return build("drive", "v3", credentials=creds)

# =========================
# Shared Drives kwargs (SAFE per method)
# =========================
def _kwargs_for_list() -> Dict[str, Any]:
    if USE_SHARED_DRIVES:
        return {
            "supportsAllDrives": True,
            "includeItemsFromAllDrives": True,
            "corpora": "allDrives",
        }
    return {}

def _kwargs_for_mutation() -> Dict[str, Any]:
    if USE_SHARED_DRIVES:
        return {"supportsAllDrives": True}
    return {}

def _get_media_kwargs() -> Dict[str, Any]:
    if USE_SHARED_DRIVES:
        return {"supportsAllDrives": True}
    return {}

# =========================
# Drive helpers
# =========================
def _escape_drive_q_value(s: str) -> str:
    return s.replace("\\", "\\\\").replace("'", "\\'")

def drive_list_children(service, parent_id: str, mime_type: Optional[str] = None):
    q_parts = [f"'{parent_id}' in parents", "trashed = false"]
    if mime_type:
        q_parts.append(f"mimeType = '{mime_type}'")
    q = " and ".join(q_parts)

    page_token = None
    while True:
        res = (
            service.files()
            .list(
                q=q,
                fields="nextPageToken, files(id,name,mimeType,modifiedTime,size)",
                pageSize=1000,
                pageToken=page_token,
                **_kwargs_for_list(),
            )
            .execute()
        )
        for f in res.get("files", []):
            yield f
        page_token = res.get("nextPageToken")
        if not page_token:
            break

def drive_find_child(service, parent_id: str, name: str, mime_type: Optional[str] = None) -> Optional[dict]:
    safe = _escape_drive_q_value(name)
    q_parts = [f"'{parent_id}' in parents", "trashed = false", f"name = '{safe}'"]
    if mime_type:
        q_parts.append(f"mimeType = '{mime_type}'")
    q = " and ".join(q_parts)

    res = (
        service.files()
        .list(
            q=q,
            fields="files(id,name,mimeType,parents,modifiedTime)",
            pageSize=50,
            **_kwargs_for_list(),
        )
        .execute()
    )
    files = res.get("files", []) or []
    if not files:
        return None
    return sorted(files, key=lambda f: f.get("modifiedTime") or "", reverse=True)[0]

def drive_create_folder(service, parent_id: str, name: str) -> str:
    meta = {"name": name, "mimeType": FOLDER_MIME, "parents": [parent_id]}
    created = service.files().create(body=meta, fields="id", **_kwargs_for_mutation()).execute()
    return created["id"]

def drive_download_text(service, file_id: str, mime_type: Optional[str]) -> str:
    fh = io.BytesIO()
    try:
        if mime_type and mime_type.startswith("application/vnd.google-apps."):
            request = service.files().export_media(fileId=file_id, mimeType="text/plain", **_get_media_kwargs())
        else:
            request = service.files().get_media(fileId=file_id, **_get_media_kwargs())

        downloader = MediaIoBaseDownload(fh, request, chunksize=1024 * 1024 * 4)
        done = False
        while not done:
            _, done = downloader.next_chunk()
        fh.seek(0)
        raw = fh.read()
    except HttpError as e:
        print(f"     [WARN] Could not download fileId={file_id}: {e}")
        return ""

    try:
        return raw.decode("utf-8")
    except Exception:
        return raw.decode("latin-1", errors="ignore")

def drive_upload_xlsx(service, parent_id: str, filename: str, local_path: Path) -> str:
    # pick latest if duplicates exist
    matches = [
        f for f in drive_list_children(service, parent_id, None)
        if f.get("name") == filename and f.get("mimeType") != FOLDER_MIME
    ]
    existing = None
    if matches:
        existing = sorted(matches, key=lambda x: (x.get("modifiedTime") or ""), reverse=True)[0]

    media = MediaFileUpload(str(local_path), mimetype=XLSX_MIME, resumable=True)

    if existing:
        service.files().update(fileId=existing["id"], media_body=media, **_kwargs_for_mutation()).execute()
        return existing["id"]
    else:
        meta = {"name": filename, "parents": [parent_id]}
        created = service.files().create(body=meta, media_body=media, fields="id", **_kwargs_for_mutation()).execute()
        return created["id"]

def drive_search_folder_anywhere(service, folder_name: str) -> List[dict]:
    safe_name = _escape_drive_q_value(folder_name)
    q = f"name = '{safe_name}' and mimeType = '{FOLDER_MIME}' and trashed=false"
    res = (
        service.files()
        .list(
            q=q,
            fields="files(id,name,parents,modifiedTime)",
            pageSize=200,
            **_kwargs_for_list(),
        )
        .execute()
    )
    return res.get("files", []) or []

def pick_best_named_folder(candidates: List[dict]) -> dict:
    return sorted(candidates, key=lambda c: (c.get("modifiedTime") or ""), reverse=True)[0]

def drive_grant_editor_access(service, file_id: str, emails: List[str]):
    for email in emails:
        perm = {"type": "user", "role": "writer", "emailAddress": email}
        try:
            service.permissions().create(
                fileId=file_id,
                body=perm,
                sendNotificationEmail=False,
                **_kwargs_for_mutation(),
            ).execute()
            print(f"  [PERM] Editor added: {email}")
        except HttpError as e:
            msg = (e.content or b"").decode("utf-8", errors="ignore").lower()
            if "alreadyexists" in msg or "already exists" in msg or "duplicate" in msg:
                print(f"  [PERM] Already has access: {email}")
            else:
                print(f"  [PERM] Failed for {email}: {e}")
        time.sleep(0.08)  # small throttle to avoid permission quota spikes

# =========================
# SLOT SELECTION
# =========================
def list_slot_folders(service, slots_parent_id: str) -> List[dict]:
    return sorted(
        list(drive_list_children(service, slots_parent_id, FOLDER_MIME)),
        key=lambda x: (x.get("name") or "").lower(),
    )

def choose_slot(service, slots_parent_id: str) -> dict:
    slots = list_slot_folders(service, slots_parent_id)
    if not slots:
        raise RuntimeError(f"No slot folders found under '{ROOT_SLOTS_FOLDER_NAME}'.")

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
# Score extraction -> PERCENT
# =========================
def clamp_pct(x: Optional[float]) -> Optional[float]:
    if x is None:
        return None
    if x < PCT_MIN or x > PCT_MAX:
        return None
    return x

def normalize_to_percent(value: float, unit: Optional[str]) -> Optional[float]:
    u = (unit or "").lower().strip()

    if "%" in u:
        return value
    if "/10" in u or "out of 10" in u:
        return value * 10.0

    # Heuristic if unit missing:
    if value <= 10.0:
        return value * 10.0
    if 10.0 < value <= 100.0:
        return value
    return None

def extract_score_regex_percent(text: str) -> Optional[float]:
    if not text:
        return None

    m = FAIL_PASS_PCT_RX.search(text)
    if m:
        pct_str = m.group(2) or m.group(3)
        try:
            return clamp_pct(round(float(pct_str), 2))
        except Exception:
            return None

    for rx in SCORE_REGEXES:
        m = rx.search(text)
        if m:
            try:
                val = float(m.group(1))
                unit = m.group(2) if m.lastindex and m.lastindex >= 2 else None
                pct = normalize_to_percent(val, unit)
                return clamp_pct(round(pct, 2)) if pct is not None else None
            except Exception:
                return None

    return None

def extract_score_openai_percent(client: OpenAI, text: str) -> Optional[float]:
    if not text:
        return None

    snippet = text[:MAX_CHARS_TO_MODEL]

    schema = {
        "name": "score_extraction_percent",
        "schema": {
            "type": "object",
            "additionalProperties": False,
            "properties": {
                "score_value": {"type": ["number", "null"]},
                "score_unit": {"type": "string", "enum": ["percent", "out_of_10", "unknown"]},
                "evidence": {"type": "string"},
            },
            "required": ["score_value", "score_unit", "evidence"],
        },
        "strict": True,
    }

    system_prompt = (
        "Extract the FINAL OVERALL SCORE from the given text.\n"
        "Examples:\n"
        "- 'Final Overall Score: 8/10'\n"
        "- 'Final Overall Score: 60%'\n"
        "- 'Final Overall Score: FAIL (30%)'\n"
        "Return score_value and score_unit.\n"
        "If no final score exists, score_value must be null.\n"
        "Do not guess."
    )

    try:
        resp = client.responses.create(
            model=DEFAULT_OPENAI_MODEL,
            input=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": snippet},
            ],
            text={"format": {"type": "json_schema", "json_schema": schema}},
            temperature=0,
        )
    except Exception:
        return None

    try:
        data = json.loads(safe_response_output_text(resp))
    except Exception:
        return None

    val = data.get("score_value", None)
    unit = data.get("score_unit", "unknown")
    if val is None:
        return None

    try:
        val = float(val)
    except Exception:
        return None

    unit_hint = "%" if unit == "percent" else ("/10" if unit == "out_of_10" else None)
    pct = normalize_to_percent(val, unit_hint)
    return clamp_pct(round(pct, 2)) if pct is not None else None

def pick_best_score_from_llm_outputs_percent(
    service,
    openai_client: OpenAI,
    files: List[dict],
) -> Tuple[Optional[float], Optional[str]]:
    llm_files = [
        f for f in files
        if f.get("mimeType") != FOLDER_MIME and LLM_OUTPUT_FILE_RE.match(f.get("name", ""))
    ]
    if not llm_files:
        return None, None

    llm_files.sort(key=lambda x: (x.get("modifiedTime") or ""), reverse=True)

    for f in llm_files:
        text = drive_download_text(service, f["id"], f.get("mimeType"))

        s = extract_score_regex_percent(text)
        if s is not None:
            return s, f.get("name")

        s = extract_score_openai_percent(openai_client, text)
        if s is not None:
            return s, f.get("name")

    return None, None

# =========================
# Excel creation
# =========================
def autosize_columns(ws):
    for col in ws.columns:
        max_len = 0
        col_letter = get_column_letter(col[0].column)
        for cell in col:
            v = "" if cell.value is None else str(cell.value)
            max_len = max(max_len, len(v))
        ws.column_dimensions[col_letter].width = min(max(12, max_len + 2), 60)

def set_percent_cell(cell, pct_value: Optional[float]):
    """
    Store numeric value as fraction so Excel treats it as a real percent.
    pct_value is 0..100 (e.g., 75.5)
    """
    if pct_value is None:
        cell.value = None
        return
    cell.value = float(pct_value) / 100.0  # fraction 0..1
    cell.number_format = "0%"              # change to "0.00%" if you want decimals
    cell.alignment = Alignment(horizontal="center", vertical="center")

def build_slot_workbook_percent(rows: List[Dict[str, Any]]) -> Workbook:
    wb = Workbook()
    ws = wb.active
    ws.title = "Deliverables Analysis Sheet"

    headers = ["Person Name"] + FOLDER_NAMES + [AVG_HEADER]
    ws.append(headers)

    # Header style
    for c in range(1, len(headers) + 1):
        cell = ws.cell(row=1, column=c)
        cell.font = Font(bold=True)
        cell.alignment = Alignment(horizontal="center", vertical="center", wrap_text=True)

    # Map header -> column index
    header_to_col = {h: i + 1 for i, h in enumerate(headers)}

    for r in rows:
        person = r["person"]
        ws.append([person] + [None] * (len(headers) - 1))
        row_idx = ws.max_row

        # Fill deliverable columns
        for folder_name in FOLDER_NAMES:
            col_idx = header_to_col[folder_name]
            pct = r.get(folder_name)
            cell = ws.cell(row=row_idx, column=col_idx)
            set_percent_cell(cell, pct)

            # ✅ Green if meets threshold
            thr = THRESHOLDS_PCT.get(folder_name)
            if thr is not None and isinstance(pct, (int, float)) and float(pct) >= float(thr):
                cell.fill = GREEN_FILL

        # Compute average of available numeric scores
        scores_pct = [r.get(folder) for folder in FOLDER_NAMES]
        numeric = [s for s in scores_pct if isinstance(s, (int, float))]
        avg = round(sum(numeric) / len(numeric), 2) if numeric else None

        # Fill average column
        avg_col = header_to_col[AVG_HEADER]
        avg_cell = ws.cell(row=row_idx, column=avg_col)
        set_percent_cell(avg_cell, avg)

        thr_avg = THRESHOLDS_PCT.get(AVG_HEADER)
        if thr_avg is not None and isinstance(avg, (int, float)) and float(avg) >= float(thr_avg):
            avg_cell.fill = GREEN_FILL

        # Person name alignment
        ws.cell(row=row_idx, column=1).alignment = Alignment(horizontal="left", vertical="center")

    ws.freeze_panes = "A2"
    autosize_columns(ws)
    return wb

# =========================
# MAIN
# =========================
def main():
    openai_client = init_openai_client()
    service = get_drive_service()

    candidates_root = drive_search_folder_anywhere(service, ROOT_SLOTS_FOLDER_NAME)
    if not candidates_root:
        raise RuntimeError(f"Could not find folder '{ROOT_SLOTS_FOLDER_NAME}' anywhere in Drive.")
    base_root = pick_best_named_folder(candidates_root)
    slots_parent_id = base_root["id"]

    candidates_out = drive_search_folder_anywhere(service, OUTPUT_ROOT_FOLDER_NAME)
    if not candidates_out:
        raise RuntimeError(
            f"Could not find output folder '{OUTPUT_ROOT_FOLDER_NAME}' anywhere in Drive. Create it and run again."
        )
    output_root = pick_best_named_folder(candidates_out)
    output_root_id = output_root["id"]

    slot = choose_slot(service, slots_parent_id)
    slot_name = slot["name"]
    slot_id = slot["id"]

    print(f"\n=== SLOT (SOURCE): {slot_name} ===")

    slot_out = drive_find_child(service, output_root_id, slot_name, FOLDER_MIME)
    if not slot_out:
        slot_out_id = drive_create_folder(service, output_root_id, slot_name)
        print(f"[OUTPUT] Created folder: {OUTPUT_ROOT_FOLDER_NAME}/{slot_name}")
    else:
        slot_out_id = slot_out["id"]

    print("[PERM] Setting folder editors...")
    drive_grant_editor_access(service, slot_out_id, EDITOR_EMAILS)

    people_all = sorted(
        list(drive_list_children(service, slot_id, FOLDER_MIME)),
        key=lambda x: (x.get("name") or "").lower(),
    )
    people = [p for p in people_all if (p.get("name") or "") not in SKIP_PERSON_FOLDERS]

    slot_rows: List[Dict[str, Any]] = []

    for person in people:
        person_name = person["name"]
        print(f" - Person: {person_name}")
        row: Dict[str, Any] = {"person": person_name}

        person_child_folders = list(drive_list_children(service, person["id"], FOLDER_MIME))
        folder_map = {
            f["name"]: f
            for f in person_child_folders
            if (f.get("name") or "") not in SKIP_DELIVERABLE_FOLDERS
        }

        for folder_name in FOLDER_NAMES:
            folder_node = folder_map.get(folder_name)
            if not folder_node:
                row[folder_name] = None
                print(f"   * {folder_name}: folder missing")
                continue

            files = list(drive_list_children(service, folder_node["id"], None))
            score_pct, used_filename = pick_best_score_from_llm_outputs_percent(service, openai_client, files)
            row[folder_name] = score_pct

            if score_pct is None:
                print(f"   * {folder_name}: no score found in any LLM_OUTPUT__* file")
            else:
                print(f"   * {folder_name}: {score_pct}% (from {used_filename})")

        slot_rows.append(row)

    wb = build_slot_workbook_percent(slot_rows)

    with tempfile.TemporaryDirectory() as td:
        td = Path(td)
        local_xlsx = td / OUTPUT_XLSX_NAME
        wb.save(local_xlsx)

        uploaded_file_id = drive_upload_xlsx(service, slot_out_id, OUTPUT_XLSX_NAME, local_xlsx)
        print(f"[OK] Uploaded: {OUTPUT_ROOT_FOLDER_NAME}/{slot_name}/{OUTPUT_XLSX_NAME}")

        print("[PERM] Setting sheet editors...")
        drive_grant_editor_access(service, uploaded_file_id, EDITOR_EMAILS)

    time.sleep(SLEEP_BETWEEN_UPLOADS_SEC)

if __name__ == "__main__":
    main()
