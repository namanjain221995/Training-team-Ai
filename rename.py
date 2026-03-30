import re
from google.oauth2.service_account import Credentials
from googleapiclient.discovery import build

# =========================================================
# CONFIG
# =========================================================
SERVICE_ACCOUNT_FILE = "service-account.json"

# 2026 folder ID
PARENT_FOLDER_ID = "1GRwwb95GK2lv6XUM1gUn3X_j64v6dTaO"

# Slot folder to scan
SOURCE_SLOT_FOLDER_NAME = "0. Re training Deliverables - 42 Candiates"

SCOPES = ["https://www.googleapis.com/auth/drive"]

# =========================================================
# Extension groups
# =========================================================
VIDEO_EXTS = {
    "mp4", "mov", "mkv", "avi", "wmv", "flv", "webm", "m4v", "3gp", "mpeg", "mpg"
}
IMAGE_EXTS = {
    "png", "jpg", "jpeg", "webp", "bmp", "gif", "tif", "tiff", "heic", "heif"
}
TEXT_EXTS = {"txt"}
DIAGRAM_EXTS = {"drawio"}

# =========================================================
# Folder names
# =========================================================
FOLDER_INTRO = "3. Introduction Video"
FOLDER_MOCK = "4. Mock Interview (First Call)"
FOLDER_PROJECT = "5. Project Scenarios"
FOLDER_NICHE = "6. 30 Questions Related to Their Niche"
FOLDER_RESUME = "7. 50 Questions Related to the Resume"
FOLDER_TOOLS = "8. Tools & Technology Videos"
FOLDER_SYSTEM_DESIGN = "9. System Design Video (with Draw.io)"
FOLDER_PERSONA = "10. Persona Video"
FOLDER_SMALL_TALK = "11. Small Talk"
FOLDER_JD = "12. JD Video"

# =========================================================
# Rename rules
# =========================================================
RENAME_RULES = {
    FOLDER_INTRO: {
        ("1", "video"): "Day3_Intro_<FullName>.<ext>",
    },
    FOLDER_MOCK: {
        ("1", "video"): "Day1_HR_<FullName>.<ext>",
    },
    FOLDER_PROJECT: {
        ("1", "video"): "Day2_Project_scenario_1_<FullName>.<ext>",
        ("2", "video"): "Day2_Project_scenario_2_<FullName>.<ext>",
        ("3", "video"): "Day2_Project_scenario_3_<FullName>.<ext>",
        ("4", "video"): "Day2_Project_scenario_4_<FullName>.<ext>",
        ("5", "video"): "Day2_Project_scenario_5_<FullName>.<ext>",
    },
    FOLDER_NICHE: {
        ("1", "video"): "Day1_Niche_<FullName>.<ext>",
    },
    FOLDER_RESUME: {
        ("1", "video"): "Day3_Resume_Mock_<FullName>.<ext>",
        ("2", "video"): "Day4_Resume_Mock_<FullName>.<ext>",
        ("3", "video"): "Day5_Resume_Mock_<FullName>.<ext>",
    },
    FOLDER_TOOLS: {
        ("1", "video"): "Day3_part1_Tools_System_<FullName>.<ext>",
        ("2", "video"): "Day3_part2_Tools_System_<FullName>.<ext>",
        ("3", "video"): "Day3_part3_Tools_System_<FullName>.<ext>",
        ("4", "video"): "Day3_Team_Structure_<FullName>.<ext>",
        ("4", "image"): "Day3_Team_Structure_<FullName>.<ext>",
        ("4", "drawio"): "Day3_Team_Structure_<FullName>.drawio",
    },
    FOLDER_SYSTEM_DESIGN: {
        ("1", "video"): "Day6_SystemDesign_Problem1_<FullName>.<ext>",
        ("1", "image"): "Day6_SystemDesign_Problem1_<FullName>.<ext>",
        ("1", "drawio"): "Day6_SystemDesign_Problem1_<FullName>.drawio",
    },
    FOLDER_PERSONA: {
        ("1", "video"): "Day4_Recruiter_<FullName>.<ext>",
        ("2", "video"): "Day4_Manager_<FullName>.<ext>",
        ("3", "video"): "Day4_Architect_<FullName>.<ext>",
    },
    FOLDER_SMALL_TALK: {
        ("1", "video"): "Day5_SmallTalk_<FullName>.<ext>",
    },
    FOLDER_JD: {
        ("1", "video"): "Day5_JD_Mapping_1_<FullName>.<ext>",
        ("1", "image"): "Day5_JD_Mapping_1_<FullName>.<ext>",
        ("1", "txt"): "Day5_JD_Mapping_1_<FullName>.txt",
        ("2", "video"): "Day5_JD_Mapping_2_<FullName>.<ext>",
        ("2", "image"): "Day5_JD_Mapping_2_<FullName>.<ext>",
        ("2", "txt"): "Day5_JD_Mapping_2_<FullName>.txt",
    },
}

# =========================================================
# AUTH
# =========================================================
creds = Credentials.from_service_account_file(
    SERVICE_ACCOUNT_FILE,
    scopes=SCOPES,
)
drive = build("drive", "v3", credentials=creds)

# =========================================================
# HELPERS
# =========================================================
def norm_name(s: str) -> str:
    s = (s or "").strip()
    s = re.sub(r"\s+", " ", s)
    return s

def underscored_name(s: str) -> str:
    s = norm_name(s)
    s = re.sub(r"[^A-Za-z0-9]+", "_", s).strip("_")
    return s

def classify_ext(ext: str):
    ext = (ext or "").lower()
    if ext in VIDEO_EXTS:
        return "video"
    if ext in IMAGE_EXTS:
        return "image"
    if ext in TEXT_EXTS:
        return "txt"
    if ext in DIAGRAM_EXTS:
        return "drawio"
    return None

def parse_simple_numbered_filename(file_name: str):
    # Matches: 1.mp4 / 2.png / 1.txt / 1.drawio
    m = re.fullmatch(r"\s*(\d+)\s*\.\s*([A-Za-z0-9]+)\s*", file_name or "")
    if not m:
        return None, None
    return m.group(1), m.group(2).lower()

def get_target_name(deliverable_folder: str, file_name: str, candidate_folder: str):
    rules = RENAME_RULES.get(deliverable_folder)
    if not rules:
        return None

    number, ext = parse_simple_numbered_filename(file_name)
    if not number or not ext:
        return None

    ext_group = classify_ext(ext)
    if not ext_group:
        return None

    template = rules.get((number, ext_group))
    if not template:
        return None

    candidate_token = underscored_name(candidate_folder)
    return template.replace("<FullName>", candidate_token).replace("<ext>", ext)

def list_child_folders(parent_id: str):
    all_files = []
    page_token = None

    while True:
        resp = drive.files().list(
            q=(
                f"'{parent_id}' in parents and "
                f"mimeType='application/vnd.google-apps.folder' and "
                f"trashed=false"
            ),
            fields="nextPageToken, files(id, name)",
            supportsAllDrives=True,
            includeItemsFromAllDrives=True,
            pageSize=1000,
            pageToken=page_token
        ).execute()

        all_files.extend(resp.get("files", []))
        page_token = resp.get("nextPageToken")
        if not page_token:
            break

    return all_files

def list_child_files(parent_id: str):
    all_files = []
    page_token = None

    while True:
        resp = drive.files().list(
            q=(
                f"'{parent_id}' in parents and "
                f"mimeType!='application/vnd.google-apps.folder' and "
                f"trashed=false"
            ),
            fields="nextPageToken, files(id, name, mimeType)",
            supportsAllDrives=True,
            includeItemsFromAllDrives=True,
            pageSize=1000,
            pageToken=page_token
        ).execute()

        all_files.extend(resp.get("files", []))
        page_token = resp.get("nextPageToken")
        if not page_token:
            break

    return all_files

def find_folder(parent_id: str, name: str):
    resp = drive.files().list(
        q=(
            f"'{parent_id}' in parents and "
            f"name='{name}' and "
            f"mimeType='application/vnd.google-apps.folder' and "
            f"trashed=false"
        ),
        fields="files(id, name)",
        supportsAllDrives=True,
        includeItemsFromAllDrives=True,
        pageSize=10
    ).execute()

    files = resp.get("files", [])
    return files[0] if files else None

def rename_file(file_id: str, new_name: str):
    drive.files().update(
        fileId=file_id,
        body={"name": new_name},
        supportsAllDrives=True
    ).execute()

# =========================================================
# MAIN
# =========================================================
def main():
    slot = find_folder(PARENT_FOLDER_ID, SOURCE_SLOT_FOLDER_NAME)
    if not slot:
        raise Exception(f"Slot folder not found: {SOURCE_SLOT_FOLDER_NAME}")

    slot_id = slot["id"]
    print(f"Found slot folder: {SOURCE_SLOT_FOLDER_NAME}")
    print(f"Slot ID: {slot_id}")

    candidate_folders = list_child_folders(slot_id)

    total_renamed = 0
    renamed_candidates = set()

    for candidate in candidate_folders:
        candidate_id = candidate["id"]
        candidate_name = candidate["name"]
        candidate_had_rename = False

        deliverable_folders = list_child_folders(candidate_id)

        for deliverable in deliverable_folders:
            deliverable_id = deliverable["id"]
            deliverable_name = deliverable["name"]

            if deliverable_name not in RENAME_RULES:
                continue

            files = list_child_files(deliverable_id)

            for file_obj in files:
                file_id = file_obj["id"]
                old_name = file_obj["name"]

                target_name = get_target_name(deliverable_name, old_name, candidate_name)
                if not target_name:
                    continue

                if old_name == target_name:
                    continue

                print(f"[RENAMED] {candidate_name} | {deliverable_name} | {old_name} -> {target_name}")
                rename_file(file_id, target_name)

                total_renamed += 1
                candidate_had_rename = True

        if candidate_had_rename:
            renamed_candidates.add(candidate_name)

    print("\n" + "=" * 90)
    print("RENAME COMPLETE")
    print("=" * 90)
    print(f"Total files renamed: {total_renamed}")
    print(f"Candidates with renamed files: {len(renamed_candidates)}")

    if renamed_candidates:
        print("\nCandidate names:")
        for name in sorted(renamed_candidates):
            print(name)
    else:
        print("\nNo candidate files needed renaming.")

if __name__ == "__main__":
    main()