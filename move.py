from google.oauth2.service_account import Credentials
from googleapiclient.discovery import build

# =========================
# CONFIG
# =========================
SERVICE_ACCOUNT_FILE = "service-account.json"

PARENT_FOLDER_ID = "1GRwwb95GK2lv6XUM1gUn3X_j64v6dTaO"

SOURCE_FOLDER_NAME = "0. Re training Deliverables - 42 Candiates"
DEST_FOLDER_NAME = "9. Complete candidate"

CANDIDATES_TO_MOVE = [
    "Harshil Sanghavi",
    "Guru Sai Supriya",
    "Sai Ritvik",
    "SaiRam",
    "Shiva Kalivemula",
    "Sindooja",
    "Sony Arravena",
    "Soumya Piya",
    "Upender Mula",
    "V. Kaushik",
    "Vaishnavi Kaleru",
    "Vishwajit K",
    "Harshil Sanghavi"
]

# =========================
# AUTH
# =========================
SCOPES = ["https://www.googleapis.com/auth/drive"]

creds = Credentials.from_service_account_file(
    SERVICE_ACCOUNT_FILE,
    scopes=SCOPES,
)

drive = build("drive", "v3", credentials=creds)

# =========================
# HELPERS
# =========================
def find_folder(parent_id, name):
    query = (
        f"'{parent_id}' in parents and "
        f"name = '{name}' and "
        f"mimeType = 'application/vnd.google-apps.folder' and "
        f"trashed = false"
    )

    result = drive.files().list(
        q=query,
        fields="files(id, name)",
        supportsAllDrives=True,
        includeItemsFromAllDrives=True,
    ).execute()

    files = result.get("files", [])
    return files[0]["id"] if files else None


def find_child_folder(parent_id, name):
    return find_folder(parent_id, name)


def move_folder(folder_id, new_parent_id):
    # get current parents
    file = drive.files().get(
        fileId=folder_id,
        fields="parents",
        supportsAllDrives=True
    ).execute()

    previous_parents = ",".join(file.get("parents"))

    drive.files().update(
        fileId=folder_id,
        addParents=new_parent_id,
        removeParents=previous_parents,
        supportsAllDrives=True
    ).execute()


# =========================
# MAIN
# =========================
def main():
    print("Finding source and destination folders...")

    source_id = find_folder(PARENT_FOLDER_ID, SOURCE_FOLDER_NAME)
    dest_id = find_folder(PARENT_FOLDER_ID, DEST_FOLDER_NAME)

    if not source_id:
        raise Exception(f"Source folder not found: {SOURCE_FOLDER_NAME}")
    if not dest_id:
        raise Exception(f"Destination folder not found: {DEST_FOLDER_NAME}")

    print("Source ID:", source_id)
    print("Destination ID:", dest_id)

    moved = 0
    not_found = 0

    for name in CANDIDATES_TO_MOVE:
        print(f"\nProcessing: {name}")

        folder_id = find_child_folder(source_id, name)

        if not folder_id:
            print(f"  ❌ Not found in source")
            not_found += 1
            continue

        move_folder(folder_id, dest_id)
        print(f"  ✅ Moved successfully")
        moved += 1

    print("\n" + "=" * 80)
    print("MOVE COMPLETE")
    print("=" * 80)
    print(f"Moved: {moved}")
    print(f"Not found: {not_found}")


if __name__ == "__main__":
    main()