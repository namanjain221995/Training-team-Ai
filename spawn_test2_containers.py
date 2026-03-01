import os
import time
import docker
from dotenv import load_dotenv

load_dotenv()

from test2 import (
    get_drive_service,
    drive_search_folder_anywhere_in_my_drive,
    choose_slot,
    drive_list_children,
    drive_find_child,
    is_video_name,
    transcript_output_name,
    FOLDER_MIME,
    FOLDER_NAMES_TO_PROCESS,
    ROOT_2026_FOLDER_NAME,
)

# Image to run for each per-video container
IMAGE_NAME = os.getenv("TEST2_IMAGE", "test_runner-image:latest")

# How many test2 containers to run concurrently
MAX_PARALLEL = int(os.getenv("MAX_PARALLEL_TEST2", "8"))

# Poll interval for checking running containers
POLL_SECONDS = int(os.getenv("POLL_SECONDS", "3"))

# Cache volume name (for HF / whisper models cache)
CACHE_VOL = os.getenv("CACHE_VOL", "whisper_cache")

# Optional: keep HF token to avoid rate-limits (pass to spawned containers)
HF_TOKEN = os.getenv("HF_TOKEN", "").strip()

def find_tasks(service):
    candidates = drive_search_folder_anywhere_in_my_drive(service, ROOT_2026_FOLDER_NAME)
    if not candidates:
        raise RuntimeError(f"Could not find folder '{ROOT_2026_FOLDER_NAME}'")

    candidates.sort(key=lambda x: x.get("modifiedTime", ""), reverse=True)
    root_2026_id = candidates[0]["id"]
    slot = choose_slot(service, root_2026_id)

    people = sorted(list(drive_list_children(service, slot["id"], FOLDER_MIME)), key=lambda x: x["name"])

    tasks = []
    for person in people:
        for folder_name in FOLDER_NAMES_TO_PROCESS:
            target = drive_find_child(service, person["id"], folder_name, FOLDER_MIME)
            if not target:
                continue

            folder_id = target["id"]
            children = list(drive_list_children(service, folder_id, None))
            videos = [f for f in children if f.get("mimeType") != FOLDER_MIME and is_video_name(f["name"])]

            for vid in videos:
                out_name = transcript_output_name(vid["name"])
                exists = drive_find_child(service, folder_id, out_name, None)
                if exists:
                    continue

                tasks.append({
                    "video_file_id": vid["id"],
                    "target_folder_id": folder_id,
                    "video_name": vid["name"],
                    "person": person["name"],
                    "folder": folder_name,
                    "slot": slot["name"],
                })
    return tasks

def spawn_one(client, task, network_name: str | None):
    env = {
        # Auth files are baked into the image (credentials.json + token.json),
        # but we still set paths explicitly to avoid ambiguity.
        "HEADLESS_AUTH": os.getenv("HEADLESS_AUTH", "1"),
        "CREDENTIALS_FILE": os.getenv("CREDENTIALS_FILE", "/app/credentials.json"),
        "TOKEN_FILE": os.getenv("TOKEN_FILE", "/app/token.json"),
        "XDG_CACHE_HOME": os.getenv("XDG_CACHE_HOME", "/app/cache"),

        # Single-video work assignment
        "VIDEO_FILE_ID": task["video_file_id"],
        "TARGET_FOLDER_ID": task["target_folder_id"],
        "VIDEO_NAME": task["video_name"],

        # Transcription tuning (inherited)
        "DEVICE": os.getenv("DEVICE", "cpu"),
        "COMPUTE_TYPE": os.getenv("COMPUTE_TYPE", "int8"),
        "WHISPER_MODEL": os.getenv("WHISPER_MODEL", "medium"),
        "USE_VAD": os.getenv("USE_VAD", "true"),
        "CHUNK_SECONDS": os.getenv("CHUNK_SECONDS", "540"),
        "OVERLAP_SECONDS": os.getenv("OVERLAP_SECONDS", "2"),
        "USE_SHARED_DRIVES": os.getenv("USE_SHARED_DRIVES", "0"),
        "FORCE_RETRANSCRIBE": os.getenv("FORCE_RETRANSCRIBE", "0"),
        "LANGUAGE": os.getenv("LANGUAGE", "en"),
    }

    # Optional HF token to avoid anonymous throttling
    if HF_TOKEN:
        env["HF_TOKEN"] = HF_TOKEN
        env["HUGGINGFACE_HUB_TOKEN"] = HF_TOKEN

    volumes = [
        f"{CACHE_VOL}:/app/cache:rw",
    ]

    name = f"test2_{task['video_file_id'][:8]}_{int(time.time())}"
    print(f"[SPAWN] {name} | {task['person']} | {task['folder']} | {task['video_name']}")

    kwargs = {}
    if network_name:
        kwargs["network"] = network_name

    return client.containers.run(
        IMAGE_NAME,
        name=name,
        command=["python", "test2.py"],
        environment=env,
        volumes=volumes,
        detach=True,
        remove=True,  # auto-remove container after exit
        **kwargs,
    )

def main():
    service = get_drive_service()
    tasks = find_tasks(service)

    print(f"[DISPATCH] missing transcripts = {len(tasks)}")
    if not tasks:
        print("[DISPATCH] nothing to do ✅")
        return

    client = docker.from_env()

    # If you want to force a specific network, set COMPOSE_NETWORK in .env
    network_name = os.getenv("COMPOSE_NETWORK", "").strip() or None

    running = []
    idx = 0

    while idx < len(tasks) or running:
        # Start up to MAX_PARALLEL containers
        while idx < len(tasks) and len(running) < MAX_PARALLEL:
            running.append(spawn_one(client, tasks[idx], network_name))
            idx += 1

        # Refresh statuses, keep only running ones
        still_running = []
        for c in running:
            try:
                c.reload()
                if c.status in ("created", "running"):
                    still_running.append(c)
            except Exception:
                # remove=True => container might already be gone
                pass

        running = still_running
        print(f"[STATUS] started={idx}/{len(tasks)} running={len(running)}")
        time.sleep(POLL_SECONDS)

    print("[DONE] all test2 containers completed ✅")

if __name__ == "__main__":
    main()