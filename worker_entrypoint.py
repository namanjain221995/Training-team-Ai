import os
import subprocess
import boto3

def must(name: str) -> str:
    v = os.getenv(name, "").strip()
    if not v:
        raise RuntimeError(f"Missing required env var: {name}")
    return v

def ssm_get(region: str, name: str) -> str:
    ssm = boto3.client("ssm", region_name=region)
    return ssm.get_parameter(Name=name, WithDecryption=True)["Parameter"]["Value"]

def main():
    region = must("AWS_REGION")

    video_batch_json = os.getenv("VIDEO_BATCH_JSON", "").strip()
    video_file_id = os.getenv("VIDEO_FILE_ID", "").strip()
    target_folder_id = os.getenv("TARGET_FOLDER_ID", "").strip()
    video_name = os.getenv("VIDEO_NAME", "").strip()

    if not video_batch_json and not (video_file_id and target_folder_id):
        raise RuntimeError("Either VIDEO_BATCH_JSON or VIDEO_FILE_ID + TARGET_FOLDER_ID must be provided")

    env_param = os.getenv("SSM_ENV_PARAM", "/transcription/worker/env")
    cred_param = os.getenv("SSM_CREDENTIALS_PARAM", "/transcription/worker/credentials_json")
    token_param = os.getenv("SSM_TOKEN_PARAM", "/transcription/worker/token_json")

    os.makedirs("/app/state", exist_ok=True)

    print(f"[WORKER] Fetching SSM env from {env_param}")
    with open("/app/.env", "w", encoding="utf-8") as f:
        f.write(ssm_get(region, env_param))

    print(f"[WORKER] Fetching SSM credentials from {cred_param}")
    with open("/app/credentials.json", "w", encoding="utf-8") as f:
        f.write(ssm_get(region, cred_param))

    print(f"[WORKER] Fetching SSM token from {token_param}")
    with open("/app/token.json", "w", encoding="utf-8") as f:
        f.write(ssm_get(region, token_param))

    os.environ["CREDENTIALS_FILE"] = "/app/credentials.json"
    os.environ["TOKEN_FILE"] = "/app/token.json"
    os.environ["HEADLESS_AUTH"] = "1"

    if video_batch_json:
        os.environ["VIDEO_BATCH_JSON"] = video_batch_json
        print("[WORKER] Starting batch mode")
    else:
        os.environ["VIDEO_FILE_ID"] = video_file_id
        os.environ["TARGET_FOLDER_ID"] = target_folder_id
        if video_name:
            os.environ["VIDEO_NAME"] = video_name
        print(f"[WORKER] Starting single-video mode for {video_name or video_file_id}")

    subprocess.run(["python", "-u", "test2.py"], check=True)

if __name__ == "__main__":
    main()