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
    video_file_id = must("VIDEO_FILE_ID")
    target_folder_id = must("TARGET_FOLDER_ID")
    video_name = os.getenv("VIDEO_NAME", "").strip()

    env_param = os.getenv("SSM_ENV_PARAM", "/transcription/worker/env")
    cred_param = os.getenv("SSM_CREDENTIALS_PARAM", "/transcription/worker/credentials_json")
    token_param = os.getenv("SSM_TOKEN_PARAM", "/transcription/worker/token_json")

    os.makedirs("/app/state", exist_ok=True)

    # Fetch and write files expected by your scripts
    with open("/app/.env", "w", encoding="utf-8") as f:
        f.write(ssm_get(region, env_param))
    with open("/app/credentials.json", "w", encoding="utf-8") as f:
        f.write(ssm_get(region, cred_param))
    with open("/app/token.json", "w", encoding="utf-8") as f:
        f.write(ssm_get(region, token_param))

    os.environ["CREDENTIALS_FILE"] = "/app/credentials.json"
    os.environ["TOKEN_FILE"] = "/app/token.json"
    os.environ["TOKEN_FILE_WRITE"] = "/app/state/token_refreshed.json"
    os.environ["HEADLESS_AUTH"] = "1"

    os.environ["VIDEO_FILE_ID"] = video_file_id
    os.environ["TARGET_FOLDER_ID"] = target_folder_id
    if video_name:
        os.environ["VIDEO_NAME"] = video_name

    print(f"[WORKER] Starting test2.py for {video_name or video_file_id}")
    subprocess.run(["python", "test2.py"], check=False)

if __name__ == "__main__":
    main()