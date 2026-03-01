import os
import time
import argparse
from datetime import datetime, timezone
import boto3

REGION = os.getenv("AWS_REGION", "us-east-1")
LOG_GROUP = os.getenv("WORKER_LOG_GROUP", "/transcription/workers")
OUT_DIR_DEFAULT = os.getenv("WORKER_LOG_OUT_DIR", "/home/ec2-user/worker_logs")

ec2 = boto3.client("ec2", region_name=REGION)
logs = boto3.client("logs", region_name=REGION)

def list_workers(limit: int = 50):
    resp = ec2.describe_instances(
        Filters=[
            {"Name": "tag:Purpose", "Values": ["test2"]},
            {"Name": "instance-state-name", "Values": ["pending", "running", "stopping", "stopped", "shutting-down"]},
        ]
    )
    out = []
    for r in resp.get("Reservations", []):
        for i in r.get("Instances", []):
            out.append({
                "id": i["InstanceId"],
                "state": i["State"]["Name"],
                "launch": i["LaunchTime"],
                "az": i["Placement"]["AvailabilityZone"],
                "ip": i.get("PublicIpAddress", "-"),
                "name": next((t["Value"] for t in i.get("Tags", []) if t["Key"] == "Name"), "-"),
                "person": next((t["Value"] for t in i.get("Tags", []) if t["Key"] == "Person"), "-"),
                "slot": next((t["Value"] for t in i.get("Tags", []) if t["Key"] == "Slot"), "-"),
            })
    out.sort(key=lambda x: x["launch"], reverse=True)
    return out[:limit]

def tail_log_stream(instance_id: str, out_dir: str):
    os.makedirs(out_dir, exist_ok=True)
    out_path = os.path.join(out_dir, f"{instance_id}.log")

    next_token = None
    seen = set()

    print(f"\n[TARGET] {instance_id}")
    print(f"[LOG_GROUP] {LOG_GROUP}")
    print(f"[SAVE] {out_path}\n")

    while True:
        kwargs = {
            "logGroupName": LOG_GROUP,
            "logStreamName": instance_id,
            "startFromHead": True,
        }
        if next_token:
            kwargs["nextToken"] = next_token

        try:
            resp = logs.get_log_events(**kwargs)
        except logs.exceptions.ResourceNotFoundException:
            print("[WAIT] log stream not created yet... retrying in 3s")
            time.sleep(3)
            continue

        next_token = resp.get("nextForwardToken")

        new_lines = []
        for e in resp.get("events", []):
            key = (e["timestamp"], e["message"])
            if key in seen:
                continue
            seen.add(key)

            ts = datetime.fromtimestamp(e["timestamp"] / 1000, tz=timezone.utc).astimezone().strftime("%Y-%m-%d %H:%M:%S")
            msg = e["message"].rstrip()
            new_lines.append(f"{ts} | {msg}")

        if new_lines:
            with open(out_path, "a", encoding="utf-8") as f:
                for line in new_lines:
                    print(line)
                    f.write(line + "\n")

        time.sleep(2)

def main():
    ap = argparse.ArgumentParser(description="View worker logs in real time from CloudWatch.")
    ap.add_argument("--pick", type=int, default=0, help="Pick worker number from list (1..N)")
    ap.add_argument("--id", type=str, default="", help="Tail logs for this instance id directly")
    ap.add_argument("--limit", type=int, default=30, help="How many workers to show")
    ap.add_argument("--out-dir", type=str, default=OUT_DIR_DEFAULT, help="Where to save logs locally")
    args = ap.parse_args()

    if args.id:
        tail_log_stream(args.id, args.out_dir)
        return

    workers = list_workers(limit=args.limit)
    if not workers:
        print("No worker instances found (Purpose=test2).")
        return

    print("\nWorkers (latest first):")
    for idx, w in enumerate(workers, start=1):
        print(
            f"{idx:2}. {w['id']}  state={w['state']}  az={w['az']}  "
            f"slot={w['slot']}  person={w['person']}  launch={w['launch']}"
        )

    if args.pick <= 0 or args.pick > len(workers):
        print("\nRun with: python3 worker_logs.py --pick 1")
        return

    target = workers[args.pick - 1]["id"]
    tail_log_stream(target, args.out_dir)

if __name__ == "__main__":
    main()