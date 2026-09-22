"""
Stage 4 -- Submit ONE chunk of batch requests to Vertex, wait for it to
finish, and download the results.

Run once per chunk. Check the output and your Billing report between
chunks rather than submitting all of them back to back.

Usage:
    python scripts/submit_batch_job.py \
        --input-jsonl data/batch_requests/requests_chunk_0000.jsonl \
        --bucket your-project-story-data \
        --project YOUR_PROJECT_ID \
        --chunk-name chunk_0000

This script deliberately does NOT estimate cost before submitting. Use
--dry-run to see the request count and validate the file without spending
anything, then check Billing > Reports after the chunk completes.
"""
import argparse
import json
import shutil
import sys
import time
from pathlib import Path

from google import genai
from google.genai.types import CreateBatchJobConfig, JobState, HttpOptions
from google.cloud import storage

MAX_REQUESTS_PER_JOB = 150_000


def validate_request_file(path: Path) -> int:
    """
    Check the file is real, parseable JSONL with the expected shape
    BEFORE uploading. An upload of a malformed file succeeds and then the
    batch job fails, which wastes a full round trip.
    """
    if not path.exists():
        raise SystemExit(f"Input file not found: {path}")
    if path.stat().st_size == 0:
        raise SystemExit(f"Input file is empty: {path}")

    count = 0
    with open(path, "r", encoding="utf-8") as f:
        for line_no, line in enumerate(f, 1):
            line = line.strip()
            if not line:
                continue
            try:
                row = json.loads(line)
            except json.JSONDecodeError as e:
                raise SystemExit(f"{path}:{line_no} is not valid JSON: {e}")
            if "request" not in row:
                raise SystemExit(f"{path}:{line_no} has no 'request' field")
            try:
                row["request"]["contents"][0]["parts"][0]["text"]
            except (KeyError, IndexError, TypeError):
                raise SystemExit(
                    f"{path}:{line_no} 'request' is not shaped like a "
                    f"GenerateContentRequest"
                )
            count += 1

    if count == 0:
        raise SystemExit(f"No requests found in {path}")
    if count > MAX_REQUESTS_PER_JOB:
        raise SystemExit(
            f"{count:,} requests exceeds the {MAX_REQUESTS_PER_JOB:,} per-job "
            f"limit. Re-run build_batch_requests.py with a smaller --chunk-size."
        )
    return count


def upload_to_gcs(local_path: str, bucket_name: str, dest_blob: str,
                  retries: int = 3) -> str:
    client = storage.Client()
    bucket = client.bucket(bucket_name)
    blob = bucket.blob(dest_blob)
    last_err = None
    for attempt in range(retries):
        try:
            blob.upload_from_filename(local_path)
            return f"gs://{bucket_name}/{dest_blob}"
        except Exception as e:  # noqa: BLE001 -- network/permission/transient
            last_err = e
            if attempt < retries - 1:
                print(f"  upload attempt {attempt + 1} failed ({e}); retrying...")
                time.sleep(2 ** attempt)
    raise SystemExit(f"Upload failed after {retries} attempts: {last_err}")


def download_from_gcs(bucket_name: str, prefix: str, local_dir: str) -> int:
    """
    Mirrors the blob's path under local_dir instead of flattening to the
    basename -- two blobs sharing a basename in different subfolders would
    otherwise silently overwrite each other.
    """
    client = storage.Client()
    bucket = client.bucket(bucket_name)
    local_root = Path(local_dir)
    local_root.mkdir(parents=True, exist_ok=True)

    downloaded = 0
    for blob in bucket.list_blobs(prefix=prefix):
        if blob.name.endswith("/"):
            continue
        relative = blob.name[len(prefix):].lstrip("/") or Path(blob.name).name
        dest = local_root / relative
        dest.parent.mkdir(parents=True, exist_ok=True)
        blob.download_to_filename(str(dest))
        print(f"  downloaded {blob.name} -> {dest}")
        downloaded += 1
    return downloaded


def count_output_rows(local_dir: str) -> int:
    total = 0
    for p in Path(local_dir).rglob("*.jsonl"):
        with open(p, "r", encoding="utf-8") as f:
            total += sum(1 for line in f if line.strip())
    return total


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input-jsonl", required=True)
    parser.add_argument("--bucket", required=True)
    parser.add_argument("--project", required=True)
    parser.add_argument("--location", default="us-central1")
    parser.add_argument("--model", default="gemini-2.5-flash-lite")
    parser.add_argument("--chunk-name", required=True)
    parser.add_argument("--poll-seconds", type=int, default=15)
    parser.add_argument("--dry-run", action="store_true",
                        help="Validate the request file and stop. No upload, "
                             "no job, no cost.")
    args = parser.parse_args()

    in_path = Path(args.input_jsonl)
    request_count = validate_request_file(in_path)
    print(f"Validated {request_count:,} requests in {in_path}")

    if args.dry_run:
        print("--dry-run set: stopping before upload. Nothing was spent.")
        return

    local_out = Path(f"data/batch_results/{args.chunk_name}")
    if local_out.exists() and any(local_out.iterdir()):
        print(f"\n{local_out} already exists and is not empty.")
        print("Mixing old and new output would corrupt the join in "
              "clean_dataset.py.")
        answer = input("Delete it and continue? [y/N] ").strip().lower()
        if answer != "y":
            raise SystemExit("Aborted. Use a different --chunk-name.")
        shutil.rmtree(local_out)

    input_blob = f"batch_input/{args.chunk_name}.jsonl"
    print(f"\nUploading to gs://{args.bucket}/{input_blob} ...")
    src_uri = upload_to_gcs(str(in_path), args.bucket, input_blob)

    output_prefix = f"batch_output/{args.chunk_name}"
    dest_uri = f"gs://{args.bucket}/{output_prefix}/"

    client = genai.Client(
        vertexai=True, project=args.project, location=args.location,
        http_options=HttpOptions(api_version="v1"),
    )

    print(f"Submitting batch job for {args.chunk_name} ...")
    try:
        job = client.batches.create(
            model=args.model,
            src=src_uri,
            config=CreateBatchJobConfig(dest=dest_uri),
        )
    except Exception as e:  # noqa: BLE001
        raise SystemExit(f"Batch job creation failed: {type(e).__name__}: {e}")

    print(f"Job created: {job.name}  (state: {job.state})")

    terminal_states = (
        JobState.JOB_STATE_SUCCEEDED,
        JobState.JOB_STATE_FAILED,
        JobState.JOB_STATE_CANCELLED,
    )
    while job.state not in terminal_states:
        time.sleep(args.poll_seconds)
        try:
            job = client.batches.get(name=job.name)
        except Exception as e:  # noqa: BLE001 -- transient poll failure
            print(f"  poll failed ({e}); retrying...")
            continue
        print(f"  ...state: {job.state}")

    if job.state != JobState.JOB_STATE_SUCCEEDED:
        raise SystemExit(
            f"Batch job ended in state {job.state}. Check the job in the "
            f"console for error details. Job name: {job.name}"
        )

    print(f"\nJob succeeded. Downloading results to {local_out} ...")
    files = download_from_gcs(args.bucket, output_prefix, str(local_out))
    if files == 0:
        raise SystemExit(
            "Job succeeded but no output files were found. Check the output "
            f"location: {dest_uri}"
        )

    output_rows = count_output_rows(str(local_out))
    print(f"\nRequests sent:  {request_count:,}")
    print(f"Rows returned:  {output_rows:,}")
    if output_rows != request_count:
        print(f"MISMATCH: {abs(request_count - output_rows):,} row difference. "
              f"Some requests may have failed. clean_dataset.py will report "
              f"exactly how many usable stories came through.")

    print("\nInspect one line before trusting the rest:")
    print(f"  head -n 1 {local_out}/*.jsonl | python -m json.tool")


if __name__ == "__main__":
    main()
