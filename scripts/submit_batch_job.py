"""
Stage 4 -- Submit ONE chunk of batch requests to Vertex AI, wait for it to
finish, and download the results.

Run this once per chunk. Check the output and your Billing report between
chunks rather than submitting all of them back to back.

Usage:
    python scripts/submit_batch_job.py \
        --input-jsonl data/batch_requests/requests_chunk_0000.jsonl \
        --bucket your-project-story-data \
        --project YOUR_PROJECT_ID \
        --chunk-name chunk_0000
"""
import argparse
import time
from pathlib import Path

from google import genai
from google.genai.types import CreateBatchJobConfig, JobState, HttpOptions
from google.cloud import storage


def upload_to_gcs(local_path: str, bucket_name: str, dest_blob: str) -> str:
    client = storage.Client()
    bucket = client.bucket(bucket_name)
    blob = bucket.blob(dest_blob)
    blob.upload_from_filename(local_path)
    return f"gs://{bucket_name}/{dest_blob}"


def download_from_gcs(bucket_name: str, prefix: str, local_dir: str):
    client = storage.Client()
    bucket = client.bucket(bucket_name)
    Path(local_dir).mkdir(parents=True, exist_ok=True)
    for blob in bucket.list_blobs(prefix=prefix):
        if blob.name.endswith("/"):
            continue
        dest = Path(local_dir) / Path(blob.name).name
        blob.download_to_filename(str(dest))
        print(f"Downloaded {blob.name} -> {dest}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input-jsonl", required=True)
    parser.add_argument("--bucket", required=True)
    parser.add_argument("--project", required=True)
    parser.add_argument("--location", default="us-central1")
    parser.add_argument("--model", default="gemini-2.5-flash-lite")
    parser.add_argument("--chunk-name", required=True)
    parser.add_argument("--poll-seconds", type=int, default=30)
    args = parser.parse_args()

    input_blob = f"batch_input/{args.chunk_name}.jsonl"
    print(f"Uploading {args.input_jsonl} to gs://{args.bucket}/{input_blob} ...")
    src_uri = upload_to_gcs(args.input_jsonl, args.bucket, input_blob)

    dest_uri = f"gs://{args.bucket}/batch_output/{args.chunk_name}/"

    client = genai.Client(
        vertexai=True, project=args.project, location=args.location,
        http_options=HttpOptions(api_version="v1"),
    )

    print(f"Submitting batch job for {args.chunk_name} ...")
    job = client.batches.create(
        model=args.model,
        src=src_uri,
        config=CreateBatchJobConfig(dest=dest_uri),
    )
    print(f"Job created: {job.name}  (state: {job.state})")

    terminal_states = (
        JobState.JOB_STATE_SUCCEEDED,
        JobState.JOB_STATE_FAILED,
        JobState.JOB_STATE_CANCELLED,
    )
    while job.state not in terminal_states:
        time.sleep(args.poll_seconds)
        job = client.batches.get(name=job.name)
        print(f"  ...state: {job.state}")

    if job.state != JobState.JOB_STATE_SUCCEEDED:
        raise SystemExit(
            f"Batch job ended in state {job.state} -- check the console for "
            f"error details before retrying this chunk."
        )

    local_out = f"data/batch_results/{args.chunk_name}"
    print(f"Job succeeded. Downloading results to {local_out} ...")
    download_from_gcs(args.bucket, f"batch_output/{args.chunk_name}/", local_out)
    print("Done. Inspect one line before trusting the rest:")
    print(f"  head -n 1 {local_out}/*.jsonl | python -m json.tool")


if __name__ == "__main__":
    main()
