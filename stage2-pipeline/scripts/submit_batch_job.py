"""
Stage 4 -- Submit ONE chunk of batch requests to Vertex, wait for it to
finish, and download the results.

Run once per chunk. Check the output and your Billing report between
chunks rather than submitting all of them back to back.

Takes one already-built JSONL request chunk, validates that the request file has the right structure,
uploads it to Google Cloud Storage, starts a Gemini Enterprise Agent Platform batch job, polls until job
reaches terminal state, downloads the output files into a local directory, and prints a quick output-row count to 
detect partial success.

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
import shutil
import sys
import time
from pathlib import Path

# HELPERS
sys.path.append(str(Path(__file__).parent))
from batch_schema import MAX_REQUESTS_PER_JOB, extract_prompt_text  # noqa: E402
from gemini_io import FatalConfigError, call_with_retry  # noqa: E402
from pipeline_io import iter_jsonl  # noqa: E402

# Gemini client, enum-like job state constants, and GCS client
from google import genai
from google.genai.types import CreateBatchJobConfig, JobState, HttpOptions
from google.cloud import storage


def validate_request_file(path: Path) -> int:
    """
    Check the file is real, parseable JSONL with the expected shape
    BEFORE uploading. An upload of a malformed file succeeds and then the
    batch job fails, which wastes a full round trip.
    """
    # Prevent silent, empty submission
    if not path.exists():
        raise SystemExit(f"Input file not found: {path}")
    if path.stat().st_size == 0:
        raise SystemExit(f"Input file is empty: {path}")

    def bad_line(line_no, line, exc):
        raise SystemExit(f"{path}:{line_no} is not valid JSON: {exc}")

    # If one line is not valid JSON, the run stops instead of trying a chunk
    count = 0
    for line_no, row in iter_jsonl(path, on_bad=bad_line):
        if "request" not in row:
            raise SystemExit(f"{path}:{line_no} has no 'request' field")
        if extract_prompt_text(row["request"]) is None:
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
    return count # return count of valid lines


# Singleton pattern
_STORAGE_CLIENT = None 


def _storage_bucket(bucket_name: str):
    """One client per python process -- constructing it rediscovers credentials."""
    global _STORAGE_CLIENT
    if _STORAGE_CLIENT is None:
        _STORAGE_CLIENT = storage.Client()
    return _STORAGE_CLIENT.bucket(bucket_name)

# upload generated request JSON file into GCS so Gemini batch job can read it
def upload_to_gcs(local_path: str, bucket_name: str, dest_blob: str,
                  retries: int = 3) -> str:
    # Production chunks are ~200 MB. Sent as one request, that outlasts the
    # library's default timeout on an ordinary home upload link, and every
    # retry starts again from byte 0. An explicit chunk_size makes it a
    # resumable upload in 8 MB pieces: no single request is slow enough to
    # time out, and a failed piece is resent without restarting the file.
    blob = _storage_bucket(bucket_name).blob(dest_blob, chunk_size=8 * 1024 * 1024)

    def report(attempt, exc):
        print(f"  upload attempt {attempt + 1} failed ({exc}); retrying...")

    try:
        call_with_retry(
            lambda: blob.upload_from_filename(local_path, timeout=300), # takes local JSONL into bucket location.
            retries=retries, 
            on_retry=report,
        )
    except (FatalConfigError, RuntimeError) as e:
        raise SystemExit(f"Upload failed: {e}")
    return f"gs://{bucket_name}/{dest_blob}" # Google Storage URI for batch job


def download_from_gcs(bucket_name: str, prefix: str, local_dir: str) -> int:
    """
    Mirrors the blob's path under local_dir instead of flattening to the
    basename -- two blobs sharing a basename in different subfolders would
    otherwise silently overwrite each other.
    """
    bucket = _storage_bucket(bucket_name)
    local_root = Path(local_dir)
    local_root.mkdir(parents=True, exist_ok=True)

    downloaded = 0
    for blob in bucket.list_blobs(prefix=prefix):
        if blob.name.endswith("/"): # if its a folder in cloud
            continue
        relative = blob.name[len(prefix):].lstrip("/") or Path(blob.name).name
        dest = local_root / relative # calculate where to store locally
        dest.parent.mkdir(parents=True, exist_ok=True)
        blob.download_to_filename(str(dest)) # download from cloud to local pathway
        print(f"  downloaded {blob.name} -> {dest}")
        downloaded += 1
    return downloaded

# count how many non-empty lines exist inside the downloaded JSONL output files
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
    parser.add_argument("--dry-run", action="store_true", # it does not upload to GCS or start a job
                        help="Validate the request file and stop. No upload, "
                             "no job, no cost.")
    args = parser.parse_args()

    in_path = Path(args.input_jsonl)
    request_count = validate_request_file(in_path)
    print(f"Validated {request_count:,} requests in {in_path}")

    if args.dry_run:
        print("--dry-run set: stopping before upload. Nothing was spent.")
        return

    # script is careful not to merge old and new results
    local_out = Path(f"data/batch_results/{args.chunk_name}")
    if local_out.exists() and any(local_out.iterdir()):
        # Say what is about to be destroyed. These are completed inferences
        # that were already paid for, and a repeated --chunk-name is an easy
        # mistake to make across dozens of submissions.
        existing_rows = count_output_rows(str(local_out))
        print(f"\n{local_out} already exists and is not empty.")
        print(f"It holds {existing_rows:,} result row(s) from a previous run "
              f"-- stories you have ALREADY PAID FOR.")
        print("Deleting them here does not delete them from Cloud Storage, but "
              "submitting this job will overwrite that output prefix too.")
        print("Mixing old and new output would corrupt the join in "
              "clean_dataset.py, so this run cannot continue without deleting.")
        print(f"\nTo keep them, answer N and re-run with a different "
              f"--chunk-name.")
        answer = input(f"Permanently delete {existing_rows:,} downloaded rows "
                       f"and continue? [y/N] ").strip().lower()
        if answer != "y":
            raise SystemExit("Aborted. Nothing was deleted.")
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

    # Every state a job can stop in. PAUSED, EXPIRED and PARTIALLY_SUCCEEDED
    # are terminal for our purposes too -- leaving them out means polling a
    # job forever that is never going to change state again.
    terminal_states = (
        JobState.JOB_STATE_SUCCEEDED,
        JobState.JOB_STATE_PARTIALLY_SUCCEEDED,
        JobState.JOB_STATE_FAILED,
        JobState.JOB_STATE_CANCELLED,
        JobState.JOB_STATE_PAUSED,
        JobState.JOB_STATE_EXPIRED,
    )
    while job.state not in terminal_states:
        time.sleep(args.poll_seconds)
        try:
            job = client.batches.get(name=job.name) # poll after time interval
        except Exception as e:  # noqa: BLE001 -- transient poll failure
            print(f"  poll failed ({e}); retrying...")
            continue
        print(f"  ...state: {job.state}")

    # Completed rows are exported -- and billed -- even when the job as a
    # whole didn't succeed, so a partial job is worth downloading rather
    # than throwing away.
    usable_states = (
        JobState.JOB_STATE_SUCCEEDED,
        JobState.JOB_STATE_PARTIALLY_SUCCEEDED,
    )
    if job.state not in usable_states:
        raise SystemExit(
            f"Batch job ended in state {job.state}. Check the job in the "
            f"console for error details. Job name: {job.name}"
        )
    if job.state == JobState.JOB_STATE_PARTIALLY_SUCCEEDED:
        print(f"\nWARNING: job ended in {job.state} -- some requests failed. "
              f"Downloading the rows that did complete; clean_dataset.py will "
              f"report how many are usable.")

    print(f"\nDownloading results to {local_out} ...")
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
