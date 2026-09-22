"""
Shared file plumbing for the pipeline stages.

Every stage reads JSONL, resumes from its own output, and prints a stats
summary. Those were written out separately in each script, and they had
already drifted: clean_dataset.load_metadata crashed on a truncated line
that extract_templates.load_done_ids was specifically hardened to
tolerate, and pilot_generate's copy of the resume reader had silently
dropped the warning that tells you a row was discarded.

No third-party dependencies -- importable anywhere in the pipeline.
"""
import json
from pathlib import Path

# Carried from the source export through every later stage so a generated
# story can still be traced back to the video it was modelled on. Stage 1's
# output IS the source record, so it keeps the bare names; later stages
# emit derived records and prefix them to avoid colliding with their own
# fields.
SOURCE_METADATA_FIELDS = ("title", "channel", "views")


def iter_jsonl(path, on_bad=None):
    """
    Yield (line_no, row) for each non-blank line of a JSONL file.

    on_bad(line_no, line, exc) is called for a line that doesn't parse,
    and the line is then skipped; pass None to skip bad lines silently.
    Callers that must abort instead can raise from on_bad.
    """
    with open(path, "r", encoding="utf-8") as f:
        for line_no, line in enumerate(f, 1): # 1, json
            line = line.strip() # remove whitespace from both ends
            if not line: # if the line is empty, skip it
                continue
            try:
                row = json.loads(line) # create a Python dictionary from JSON
            except json.JSONDecodeError as e: # call on_bad if the line is not valid JSON
                if on_bad is not None:
                    on_bad(line_no, line, e)
                continue
            yield line_no, row # (line_number, json_object) one at a time


def load_ids(path, id_field="id"):
    """
    IDs already present in an output file, so a run can resume.

    Tolerates a truncated final line -- that is what a previous run being
    killed mid-write leaves behind, and it shouldn't block resuming.
    """
    path = Path(path)
    if not path.exists():
        return set() # no previous output, so no IDs to load

    def warn(line_no, line, exc):
        print(f"  warning: output line {line_no} is not valid JSON "
              f"(likely a truncated final line) -- ignoring it")

    return { # return {...} means a set of all the IDs in the output file
        row[id_field]
        for _, row in iter_jsonl(path, on_bad=warn) # iterate through the JSONL file
        if isinstance(row, dict) and id_field in row # if the parse JSON is a dictionary and has the id_field, add it to the set
    }


def ensure_trailing_newline(path):
    """
    If a previous run was killed mid-write, the file can end without a
    newline. Appending then concatenates the new row onto the partial one,
    corrupting BOTH -- the partial line AND the row being written, which
    silently disappears. Repair the boundary before appending.
    """
    path = Path(path)
    if not path.exists() or path.stat().st_size == 0: # does file exist or is it empty? 
        return # Then nothing to do 
    with open(path, "rb") as f: # script can inspect raw bytes to bypass any encoding issues
        f.seek(-1, 2) # (-1, 2) means go to the end of file (2) and move back one byte (-1) 
        if f.read(1) != b"\n": # reads the final byte and checks if it is a newline character
            with open(path, "a", encoding="utf-8") as fa:
                fa.write("\n") # bandage the broken JSON line
            print("  note: repaired a missing newline at the end of the "
                  "output file (previous run was interrupted mid-write)")


def append_row(f, row): # (output_file, json_object)
    """
    Write one JSONL row and flush.

    The flush is deliberate. These are append-only resume logs for rows
    that each cost an API call, so losing a buffer to a kill -9 means
    paying to generate them again. Measured at ~2.5us/row, which is
    nothing against a multi-second API call.

    Do NOT reuse this in the two scripts that write millions of rows
    (build_batch_requests, clean_dataset). There the same flush would buy
    ~2M extra syscalls to protect rows that are cheap to rebuild locally.
    """
    f.write(json.dumps(row) + "\n") # convert Python dictionary to JSON and write it to the file with a newline
    f.flush() # send the data to the hard disk immendiately instead of storing buffer data in RAM.


def carry_metadata(src, dst, prefix=""):
    """Copy whichever SOURCE_METADATA_FIELDS are present from src to dst."""
    for key in SOURCE_METADATA_FIELDS:
        if key in src:
            dst[f"{prefix}{key}"] = src[key]
    return dst


def print_stats(title, stats):
    """Summary block, right-aligned to the longest key present."""
    print(f"\n--- {title} ---")
    width = max((len(k) for k in stats), default=0)
    for key, value in stats.items():
        print(f"{key:>{width}}: {value}")
