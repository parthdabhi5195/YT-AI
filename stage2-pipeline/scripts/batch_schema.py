"""
The batch request contract, shared by the three stages that touch it:
build_batch_requests.py writes these lines, submit_batch_job.py validates
them before upload, and clean_dataset.py reads the prompt back out of the
echoed request to join on.

That path -- request.contents[0].parts[0].text -- was previously written
out by hand in all three files, which is the schema this pipeline is
least sure about hardcoded in the most places. Writer and reader live
next to each other here so a change to one is visibly a change to both.

No third-party dependencies -- importable anywhere in the pipeline.
"""
import hashlib

# Google's per-job ceiling for Vertex batch prediction.
MAX_REQUESTS_PER_JOB = 150_000

CUSTOM_ID_FIELD = "custom_id"
PROMPT_HASH_FIELD = "prompt_sha256"


def prompt_hash(prompt):
    """The join key clean_dataset falls back to when custom_id isn't returned from batch output."""
    return hashlib.sha256(prompt.encode("utf-8")).hexdigest() # if custom_id is missing, the prompt hash can be used to recover the source template and variant index from the original prompt


def build_request_line(custom_id, prompt, temperature, max_output_tokens):
    """One line of a batch request file, in GenerateContentRequest shape."""
    # remove the custom_id from the request itself since it's not part of the API. remove if it doesn't work
    return {
        CUSTOM_ID_FIELD: custom_id,
        "request": {
            "contents": [{"role": "user", "parts": [{"text": prompt}]}],
            "generationConfig": {
                "temperature": temperature,
                "maxOutputTokens": max_output_tokens,
            },
        },
    }


def extract_prompt_text(request):
    """
    Inverse of build_request_line: the prompt back out of a request
    object, or None if it isn't shaped like one.
    """
    if not isinstance(request, dict):
        return None
    try:
        return request["contents"][0]["parts"][0]["text"] # extract text
    except (KeyError, IndexError, TypeError):
        return None
