"""
Shared Vertex/Gemini call plumbing: retry policy, fatal-error detection,
and the single-call helper both generating stages use.

extract_templates.py and pilot_generate.py each carried their own copy of
this loop, which is why pilot_generate had to reach into a sibling stage
script for two underscore-private helpers. One copy lives here instead.
"""
import time

# Status codes where retrying cannot help: the request or the config is
# wrong in a way that will be wrong again next time.
#   400 INVALID_ARGUMENT is deliberately NOT here -- it is usually a
#   problem with ONE request, not the run, so that row should be skipped
#   and the run should continue.
FATAL_STATUS_CODES = frozenset({
    401,  # UNAUTHENTICATED -- credentials missing or invalid
    403,  # PERMISSION_DENIED -- wrong project, API not enabled, no role
    404,  # NOT_FOUND -- model name doesn't exist in this region
})

# Retry these even if a message happens to contain a scary word.
RETRYABLE_STATUS_CODES = frozenset({
    408, # REQUEST_TIMEOUT -- the request took too long to process
    429, # RESOURCE_EXHAUSTED -- the request rate exceeded the limit
    500, # UNKNOWN / INTERNAL -- the server encountered an unexpected condition
    502, # BAD_GATEWAY -- the server received an invalid response from an upstream server
    503, # UNAVAILABLE -- the server is currently unable to handle the request
    504  # DEADLINE_EXCEEDED -- the server did not receive a timely response from an upstream server
})

# Only consulted when no status code could be found on the exception.
# "billing" and "invalid_argument" are deliberately absent -- those were
# the two false positives.
FATAL_ERROR_MARKERS = (
    "permission_denied", "permission denied", "unauthenticated",
    "was not found", "api key not valid",
    "could not automatically determine credentials",
    "has not been used in project", "is disabled",
    # Missing Application Default Credentials. google.auth raises this
    # before any HTTP request, so there is no status code to key off, and
    # the phrasing ("credentials WERE not found") slipped past the
    # "was not found" marker above -- which meant a run with no auth
    # retried 3x and skipped, row after row, through the whole input
    # instead of stopping on the first one.
    "defaultcredentialserror",
    "default credentials were not found",
    "application default credentials",
)


class FatalConfigError(RuntimeError):
    """Configuration-level failure -- stop the whole run, don't retry."""

# error object returned as JSON object from the Gemini API. exc is the exception object raised by the SDK,
def _status_code(exc):
    """
    Best-effort status code from a google-genai / google-api-core
    exception. Attribute names differ across SDK versions and wrapper
    layers, so try the common ones, then fall back to parsing a leading
    3-digit code out of the message ("429 RESOURCE_EXHAUSTED: ...").
    """
    for attr in ("code", "status_code", "http_status"):
        value = getattr(exc, attr, None)
        if isinstance(value, int):
            return value # Return error int like 429 
        if isinstance(value, str) and value.strip().isdigit():
            return int(value) # Convert and return error int like "429"

    response = getattr(exc, "response", None)
    if response is not None:
        value = getattr(response, "status_code", None)
        if isinstance(value, int):
            return value

    head = str(exc).strip()[:3]
    if head.isdigit():
        return int(head)
    return None


def is_fatal(exc):
    """
    True if this error means the run should stop rather than retry.

    Order matters: an explicit retryable code wins over any substring,
    which is what keeps a 429 whose message mentions billing from being
    read as a billing misconfiguration.
    """
    code = _status_code(exc)
    if code is not None:
        if code in RETRYABLE_STATUS_CODES:
            return False # False if not fatal
        if code in FATAL_STATUS_CODES:
            return True # True if fatal
        # 400 and anything else unrecognized: treat as a per-request
        # problem. call_with_retry will retry it a few times and then
        # raise RuntimeError, which the stage scripts skip the row on.
        return False

    # Fallback logic if above didn't work.
    msg = f"{type(exc).__name__}: {exc}".lower() # Search the exception message for known fatal markers
    return any(marker in msg for marker in FATAL_ERROR_MARKERS)


def call_with_retry(fn, retries=3, on_retry=None):
    """
    Run fn(), retrying transient failures with 1s/2s backoff and failing
    fast on anything that looks like misconfiguration.

    Catches broadly on purpose. Beyond google.genai.errors.APIError, real
    runs hit AttributeError/TypeError when resp.text is missing or None
    (blocked or empty response), json.JSONDecodeError (a ValueError
    subclass) on malformed JSON, google.auth errors when credentials
    expire mid-run, and assorted httpx/socket errors on network blips.
    Enumerating those exactly across SDK versions is fragile, so anything
    transient is retried and anything fatal stops the run.

    on_retry(attempt, exc) is called before each backoff sleep.
    """
    last_err = None
    for attempt in range(retries):
        try:
            return fn()
        except Exception as e:  # noqa: BLE001 -- see docstring
            if is_fatal(e):
                raise FatalConfigError(
                    f"Configuration error, stopping run: {type(e).__name__}: {e}" # fatal errors mid run
                ) from e
            last_err = e
            if attempt < retries - 1:
                if on_retry is not None:
                    on_retry(attempt, e)
                time.sleep(2 ** attempt) # Exponential backoff: 1s, 2s, 4s as suggested by Gemini docs
    raise RuntimeError(
        f"failed after {retries} attempts: {type(last_err).__name__}: {last_err}" # Alert if failed after retries
    )


def generate_text(client, model, prompt, config=None):
    """
    One Gemini call, returning stripped text.

    resp.text can be absent or None when a response is blocked or empty;
    that is raised as ValueError so call_with_retry treats it as transient.
    """
    kwargs = {"config": config} if config else {} # Options like temperature, max_output_tokens, etc. can be passed in config
    resp = client.models.generate_content(model=model, contents=prompt, **kwargs)
    text = getattr(resp, "text", None) # return the text attribute of the response object
    if not text or not text.strip(): # if not text or empty string, or shaped weird, raise the ValueError
        raise ValueError("empty or missing response text")
    return text.strip()
