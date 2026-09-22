import os
import json
import re
import time
import random
import logging
import hashlib
import threading
from concurrent.futures import ThreadPoolExecutor, as_completed

import requests
import html
import yt_dlp
from deepmultilingualpunctuation import PunctuationModel

# ==============================================================================
# CONFIGURATION
# ==============================================================================
BASE_DIR = r"C:\Users\parth\Desktop\Youtube\YouTube Account (TheUnfilteredTales)\Revived\RequestedReads"
LOG_FILE = os.path.join(BASE_DIR, "scraper.log")

PROXY_USER = os.environ.get("WEBSHARE_USER", "username")
PROXY_PASS = os.environ.get("WEBSHARE_PASS", "password")
PROXY_HOST = "p.webshare.io"
PROXY_PORT = 80
PROXY_URL = f"http://{PROXY_USER}:{PROXY_PASS}@{PROXY_HOST}:{PROXY_PORT}"

CHANNELS = [
    "https://www.youtube.com/@Requestedreads/shorts",
]

MIN_VIEWS = 50_000

# How many videos to process concurrently. Each one opens its own proxy
# session, so this is bounded by your Webshare plan's concurrent-connection
# limit -- check your plan/dashboard before raising this. Start low, watch
# the error rate in scraper.log for a while, then increase.
MAX_WORKERS = 6

ENGLISH_KEYS_PRIORITY = ("en", "en-orig", "en-US", "en-GB", "en-CA", "en-AU")

# ==============================================================================
# INITIALIZATION
# ==============================================================================
def setup_logging():
    os.makedirs(BASE_DIR, exist_ok=True)
    logger = logging.getLogger("scraper")
    logger.setLevel(logging.INFO)
    fmt = logging.Formatter("%(asctime)s [%(levelname)s] %(threadName)s: %(message)s", "%H:%M:%S")

    stream = logging.StreamHandler()
    stream.setFormatter(fmt)
    logger.addHandler(stream)

    file_handler = logging.FileHandler(LOG_FILE, encoding="utf-8")
    file_handler.setFormatter(fmt)
    logger.addHandler(file_handler)

    return logger

log = setup_logging()

log.info("Loading punctuation restoration model... This may take a moment.")
punct_model = PunctuationModel()
log.info("Punctuation model loaded successfully.")

write_lock = threading.Lock()

# ==============================================================================
# DATA SERIALIZATION & STATE
# ==============================================================================
def load_processed_ids(filepath):
    processed = set()
    if os.path.exists(filepath):
        with open(filepath, 'r', encoding='utf-8') as f:
            for line in f:
                if line.strip():
                    try:
                        vid = json.loads(line).get('VideoID')
                        if vid:
                            processed.add(vid)
                    except json.JSONDecodeError:
                        continue
    return processed

def append_to_jsonl(filepath, record):
    with write_lock:
        with open(filepath, 'a', encoding='utf-8') as f:
            f.write(json.dumps(record, ensure_ascii=False) + '\n')

# ==============================================================================
# PROXY HELPERS
# ==============================================================================
def sticky_proxy_for(video_id: str) -> str:
    """Pins one video's metadata + subtitle request to the same exit IP."""
    numeric_session = int(hashlib.md5(video_id.encode()).hexdigest()[:8], 16) % 1000000
    return f"http://{PROXY_USER}-us-{numeric_session}:{PROXY_PASS}@{PROXY_HOST}:{PROXY_PORT}"

# ==============================================================================
# EXTRACTION & CLEANING
# ==============================================================================
def _find_english_track(subs, auto_subs):
    for source, label in ((subs, "manual"), (auto_subs, "auto")):
        if not source:
            continue
        for key in ENGLISH_KEYS_PRIORITY:
            if key in source:
                return source[key], f"{label}:{key}"
        for key in source:
            if key.startswith("en"):
                return source[key], f"{label}:{key}"
    return None, None

def fetch_transcript_in_memory(info, proxy_url):
    subs = info.get('subtitles', {}) or {}
    auto_subs = info.get('automatic_captions', {}) or {}
    target_sub, source_label = _find_english_track(subs, auto_subs)

    if not target_sub:
        return None

    json3_entry = next((s for s in target_sub if s.get('ext') == 'json3'), None)
    if not json3_entry:
        return None

    sub_url = json3_entry['url']
    headers = {
        'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36',
        'Accept-Language': 'en-US,en;q=0.9',
    }
    proxies = {"http": proxy_url, "https": proxy_url}

    for attempt in range(1, 4):
        try:
            response = requests.get(sub_url, headers=headers, proxies=proxies, timeout=12)
            if response.status_code == 200:
                try:
                    data = response.json()
                except ValueError:
                    return None

                lines = []
                for event in data.get('events', []):
                    if 'segs' not in event:
                        continue
                    segment_text = ''.join(s.get('utf8', '') for s in event['segs'])
                    clean_text = re.sub(r'\[.*?\]|\(.*?\)', '', segment_text)
                    clean_text = re.sub(r'\s+', ' ', clean_text).strip()
                    clean_text = html.unescape(clean_text)
                    if clean_text:
                        lines.append(clean_text)

                return " ".join(lines) or None

            # Back off on ANY non-200, not just 429. Capped shorter than
            # before -- this is a scraper, not a polite playback client, and
            # a full 3-attempt backoff was previously eating 15-20+ seconds
            # per failed video for nothing.
            time.sleep(min(2 ** attempt, 6) + random.uniform(0.5, 1.5))
        except requests.RequestException:
            time.sleep(1.0)

    return None

# ==============================================================================
# PER-VIDEO WORKER (runs inside the thread pool)
# ==============================================================================
def process_video(entry, idx, total, channel_info, channel_file):
    vid_id = entry['id']
    url = f"https://www.youtube.com/watch?v={vid_id}"
    proxy_for_video = sticky_proxy_for(vid_id)

    video_opts = {
        'quiet': True,
        'no_warnings': True,
        'skip_download': True,
        'proxy': proxy_for_video,
        'writesubtitles': False,
        'extractor_args': {
            'youtube': {
                # Default yt-dlp queries several player clients per video
                # (web/mweb/ios/android/tv...) to cover format/PO-token
                # gaps. We only need metadata + a caption URL, so pin to
                # one light client instead. 'android' has historically not
                # needed a PO token for captions (unlike 'web', which
                # increasingly does) but can occasionally return empty
                # automatic_captions for a video that does have them --
                # watch your "no transcript" rate after switching; fall
                # back to player_client=['web'] if it climbs noticeably.
                'player_client': ['android'],
            }
        },
    }

    t0 = time.monotonic()
    try:
        with yt_dlp.YoutubeDL(video_opts) as ydl_video:
            info = ydl_video.extract_info(url, download=False)
        t_extract = time.monotonic() - t0

        views = info.get('view_count') or 0
        if views < MIN_VIEWS:
            log.info("Skipped (%d/%d): %s -- %d views", idx, total, vid_id, views)
            return

        t1 = time.monotonic()
        raw_transcript = fetch_transcript_in_memory(info, proxy_for_video)
        t_subs = time.monotonic() - t1

        if not raw_transcript:
            log.info("No transcript (%d/%d): %s [extract=%.1fs subs=%.1fs]",
                      idx, total, vid_id, t_extract, t_subs)
            return

        record = {
            "VideoID": vid_id,
            "Channel": channel_info.get('channel') or channel_info.get('uploader', 'Unknown'),
            "Title": info.get('title') or "",
            "Duration": info.get('duration'),
            "Views": views,
            "RawTranscript": raw_transcript,  # punctuation restored in a separate pass below
        }
        append_to_jsonl(channel_file, record)
        log.info("Saved (%d/%d): %s... [extract=%.1fs subs=%.1fs]",
                  idx, total, record['Title'][:40], t_extract, t_subs)

    except Exception as e:
        log.error("Failed %s: %s", url, e)

    time.sleep(random.uniform(0.4, 1.0))

# ==============================================================================
# PUNCTUATION PASS -- deliberately separate from the network loop
# ==============================================================================
def restore_punctuation_pass(channel_file):
    """
    Runs once, after scraping finishes for a channel. Kept out of the
    per-video loop because model inference is CPU-bound and was previously
    blocking on every single video while YouTube requests waited -- network
    and CPU work never overlapped. Idempotent: only touches records that
    still carry RawTranscript, so it's safe to re-run.
    """
    if not os.path.exists(channel_file):
        return

    with open(channel_file, 'r', encoding='utf-8') as f:
        records = [json.loads(line) for line in f if line.strip()]

    pending = [r for r in records if 'RawTranscript' in r and 'Transcript' not in r]
    if not pending:
        return

    log.info("Restoring punctuation for %d transcripts in %s...", len(pending), channel_file)
    t0 = time.monotonic()
    for rec in pending:
        raw = rec.pop('RawTranscript')
        rec['Transcript'] = punct_model.restore_punctuation(raw) if raw.strip() else raw
    elapsed = time.monotonic() - t0

    tmp_path = channel_file + ".tmp"
    with open(tmp_path, 'w', encoding='utf-8') as f:
        for rec in records:
            f.write(json.dumps(rec, ensure_ascii=False) + '\n')
    os.replace(tmp_path, channel_file)

    log.info("Punctuation restore done for %s (%d transcripts, %.1fs total, %.2fs/video).",
              channel_file, len(pending), elapsed, elapsed / len(pending))

# ==============================================================================
# PIPELINE EXECUTION
# ==============================================================================
def process_channels(channels):
    os.makedirs(BASE_DIR, exist_ok=True)

    ydl_opts_flat = {
        'extract_flat': True,
        'quiet': True,
        'no_warnings': True,
        'ignoreerrors': True,
        'proxy': PROXY_URL,
    }

    for channel_url in channels:
        log.info("Scanning channel: %s", channel_url)

        match = re.search(r'(@[\w-]+)', channel_url)
        channel_handle = match.group(1) if match else "Unknown_Channel"
        channel_file = os.path.join(BASE_DIR, f"{channel_handle}.jsonl")
        existing_ids = load_processed_ids(channel_file)
        log.info("Loaded %d previously processed videos for %s.", len(existing_ids), channel_handle)

        try:
            with yt_dlp.YoutubeDL(ydl_opts_flat) as ydl_flat:
                channel_info = ydl_flat.extract_info(channel_url, download=False)
        except Exception as e:
            log.error("Failed to fetch channel %s: %s", channel_url, e)
            continue

        if not channel_info:
            log.error("Channel extraction returned nothing for %s", channel_url)
            continue

        all_entries = [e for e in (channel_info.get('entries') or []) if e]

        # Cheap pre-filter using the flat view_count when yt-dlp actually
        # supplies one. Only entries with NO view_count (None) fall through
        # to a full per-video check -- this avoids paying for the heavy,
        # multi-request extract_info() call on videos that were never going
        # to qualify anyway.
        candidate_entries = [
            e for e in all_entries
            if e.get('id') not in existing_ids
            and (e.get('view_count') is None or e.get('view_count') >= MIN_VIEWS)
        ]
        log.info("Found %d valid candidates to process.", len(candidate_entries))

        with ThreadPoolExecutor(max_workers=MAX_WORKERS, thread_name_prefix="worker") as pool:
            futures = [
                pool.submit(process_video, entry, idx, len(candidate_entries), channel_info, channel_file)
                for idx, entry in enumerate(candidate_entries, 1)
            ]
            for future in as_completed(futures):
                future.result()  # re-raise anything unexpected instead of swallowing it silently

        restore_punctuation_pass(channel_file)

if __name__ == "__main__":
    process_channels(CHANNELS)
