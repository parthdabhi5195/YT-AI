import os
import json
import re
import time
import random
import logging
import hashlib
import requests
import html
import yt_dlp
from deepmultilingualpunctuation import PunctuationModel

# ==============================================================================
# CONFIGURATION
# ==============================================================================
BASE_DIR = r"C:\Users\parth\Desktop\Youtube\YouTube Account (TheUnfilteredTales)\Revived\RequestedReads"
LOG_FILE = os.path.join(BASE_DIR, "scraper.log")

# --- Webshare.io Rotating Residential Proxy ---
PROXY_USER = os.environ.get("WEBSHARE_USER", "username")
PROXY_PASS = os.environ.get("WEBSHARE_PASS", "password")
PROXY_HOST = "p.webshare.io"
PROXY_PORT = 80
PROXY_URL = f"http://{PROXY_USER}:{PROXY_PASS}@{PROXY_HOST}:{PROXY_PORT}"

CHANNELS = [
    "https://www.youtube.com/@Requestedreads/shorts",
    # Add more channels here
]

MIN_VIEWS = 50_000

# Accepted English subtitle-track key prefixes, in priority order.
ENGLISH_KEYS_PRIORITY = ("en", "en-orig", "en-US", "en-GB", "en-CA", "en-AU")

# ==============================================================================
# INITIALIZATION
# ==============================================================================
def setup_logging():
    os.makedirs(BASE_DIR, exist_ok=True)
    logger = logging.getLogger("scraper")
    logger.setLevel(logging.INFO)
    fmt = logging.Formatter("%(asctime)s [%(levelname)s] %(message)s", "%H:%M:%S")
    
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

# ==============================================================================
# DATA SERIALIZATION & STATE
# ==============================================================================
def load_processed_ids(filepath):
    """Loads processed Video IDs into an O(1) set from the JSONL file."""
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
    """Incrementally writes a single record to prevent data loss on crash."""
    with open(filepath, 'a', encoding='utf-8') as f:
        f.write(json.dumps(record, ensure_ascii=False) + '\n')

# ==============================================================================
# PROXY HELPERS
# ==============================================================================
def sticky_proxy_for(video_id: str) -> str:
    """
    Plain rotating proxy by default. Pass a video ID as sticky_key to pin
    that video's metadata + subtitle requests to the same residential IP.
    """
    numeric_session = int(hashlib.md5(video_id.encode()).hexdigest()[:8], 16) % 1000000
    return f"http://{PROXY_USER}-us-{numeric_session}:{PROXY_PASS}@{PROXY_HOST}:{PROXY_PORT}"

# ==============================================================================
# EXTRACTION & CLEANING
# ==============================================================================
def _find_english_track(subs: dict, auto_subs: dict):
    """Returns (track_list, source_label) for the best available English track."""
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
    """Fetches JSON3 transcript via requests using the rotating sticky proxy."""
    subs = info.get('subtitles', {}) or {}
    auto_subs = info.get('automatic_captions', {}) or {}
    target_sub, source_label = _find_english_track(subs, auto_subs)
    
    if not target_sub:
        return None

    json3_entry = next((s for s in target_sub if s.get('ext') == 'json3'), None)
    if not json3_entry:
        log.debug("No json3 track available (source=%s)", source_label)
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
                    log.warning("Subtitle response wasn't valid JSON (source=%s)", source_label)
                    return None
                
                lines = []
                for event in data.get('events', []):
                    if 'segs' not in event:
                        continue
                    segment_text = ''.join(s.get('utf8', '') for s in event['segs'])
                    
                    # Clean the text: remove brackets, fix spacing, unescape HTML
                    clean_text = re.sub(r'\[.*?\]|\(.*?\)', '', segment_text)
                    clean_text = re.sub(r'\s+', ' ', clean_text).strip()
                    clean_text = html.unescape(clean_text)
                    
                    if clean_text:
                        lines.append(clean_text)
                        
                final_text = " ".join(lines)
                
                # Restore punctuation if the string is not empty
                if final_text.strip():
                    return punct_model.restore_punctuation(final_text)
                return None
                
            log.debug("Subtitle fetch got HTTP %s (attempt %d)", response.status_code, attempt)
            time.sleep(2 ** attempt + random.uniform(1.0, 3.0))
        except requests.RequestException as e:
            log.debug("Subtitle fetch error: %s (attempt %d)", e, attempt)
            time.sleep(1.5)
            
    return None

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
        
        # Robust regex parsing for the channel handle
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
        
        # Pre-filter candidate videos (excluding view count check to avoid extract_flat bugs)
        candidate_entries = [
            e for e in all_entries
            if e.get('id') not in existing_ids
        ]
        
        log.info("Found %d valid candidates to process.", len(candidate_entries))

        for idx, entry in enumerate(candidate_entries, 1):
            vid_id = entry['id']
            url = f"https://www.youtube.com/watch?v={vid_id}"
            proxy_for_video = sticky_proxy_for(vid_id)
            
            video_opts = {
                'quiet': True,
                'no_warnings': True,
                'skip_download': True,
                'proxy': proxy_for_video,
                'writesubtitles': False,
            }
            
            try:
                with yt_dlp.YoutubeDL(video_opts) as ydl_video:
                    info = ydl_video.extract_info(url, download=False)
                    
                # View count check moved here where metadata is fully loaded
                views = info.get('view_count') or 0
                if views < MIN_VIEWS:
                    log.info("Skipped (%d/%d): %s has %d views (below threshold)", idx, len(candidate_entries), vid_id, views)
                    continue
                    
                transcript = fetch_transcript_in_memory(info, proxy_for_video)
                
                if transcript:
                    title = info.get('title') or ""
                    record = {
                        "VideoID": vid_id,
                        "Channel": channel_info.get('channel') or channel_info.get('uploader', 'Unknown'),
                        "Title": title,
                        "Duration": info.get('duration'),
                        "Views": views,
                        "Transcript": transcript
                    }
                    
                    append_to_jsonl(channel_file, record)
                    existing_ids.add(vid_id)
                    log.info("Saved (%d/%d): %s...", idx, len(candidate_entries), title[:45])
                else:
                    log.info("No transcript for: %s", url)
                    
            except Exception as e:
                log.error("Failed %s: %s", url, e)

            # Delay to avoid channel-level rate limits
            time.sleep(random.uniform(1.2, 2.8))

if __name__ == "__main__":
    process_channels(CHANNELS)