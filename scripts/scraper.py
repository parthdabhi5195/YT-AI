import yt_dlp
import pandas as pd
import os
import requests
import time
import random
import json
import re
from datetime import datetime

# ==========================================
# 1. SETUP 
# ==========================================
CHANNEL_URL = "https://www.youtube.com/@Requestedreads/shorts"

BASE_DIR = r"C:\Users\parth\Desktop\Youtube\YouTube Account (TheUnfilteredTales)\Revived\RequestedReads"
EXCEL_FILE = os.path.join(BASE_DIR, "Master_YouTube_Analysis.xlsx")
QUEUE_FILE = os.path.join(BASE_DIR, "queue.json")

def extract_video_id(url):
    """Extracts the exact 11-character YouTube video ID to prevent format mismatches."""
    match = re.search(r'(?:v=|shorts\/|youtu\.be\/)([A-Za-z0-9_-]{11})', str(url))
    return match.group(1) if match else str(url)

# ==========================================
# 2. CORE FUNCTIONS
# ==========================================
def format_time(seconds):
    if seconds is None: return "0:00"
    mins = int(seconds // 60)
    secs = int(seconds % 60)
    return f"{mins}:{secs:02d}"

def get_transcript_manually(info, url):
    try:
        subs = info.get('subtitles', {})
        auto_subs = info.get('automatic_captions', {})
        target_sub = subs.get('en') or auto_subs.get('en')
        if not target_sub: return "Transcript extraction failed: No English transcript found."
        
        sub_url = next((s['url'] for s in target_sub if s.get('ext') == 'json3'), target_sub[0]['url'])
        headers = {
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/123.0.0.0 Safari/537.36',
            'Accept-Language': 'en-US,en;q=0.9',
            'Accept': 'text/html,application/xhtml+xml,application/xml;q=0.9,image/avif,image/webp,*/*;q=0.8',
            'Referer': 'https://www.youtube.com/'
        }
        response = requests.get(sub_url, headers=headers, timeout=15)
        
        if response.status_code != 200: return f"Transcript extraction failed: Error {response.status_code}"
        
        data = response.json()
        lines = [f"[{format_time(e.get('tStartMs', 0)/1000)}] {''.join([s['utf8'] for s in e['segs'] if 'utf8' in s]).strip()}" for e in data.get('events', []) if 'segs' in e]
        return "\n".join(lines)
    except Exception as e:
        return "Transcript extraction failed: YouTube blocked data (429/CAPTCHA)."

def get_shorts_data(url):
    ydl_opts = {'quiet': True, 'no_warnings': True, 'skip_download': True, 'user_agent': 'Mozilla/5.0'}
    with yt_dlp.YoutubeDL(ydl_opts) as ydl:
        try:
            info = ydl.extract_info(url, download=False)
            print(f"Fetching data for: {info.get('title')}")
            ts = info.get('timestamp')
            p_time = datetime.fromtimestamp(ts).strftime('%Y-%m-%d %H:%M:%S') if ts else datetime.strptime(info.get('upload_date'), '%Y%m%d').strftime('%Y-%m-%d 00:00:00')
            return {
                "Title": info.get('title'), "Duration": format_time(info.get('duration')),
                "Upload Date/Time": p_time, "Views": info.get('view_count'),
                "Likes": info.get('like_count'), "Comments": info.get('comment_count'),
                "Transcript": get_transcript_manually(info, url), "URL": url
            }
        except Exception as e:
            print(f"Error scraping {url}: {e}")
            return None

# ==========================================
# 3. AUTOMATION & AUTO-CLEANER
# ==========================================
def build_delta_queue():
    print("\n[PHASE 1] Performing High-Speed Delta Sync & Checking for Duplicates...")
    
    existing_ids = set()
    if os.path.exists(EXCEL_FILE):
        try:
            df = pd.read_excel(EXCEL_FILE)
            if 'URL' in df.columns:
                # Use exact Video IDs to find duplicates
                df['VideoID'] = df['URL'].apply(extract_video_id)
                existing_ids = set(df['VideoID'].tolist())
                
                # AUTO-CLEANER: If the file has more rows than unique IDs, it cleans the file instantly.
                if len(df) > len(existing_ids):
                    print(f" -> [!] Found {len(df) - len(existing_ids)} duplicate entries from format mismatch. Auto-cleaning the Master file...")
                    df.drop_duplicates(subset=['VideoID'], keep='last', inplace=True)
                    df.drop(columns=['VideoID'], inplace=True)
                    df.to_excel(EXCEL_FILE, index=False)
                    print(" -> [✓] Master file successfully cleaned!")
                    
            print(f" -> Found {len(existing_ids)} unique videos already in Master Excel.")
        except: pass

    ydl_opts = {'extract_flat': True, 'quiet': True, 'no_warnings': True}
    print(" -> Fetching complete channel list from YouTube...")
    with yt_dlp.YoutubeDL(ydl_opts) as ydl:
        info = ydl.extract_info(CHANNEL_URL, download=False)
        all_urls = [f"https://www.youtube.com/watch?v={e['id']}" for e in info.get('entries', [])]

    # Filter out what we already have using the strict 11-Character ID
    new_urls = [u for u in all_urls if extract_video_id(u) not in existing_ids]
    
    if not new_urls:
        print("\n[✓] Excel sheet is completely up to date. No new videos to scrape.")
        return False

    chunks = [new_urls[i:i + 9] for i in range(0, len(new_urls), 9)]
    with open(QUEUE_FILE, "w") as f: json.dump(chunks, f)
    print(f"\nSUCCESS: Found {len(new_urls)} entirely new videos. Built {len(chunks)} batches.")
    return True

def run_automated_batch():
    with open(QUEUE_FILE, "r") as f: queue = json.load(f)
    if not queue: return

    current_batch = queue[0]
    print(f"\n--- [PHASE 2] STARTING BATCH OF {len(current_batch)}. {len(queue)-1} batches remain. ---")
    
    results = []
    for i, url in enumerate(current_batch):
        data = get_shorts_data(url)
        
        if data:
            if "Transcript extraction failed" in data["Transcript"]:
                print(f"\n[!!!] FATAL ABORT [!!!]")
                print("YouTube blocked the transcript. Your IP is flagged (429 Error).")
                print("ABORTING batch to protect Excel file. Toggle Hotspot to reset IP.")
                exit(1) 

            results.append(data)
        
        if i < len(current_batch) - 1:
            if (i + 1) % 3 == 0:
                cooldown = random.uniform(60, 120)
                print(f"Taking a long 'Human Break' ({cooldown:.2f}s)...")
                time.sleep(cooldown)
            else:
                wait_time = random.uniform(15, 20)
                print(f"Sleeping for {wait_time:.2f} seconds...")
                time.sleep(wait_time)

    if not results: return
    new_data = pd.DataFrame(results)

    try:
        if os.path.exists(EXCEL_FILE):
            existing_df = pd.read_excel(EXCEL_FILE)
            cols_to_remove = ['Upload Date', 'Upload Time']
            existing_df = existing_df.drop(columns=[c for c in cols_to_remove if c in existing_df.columns])
            final_df = pd.concat([existing_df, new_data], ignore_index=True)
        else:
            final_df = new_data

        # Final Sort & Strict VideoID Cleanup
        final_df['VideoID'] = final_df['URL'].apply(extract_video_id)
        final_df.drop_duplicates(subset=['VideoID'], keep='last', inplace=True)
        
        final_df['Upload Date/Time'] = pd.to_datetime(final_df['Upload Date/Time'], errors='coerce')
        final_df = final_df.dropna(subset=['Upload Date/Time'])
        final_df.sort_values(by='Upload Date/Time', ascending=False, inplace=True)
        final_df['Upload Date/Time'] = final_df['Upload Date/Time'].dt.strftime('%Y-%m-%d %H:%M:%S')
        
        # Remove Temp ID Column and Save
        final_df = final_df.drop(columns=['VideoID'])
        final_df = final_df.reindex(columns=["Title", "Duration", "Upload Date/Time", "Views", "Likes", "Comments", "Transcript", "URL"])

        final_df.to_excel(EXCEL_FILE, index=False)
        print(f"\nSUCCESS! Added {len(results)} new videos. Master file updated.")
        
        queue.pop(0)
        with open(QUEUE_FILE, "w") as f: json.dump(queue, f)
            
    except PermissionError:
        print("\n--- FATAL ERROR --- The Excel file is currently OPEN!")
        print("You must CLOSE Excel for the script to save data.")
        exit(1)
    except Exception as e:
        print(f"An error occurred during save: {e}")
        exit(1)

if __name__ == "__main__":
    if not os.path.exists(QUEUE_FILE) or os.path.getsize(QUEUE_FILE) <= 2: 
        if not build_delta_queue(): exit()
    
    run_automated_batch()