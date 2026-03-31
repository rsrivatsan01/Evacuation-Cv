# ─────────────────────────────────────────────
# download_video.py
# Downloads a sample crowd video for development
# and testing. Run once during Phase 1 setup.
#
# Usage:
#   python download_video.py
# ─────────────────────────────────────────────

import os
import sys
import requests
from tqdm import tqdm
from config import SAMPLE_VIDEO_PATH, VIDEO_DIR

# ── Video Sources ─────────────────────────────
# MOT17-09 is a publicly available sequence filmed
# inside a shopping mall — perfect for our use case.
# We use a mirror hosted by the MOT Challenge team.

VIDEO_SOURCES = [
    {
        "name": "MOT17 Mall Sequence (primary)",
        "url": "https://motchallenge.net/sequenceVideos/MOT17-09-raw.webm",
        "filename": "sample_crowd.mp4",
    },
    {
        "name": "Pexels Crowd Fallback",
        "url": "https://www.pexels.com/download/video/854985/",
        "filename": "sample_crowd.mp4",
    },
]

def download_file(url: str, dest_path: str, label: str) -> bool:
    """Download a file with a progress bar. Returns True on success."""
    try:
        print(f"\n⬇  Downloading: {label}")
        print(f"   URL : {url}")
        print(f"   Dest: {dest_path}\n")

        response = requests.get(url, stream=True, timeout=30)
        response.raise_for_status()

        total = int(response.headers.get("content-length", 0))
        with open(dest_path, "wb") as f, tqdm(
            total=total, unit="B", unit_scale=True, unit_divisor=1024
        ) as bar:
            for chunk in response.iter_content(chunk_size=8192):
                f.write(chunk)
                bar.update(len(chunk))

        print(f"\n✅ Saved to: {dest_path}")
        return True

    except Exception as e:
        print(f"❌ Failed: {e}")
        if os.path.exists(dest_path):
            os.remove(dest_path)
        return False


def main():
    os.makedirs(VIDEO_DIR, exist_ok=True)

    # Skip if already downloaded
    if os.path.exists(SAMPLE_VIDEO_PATH) and os.path.getsize(SAMPLE_VIDEO_PATH) > 0:
        size_mb = os.path.getsize(SAMPLE_VIDEO_PATH) / (1024 * 1024)
        print(f"✅ Sample video already exists ({size_mb:.1f} MB). Skipping download.")
        print(f"   Path: {SAMPLE_VIDEO_PATH}")
        return

    # Try each source in order
    for source in VIDEO_SOURCES:
        dest = os.path.join(VIDEO_DIR, source["filename"])
        success = download_file(source["url"], dest, source["name"])
        if success:
            print("\n🎉 Video ready for use in development!")
            return

    # All sources failed — give manual instructions
    print("\n⚠️  Automatic download failed.")
    print("Please manually download a crowd scene video and place it at:")
    print(f"   {SAMPLE_VIDEO_PATH}")
    print("\nSuggested free sources:")
    print("  • https://motchallenge.net/data/MOT17/  (MOT17-09 sequence)")
    print("  • https://www.pexels.com/search/videos/crowd/  (free stock footage)")
    print("  • https://viratdata.org/  (VIRAT surveillance dataset)")
    sys.exit(1)


if __name__ == "__main__":
    main()
