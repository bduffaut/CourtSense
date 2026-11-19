"""
Download and preprocess the training data into Data/.

    python -m scripts.prepare_data            # download + extract + cache
    python -m scripts.prepare_data --skip-download

Sources (Hugging Face mirrors of the original public releases):
  * TrackNet tennis dataset (Huang et al., 2019): 10 broadcast matches,
    ~19.8k 1280x720 frames with per-frame ball positions.       ~2.6 GB
  * TennisCourtDetector (yastrebksv): 8,841 broadcast frames from many
    matches/surfaces, 14 court keypoints each (MIT licence).      ~7.3 GB

The 10 GB of raw data stays out of git (Data/ is gitignored).
"""

import argparse
import shutil
import time
import urllib.error
import urllib.request
import zipfile
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

from PIL import Image

from utils.datasets import BALL_DIR, COURT_DIR, DATA

SOURCES = {
    "tennis_tracknet.zip":
        "https://huggingface.co/datasets/oweng/tracknet_preprocessd_data/resolve/main/tennis.zip",
    "tennis_court_det_dataset.zip":
        "https://huggingface.co/datasets/Gholamreza/tennis_court_keypoints_dataset/resolve/main/tennis_court_det_dataset.zip",
}
DL_DIR = DATA / "_downloads"


def _fetch(url: str, dest: Path, chunk: int = 1 << 20):
    """Resumable download with plain urllib (works the same on Windows/macOS/Linux)."""
    have = dest.stat().st_size if dest.exists() else 0
    req = urllib.request.Request(url, headers={"Range": f"bytes={have}-"} if have else {})
    try:
        resp = urllib.request.urlopen(req)
    except urllib.error.HTTPError as e:
        if e.code == 416:          # already complete
            return
        raise
    total = have + int(resp.headers.get("Content-Length", 0))
    if have and resp.status != 206:   # server ignored Range: start over
        have = 0
    with resp, open(dest, "ab" if have else "wb") as f:
        done, last = have, time.time()
        while block := resp.read(chunk):
            f.write(block); done += len(block)
            if time.time() - last > 10:
                print(f"  {dest.name}: {done/1e9:.2f} / {total/1e9:.2f} GB", flush=True)
                last = time.time()


def download():
    DL_DIR.mkdir(parents=True, exist_ok=True)
    for name, url in SOURCES.items():
        print(f"Downloading {name} ...", flush=True)
        _fetch(url, DL_DIR / name)


def _extract(zip_path: Path, member_prefix: str, dest: Path):
    """Extract members under `member_prefix` into `dest`, stripping the prefix."""
    with zipfile.ZipFile(zip_path) as z:
        members = [m for m in z.infolist() if m.filename.startswith(member_prefix)
                   and not m.is_dir()]
        for i, m in enumerate(members):
            out = dest / m.filename[len(member_prefix):]
            if out.exists() and out.stat().st_size == m.file_size:
                continue
            out.parent.mkdir(parents=True, exist_ok=True)
            with z.open(m) as src, open(out, "wb") as dst:
                shutil.copyfileobj(src, dst)
            if i % 2000 == 0:
                print(f"  {i}/{len(members)}")


def extract():
    print("Extracting TrackNet ...")
    _extract(DL_DIR / "tennis_tracknet.zip", "tennis/Dataset/", BALL_DIR)
    print("Extracting court keypoints ...")
    _extract(DL_DIR / "tennis_court_det_dataset.zip", "data/", COURT_DIR)


def _cache_one(src: Path, dst: Path):
    if dst.exists():
        return
    Image.open(src).convert("RGB").save(dst, quality=92)


def cache_court_images():
    """PNG -> JPEG at the original 1280x720 (PNG decoding is the training bottleneck)."""
    out = COURT_DIR / "cache"
    out.mkdir(exist_ok=True)
    srcs = sorted((COURT_DIR / "images").glob("*.png"))
    print(f"Caching {len(srcs)} court images as JPEG ...")
    with ThreadPoolExecutor(8) as ex:
        list(ex.map(lambda p: _cache_one(p, out / f"{p.stem}.jpg"), srcs))


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--skip-download", action="store_true")
    args = ap.parse_args()
    if not args.skip_download:
        download()
    extract()
    cache_court_images()
    print("Done. Next: python -m modules.cnn.train --model keypoint")
