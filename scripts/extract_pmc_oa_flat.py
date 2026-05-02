#!/usr/bin/env python
"""Extract just the first N images from PMC-OA images.zip, sequentially.

Sequential zip read on EFS is ~100x faster than parallel random access.
Writes to a flat directory we can then hand off to the multi-GPU extractor.
"""
import json, zipfile, os, sys, time
from pathlib import Path

DATA = Path("/efs/user_folders/dnshalam/datasets/pmc_oa")
IMG_DIR = DATA / "images_flat"  # flat target dir
IMG_DIR.mkdir(exist_ok=True)

N = int(sys.argv[1]) if len(sys.argv) > 1 else 500_000

# Get the list of image basenames we need (first N rows of jsonl)
wanted = set()
with open(DATA / "pmc_oa.jsonl") as f:
    for i, line in enumerate(f):
        if i >= N: break
        wanted.add(json.loads(line)["image"])
print(f"wanted {len(wanted)} images", flush=True)

# Open zip once, iterate sequentially, extract only wanted
t0 = time.time()
done = skipped = 0
with zipfile.ZipFile(DATA / "images.zip") as zf:
    for info in zf.infolist():
        base = os.path.basename(info.filename)
        if base not in wanted: continue
        dst = IMG_DIR / base
        if dst.exists():
            skipped += 1; done += 1; continue
        with zf.open(info) as src, open(dst, "wb") as out:
            out.write(src.read())
        done += 1
        if done % 5000 == 0:
            rate = done / (time.time() - t0)
            eta = (len(wanted) - done) / max(rate, 1)
            print(f"  {done}/{len(wanted)}  ({rate:.0f}/s, ETA {eta/60:.1f}min, skipped {skipped})", flush=True)
print(f"done: {done} files in {(time.time()-t0)/60:.1f} min (skipped {skipped})")
