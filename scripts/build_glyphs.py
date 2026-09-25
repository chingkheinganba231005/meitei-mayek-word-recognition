"""Character images for composing words, one store per split, from the first project's split.

    python -m mayek.split --zip TUMMHCD-TEST-TRAIN.zip --data-dir data     # the paper's split
    python scripts/build_glyphs.py --split-dir data/splits \\
        --duplicates results/tummhcd_audit_duplicates.csv --out-dir work/glyphs \\
        --stats results/glyph_store_stats.json

Writes train.npz, val.npz and test.npz (images, labels, keys), used as:
train for synthetic training words, val for synthetic validation words, test for
synthetic test words. Test images with a pixel-identical train image, and the images in
groups of identical pixels with conflicting labels, are left out (Phase 0 duplicates file).
"""

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from mayek_words.charset import TUMMHCD  # noqa: E402
from mayek_words.glyphs import GlyphStore  # noqa: E402


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--split-dir", required=True, help="folder with train.csv, val.csv and test.csv")
    ap.add_argument("--duplicates", help="results/tummhcd_audit_duplicates.csv from Phase 0")
    ap.add_argument("--out-dir", required=True)
    ap.add_argument("--stats", default="results/glyph_store_stats.json")
    args = ap.parse_args()

    out = Path(args.out_dir)
    out.mkdir(parents=True, exist_ok=True)
    stats = {"split_dir": str(args.split_dir), "duplicates": args.duplicates, "splits": {}}
    for split in ("train", "val", "test"):
        t0 = time.time()
        store = GlyphStore.from_split(args.split_dir, split, args.duplicates)
        store.save(out / f"{split}.npz")
        per_class = np.bincount(store.labels, minlength=len(TUMMHCD))
        stats["splits"][split] = {"images": len(store), "left_out": store.dropped,
                                  "fewest_per_class": int(per_class.min()),
                                  "per_class": {f"{c:03d} {TUMMHCD[c]}": int(n) for c, n in enumerate(per_class)}}
        print(f"{split}: {len(store):,} images ({store.dropped} left out), {time.time() - t0:.0f} s")
    Path(args.stats).parent.mkdir(parents=True, exist_ok=True)
    Path(args.stats).write_text(json.dumps(stats, indent=1, ensure_ascii=False), encoding="utf-8")
    print(f"written to {out} and {args.stats}")


if __name__ == "__main__":
    main()
