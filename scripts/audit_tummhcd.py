"""Audit TUMMHCD for writer information.

    python scripts/audit_tummhcd.py --zip /content/TUMMHCD-TEST-TRAIN.zip
    python scripts/audit_tummhcd.py --dir /content/drive/MyDrive/TUMMHCD

Phase 1 composes words from characters of one writer and needs a train/test
split in which no writer appears on both sides. The dataset paper does not say
whether writer identity survives in the release, so this script looks for it
in four places:

1. Folders: any folder level below the class folders (train_000 ... test_054).
2. File names: the name templates, and for every number in the names how it
   behaves (range, repeats within a class, recurrence across classes).
3. Other files in the archive (lists, spreadsheets, read-me files).
4. Hidden grouping. A writer's images share a style: scan size, stroke width,
   ink darkness, where the ink sits in the crop. If images were saved writer
   by writer, neighbouring files in a class correlate in style; if every
   writer filled one tabular form, files with the same number in different
   classes correlate. Both are measured against random pairs.

Style features are z-scored within each class first, so the shape of a
character does not count as style. Everything is written to one JSON file
(default results/tummhcd_audit.json).
"""

import argparse
import hashlib
import io
import json
import re
import time
import zipfile
from collections import Counter, defaultdict
from pathlib import Path, PurePosixPath

import numpy as np
from PIL import Image

IMAGE_EXTS = {".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff", ".pgm", ".gif"}
TEXT_EXTS = {".txt", ".csv", ".tsv", ".md", ".json", ".xml", ".html", ".htm"}
SYSTEM_NAMES = {".ds_store", "thumbs.db", "desktop.ini"}
CLASS_DIR = re.compile(r"^(train|test)_(\d{3})$", re.IGNORECASE)
DIGITS = re.compile(r"\d+")
LETTERS = re.compile(r"[A-Za-z]+")
FEATURES = ["height", "width", "ink_fraction", "paper_level", "ink_level", "stroke_width",
            "top_margin", "bottom_margin", "left_margin", "right_margin"]
LAGS = [1, 2, 3, 5, 10, 20, 50, 100, 200]
MIN_PAIRS = 30


# ---------------------------------------------------------------- reading

def open_source(zip_path=None, dir_path=None):
    """-> (entries, read) where each entry describes one file and read(entry) returns its bytes."""
    entries = []
    if zip_path:
        z = zipfile.ZipFile(zip_path)
        for k, info in enumerate(i for i in z.infolist() if not i.is_dir()):
            entries.append({"path": info.filename, "bytes": info.file_size,
                            "time": time.mktime(info.date_time + (0, 0, -1)), "order": k})
        return entries, lambda e: z.read(e["path"])
    root = Path(dir_path)
    for p in sorted(x for x in root.rglob("*") if x.is_file()):
        st = p.stat()
        entries.append({"path": p.relative_to(root).as_posix(), "bytes": st.st_size,
                        "time": st.st_mtime, "order": None})
    return entries, lambda e: (root / e["path"]).read_bytes()


def describe(entry):
    """Adds split, label, sub-folders, stem and extension from the path."""
    parts = PurePosixPath(entry["path"]).parts
    name = parts[-1]
    entry.update(name=name, stem=PurePosixPath(name).stem, ext=PurePosixPath(name).suffix.lower(),
                 split=None, label=None, sub=())
    for k in range(len(parts) - 2, -1, -1):
        m = CLASS_DIR.match(parts[k])
        if m:
            entry.update(split=m.group(1).lower(), label=int(m.group(2)), sub=tuple(parts[k + 1:-1]))
            break
    entry["system"] = (any(p == "__MACOSX" for p in parts) or name.startswith("._")
                       or name.lower() in SYSTEM_NAMES)
    return entry


def sha256(path, chunk=1 << 20):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for block in iter(lambda: f.read(chunk), b""):
            h.update(block)
    return h.hexdigest()


# ---------------------------------------------------------------- style features

def otsu(gray):
    hist = np.bincount(gray.ravel(), minlength=256).astype(np.float64)
    p = hist / hist.sum()
    omega = np.cumsum(p)
    mu = np.cumsum(p * np.arange(256))
    with np.errstate(divide="ignore", invalid="ignore"):
        between = (mu[-1] * omega - mu) ** 2 / (omega * (1 - omega))
    return int(np.argmax(np.nan_to_num(between)))


def style_features(data):
    """Image bytes -> (feature vector as in FEATURES, hash of the decoded pixels)."""
    img = Image.open(io.BytesIO(data))
    gray = np.asarray(img.convert("L"), dtype=np.uint8)
    pixel_hash = hashlib.md5(gray.tobytes() + str(gray.shape).encode()).hexdigest()
    h, w = gray.shape
    ink = gray <= otsu(gray)
    if ink.mean() > 0.5:  # light ink on dark paper
        ink = ~ink
    n = int(ink.sum())
    if n == 0 or n == ink.size:
        return [h, w] + [np.nan] * (len(FEATURES) - 2), pixel_hash
    pad = np.pad(ink, 1, constant_values=False)
    interior = pad[1:-1, 1:-1] & pad[:-2, 1:-1] & pad[2:, 1:-1] & pad[1:-1, :-2] & pad[1:-1, 2:]
    boundary = max(n - int(interior.sum()), 1)
    ys, xs = np.nonzero(ink)
    return [h, w, n / ink.size, float(np.median(gray[~ink])), float(np.median(gray[ink])),
            2.0 * n / boundary, ys.min() / h, (h - 1 - ys.max()) / h,
            xs.min() / w, (w - 1 - xs.max()) / w], pixel_hash


# ---------------------------------------------------------------- analyses

def template(stem):
    return DIGITS.sub("#", stem)


def natural_key(entry):
    return ([int(d) for d in DIGITS.findall(entry["stem"])], entry["stem"])


def analyse_names(images):
    templates = Counter(template(e["stem"]) for e in images)
    out = {"n_templates": len(templates),
           "templates": [{"template": t, "count": c} for t, c in templates.most_common(20)],
           "letter_tokens": [{"token": t, "count": c} for t, c in
                             Counter(t for e in images for t in LETTERS.findall(e["stem"])).most_common(30)],
           "numbers": []}
    top = templates.most_common(1)[0][0]
    rows = [e for e in images if template(e["stem"]) == top]
    for i in range(top.count("#")):
        by_class = defaultdict(list)
        for e in rows:
            by_class[(e["split"], e["label"])].append(int(DIGITS.findall(e["stem"])[i]))
        values = [v for vs in by_class.values() for v in vs]
        classes_per_value = Counter()
        for vs in by_class.values():
            classes_per_value.update(set(vs))
        n_classes = len(by_class)
        out["numbers"].append({
            "position": i, "template": top, "files": len(rows),
            "distinct_values": len(set(values)), "min": min(values), "max": max(values),
            "max_repeats_within_a_class": max(max(Counter(vs).values()) for vs in by_class.values()),
            "median_fill_of_range_per_class": float(np.median(
                [len(set(vs)) / (max(vs) - min(vs) + 1) for vs in by_class.values()])),
            "median_classes_per_value": float(np.median(list(classes_per_value.values()))),
            "share_of_values_in_half_the_classes_or_more": float(np.mean(
                [c >= n_classes / 2 for c in classes_per_value.values()])),
        })
    return out, top


def zscore_within_class(F, groups):
    Z = np.full_like(F, np.nan)
    for g in np.unique(groups):
        idx = np.flatnonzero(groups == g)
        mu = np.nanmean(F[idx], axis=0)
        sd = np.nanstd(F[idx], axis=0)
        Z[idx] = (F[idx] - mu) / np.where(sd > 0, sd, 1.0)
    return Z


def pair_correlation(Z, a, b):
    """Pearson correlation of every feature between the rows a[i] and b[i]."""
    per_feature = {}
    for j, name in enumerate(FEATURES):
        x, y = Z[a, j], Z[b, j]
        ok = np.isfinite(x) & np.isfinite(y)
        if ok.sum() < MIN_PAIRS or x[ok].std() == 0 or y[ok].std() == 0:
            per_feature[name] = None
        else:
            per_feature[name] = round(float(np.corrcoef(x[ok], y[ok])[0, 1]), 4)
    vals = [v for v in per_feature.values() if v is not None]
    return {"pairs": int(len(a)), "mean": round(float(np.mean(vals)), 4) if vals else None,
            "per_feature": per_feature}


def lag_pairs(sequences, lag):
    a = [s[:-lag] for s in sequences if len(s) > lag]
    b = [s[lag:] for s in sequences if len(s) > lag]
    if not a:
        return np.array([], int), np.array([], int)
    return np.concatenate(a), np.concatenate(b)


def order_study(Z, sequences, rng):
    """Correlation of neighbours at several lags in the given within-class orders, and for random orders."""
    out = {f"lag_{lag}": pair_correlation(Z, *lag_pairs(sequences, lag)) for lag in LAGS}
    shuffled = [rng.permutation(s) for s in sequences]
    out["random_order_lag_1"] = pair_correlation(Z, *lag_pairs(shuffled, 1))
    return out


def same_number_study(Z, rows_by_key, rng):
    """Pairs of files that carry the same number in different classes of one split, against a shuffle."""
    def pairs(groups):
        a, b = [], []
        for members in groups.values():
            members = sorted(members, key=lambda m: m[0])  # by class
            for (ca, ia), (cb, ib) in zip(members, members[1:]):
                if ca != cb:
                    a.append(ia)
                    b.append(ib)
        return np.array(a, int), np.array(b, int)

    groups = defaultdict(list)
    for (split, label, value), i in rows_by_key.items():
        groups[(split, value)].append((label, i))
    real = pair_correlation(Z, *pairs(groups))
    # shuffle: the numbers are permuted within each class, so pairs link unrelated files
    by_class = defaultdict(list)
    for (split, label, value), i in rows_by_key.items():
        by_class[(split, label)].append((value, i))
    shuffled_groups = defaultdict(list)
    for (split, label), items in by_class.items():
        values = rng.permutation([v for v, _ in items])
        for v, (_, i) in zip(values, items):
            shuffled_groups[(split, v)].append((label, i))
    return {"same_number": real, "shuffled_numbers": pair_correlation(Z, *pairs(shuffled_groups))}


def read_text_file(data, max_lines=20):
    text = data.decode("utf-8", errors="replace")
    return text.splitlines()[:max_lines]


# ---------------------------------------------------------------- main

def audit(zip_path=None, dir_path=None, max_per_class=None, seed=0):
    rng = np.random.default_rng(seed)
    entries, read = open_source(zip_path, dir_path)
    entries = [describe(e) for e in entries]
    system = [e for e in entries if e["system"]]
    files = [e for e in entries if not e["system"]]
    images = [e for e in files if e["ext"] in IMAGE_EXTS and e["label"] is not None]
    others = [e for e in files if not (e["ext"] in IMAGE_EXTS and e["label"] is not None)]
    if not images:
        raise SystemExit("no images inside train_NNN / test_NNN folders; is this the TUMMHCD archive?")

    report = {"source": {"zip": str(zip_path) if zip_path else None, "dir": str(dir_path) if dir_path else None,
                         "sha256": sha256(zip_path) if zip_path else None},
              "seed": seed, "max_per_class": max_per_class}

    per_split = Counter(e["split"] for e in images)
    per_class = defaultdict(dict)
    for (split, label), c in sorted(Counter((e["split"], e["label"]) for e in images).items()):
        per_class[split][f"{label:03d}"] = c
    report["counts"] = {"entries": len(entries), "system_files_skipped": len(system),
                        "class_images": len(images), "other_files": len(others),
                        "per_split": dict(per_split), "per_class": per_class,
                        "extensions": dict(Counter(e["ext"] for e in images))}

    subs = Counter(e["sub"] for e in images)
    report["structure"] = {
        "top_level": sorted({PurePosixPath(e["path"]).parts[0] for e in files}),
        "images_in_subfolders_below_class_folder": sum(c for s, c in subs.items() if s),
        "subfolder_names": [{"subfolders": "/".join(s), "images": c} for s, c in subs.most_common(30) if s],
        "distinct_subfolder_paths": sum(1 for s in subs if s),
    }

    report["other_files"] = []
    for e in others:
        item = {"path": e["path"], "bytes": e["bytes"]}
        if e["ext"] in TEXT_EXTS and e["bytes"] < 200_000:
            item["first_lines"] = read_text_file(read(e))
        report["other_files"].append(item)

    names, top_template = analyse_names(images)
    report["names"] = names

    by_class = defaultdict(list)
    for e in images:
        by_class[(e["split"], e["label"])].append(e)

    times = np.array([e["time"] for e in images], float)
    report["timestamps"] = {
        "distinct": int(len(np.unique(times))),
        "earliest": time.strftime("%Y-%m-%d %H:%M:%S", time.localtime(times.min())),
        "latest": time.strftime("%Y-%m-%d %H:%M:%S", time.localtime(times.max())),
        "median_distinct_per_class": float(np.median(
            [len({e["time"] for e in members}) for members in by_class.values()])),
    }

    # style features, optionally on the first max_per_class files of every class (name order)
    chosen = []
    for key in sorted(by_class):
        members = sorted(by_class[key], key=natural_key)
        chosen += members[:max_per_class] if max_per_class else members
    t0 = time.time()
    F, hashes = [], []
    for e in chosen:
        f, h = style_features(read(e))
        F.append(f)
        hashes.append(h)
    F = np.array(F, float)
    groups = np.array([f"{e['split']}_{e['label']:03d}" for e in chosen])
    Z = zscore_within_class(F, groups)
    report["style"] = {"images": len(chosen), "seconds": round(time.time() - t0, 1), "features": FEATURES,
                       "feature_median": dict(zip(FEATURES, np.round(np.nanmedian(F, 0), 4).tolist())),
                       "feature_p05": dict(zip(FEATURES, np.round(np.nanpercentile(F, 5, 0), 4).tolist())),
                       "feature_p95": dict(zip(FEATURES, np.round(np.nanpercentile(F, 95, 0), 4).tolist())),
                       "orders": {}}

    row = {id(e): i for i, e in enumerate(chosen)}
    sequences = {"name": [], "archive": [], "time": []}
    for key in sorted(by_class):
        members = [e for e in by_class[key] if id(e) in row]
        sequences["name"].append(np.array([row[id(e)] for e in sorted(members, key=natural_key)]))
        if zip_path:
            sequences["archive"].append(np.array([row[id(e)] for e in sorted(members, key=lambda e: e["order"])]))
        sequences["time"].append(np.array([row[id(e)] for e in sorted(members, key=lambda e: (e["time"],) + tuple(natural_key(e)[0]))]))
    for name, seqs in sequences.items():
        if not seqs or (name == "time" and report["timestamps"]["distinct"] < 2):
            continue
        report["style"]["orders"][name] = order_study(Z, seqs, rng)

    report["style"]["same_number_across_classes"] = {}
    for pos in range(top_template.count("#")):
        rows_by_key = {}
        for e in chosen:
            if template(e["stem"]) == top_template:
                rows_by_key[(e["split"], e["label"], int(DIGITS.findall(e["stem"])[pos]))] = row[id(e)]
        report["style"]["same_number_across_classes"][f"number_{pos}"] = same_number_study(Z, rows_by_key, rng)

    groups_by_hash = defaultdict(list)
    for e, h in zip(chosen, hashes):
        groups_by_hash[h].append(e)
    dup = [g for g in groups_by_hash.values() if len(g) > 1]
    report["duplicates"] = {
        "groups_of_identical_pixels": len(dup),
        "images_in_those_groups": sum(len(g) for g in dup),
        "groups_spanning_train_and_test": sum(1 for g in dup if len({e["split"] for e in g}) > 1),
        "groups_spanning_classes": sum(1 for g in dup if len({(e["split"], e["label"]) for e in g}) > 1),
        "examples": [[e["path"] for e in g[:4]] for g in dup[:10]],
    }
    report["notes"] = notes(report)
    return report


def notes(r):
    out = []
    s = r["structure"]
    if s["images_in_subfolders_below_class_folder"]:
        out.append(f"{s['images_in_subfolders_below_class_folder']} images sit in "
                   f"{s['distinct_subfolder_paths']} sub-folders below the class folders; check whether "
                   "these are writers.")
    else:
        out.append("No folder level below the class folders.")
    t = r["names"]["templates"][0]
    out.append(f"Most common file-name template: '{t['template']}' ({t['count']} of "
               f"{r['counts']['class_images']} images, {r['names']['n_templates']} templates in total).")
    for n in r["names"]["numbers"]:
        out.append(f"Number {n['position']} in '{n['template']}': {n['distinct_values']} values "
                   f"({n['min']}-{n['max']}), up to {n['max_repeats_within_a_class']} repeats in a class, "
                   f"fills {n['median_fill_of_range_per_class']:.0%} of its range per class, "
                   f"median value appears in {n['median_classes_per_value']:.0f} classes.")
    if r["other_files"]:
        out.append(f"{len(r['other_files'])} other files in the archive: "
                   + ", ".join(o["path"] for o in r["other_files"][:10]))
    for name, study in r["style"]["orders"].items():
        out.append(f"Style correlation of neighbours in {name} order: lag 1 {study['lag_1']['mean']}, "
                   f"lag 10 {study['lag_10']['mean']}, lag 100 {study['lag_100']['mean']}; "
                   f"random order {study['random_order_lag_1']['mean']}.")
    for name, study in r["style"]["same_number_across_classes"].items():
        out.append(f"Same {name.replace('_', ' ')} in different classes: style correlation "
                   f"{study['same_number']['mean']} ({study['same_number']['pairs']} pairs); "
                   f"shuffled {study['shuffled_numbers']['mean']}.")
    d = r["duplicates"]
    out.append(f"{d['groups_of_identical_pixels']} groups of pixel-identical images "
               f"({d['groups_spanning_train_and_test']} span train and test).")
    return out


def to_json(obj):
    if isinstance(obj, dict):
        return {str(k): to_json(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [to_json(v) for v in obj]
    if isinstance(obj, (np.integer,)):
        return int(obj)
    if isinstance(obj, (np.floating,)):
        return None if np.isnan(obj) else float(obj)
    if isinstance(obj, float) and np.isnan(obj):
        return None
    return obj


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    src = ap.add_mutually_exclusive_group(required=True)
    src.add_argument("--zip", help="TUMMHCD-TEST-TRAIN.zip (copy it off Google Drive first; reading is faster)")
    src.add_argument("--dir", help="an extracted copy of the archive")
    ap.add_argument("--out", default="results/tummhcd_audit.json")
    ap.add_argument("--max-per-class", type=int, default=None,
                    help="style features for only the first N files of each class (quick check)")
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    report = to_json(audit(args.zip, args.dir, args.max_per_class, args.seed))
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(report, indent=1, ensure_ascii=False))
    print("\n".join(report["notes"]))
    print(f"\nwritten to {out}")


if __name__ == "__main__":
    main()
