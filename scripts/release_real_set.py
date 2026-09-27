"""Make the public release of the real word set: the written items only, with any corrections
of their labels applied and recorded.

    python scripts/release_real_set.py --set work/real/real_test.tar \\
        --corrections real_words/corrections.json --out real_words

--set is the set cut by scripts/cut_writing_pages.py (folder or .tar: images/, labels.tsv,
manifest.json). Its manifest lists every item of the writing pages; the release keeps the
written ones, so the texts of the items not yet written stay private. --corrections maps an
image file to its corrected text and the reason, for example

    {"000035.png": {"text": "...", "reason": "the full stop was not written"}}

Writes images/, labels.tsv and manifest.json (the written items, the corrections applied,
listed under "corrections"). README.md and LICENSE of the release are kept as they are.
--summary writes the released set's counts (no words) as a results file.
"""

import argparse
import csv
import io
import json
import shutil
import tarfile
from collections import Counter
from pathlib import Path


def read_set(path):
    """Folder or .tar of a cut set -> (manifest, labels {file: text}, image bytes {file: bytes})."""
    path = Path(path)
    if path.is_dir():
        files = {p.relative_to(path).as_posix(): p.read_bytes() for p in path.rglob("*") if p.is_file()}
    else:                                       # member names as mayek_htr.data reads them
        with tarfile.open(path) as tar:
            files = {m.name.lstrip("./"): tar.extractfile(m).read() for m in tar.getmembers() if m.isfile()}
    manifest = json.loads(files["manifest.json"].decode("utf-8"))
    labels = {r[0]: r[1] for r in csv.reader(io.StringIO(files["labels.tsv"].decode("utf-8")), delimiter="\t")
              if r}
    images = {k.split("/", 1)[1]: v for k, v in files.items() if k.startswith("images/")}
    return manifest, labels, images


def release(manifest, labels, images, corrections, out):
    out = Path(out)
    written = [it for it in manifest["items"] if it.get("status") == "written"]
    if set(corrections) - {it["file"] for it in written}:
        raise ValueError(f"corrections for files not in the set: {sorted(set(corrections) - set(labels))}")
    changes = []
    items = []
    for it in written:
        text = labels[it["file"]]
        if it["text"] != text:
            raise ValueError(f"{it['file']}: manifest and labels.tsv disagree")
        if it["file"] in corrections:
            new = corrections[it["file"]]["text"]
            changes.append({"item": it["item"], "file": it["file"], "was": text, "now": new,
                            "reason": corrections[it["file"]]["reason"]})
            text = new
        items.append({"item": it["item"], "page": it["page"], "file": it["file"], "text": text, "kind": it["kind"],
                      "flags": it.get("flags", []), "size": it.get("size")})
    if (out / "images").exists():
        shutil.rmtree(out / "images")
    (out / "images").mkdir(parents=True)
    for it in items:
        (out / "images" / it["file"]).write_bytes(images[it["file"]])
    with open(out / "labels.tsv", "w", encoding="utf-8", newline="") as f:
        for it in items:
            f.write(f"{it['file']}\t{it['text']}\n")
    info = {"format": manifest.get("format"), "title": manifest.get("title"),
            "selection": {k: v for k, v in manifest.get("selection", {}).items()},
            "pages": [{k: p[k] for k in ("page", "corner_squares", "printed_ink_missing") if k in p}
                      for p in manifest.get("pages", [])],
            "words": len(items), "characters": sum(len(it["text"]) for it in items),
            "corrections": changes, "items": items}
    (out / "manifest.json").write_text(json.dumps(info, indent=1, ensure_ascii=False), encoding="utf-8")
    return info


def summary(info):
    """The released set's counts, without its words."""
    items = info["items"]
    chars = Counter(ch for it in items for ch in it["text"])
    return {"words": len(items), "by_kind": dict(Counter(it["kind"] for it in items)),
            "characters": sum(chars.values()), "distinct_characters": len(chars),
            "character_counts": dict(sorted(chars.items(), key=lambda kv: -kv[1])),
            "ending_with_full_stop": sum(it["text"].endswith("꯫") for it in items),
            "flagged": dict(Counter(f for it in items for f in it["flags"])),
            "corrections": [{"item": c["item"], "file": c["file"], "reason": c["reason"]} for c in info["corrections"]]}


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--set", required=True, help="the cut set (folder or .tar)")
    ap.add_argument("--corrections", help="JSON: image file -> {text, reason}")
    ap.add_argument("--out", required=True)
    ap.add_argument("--summary", help="results .json: the released set's counts")
    args = ap.parse_args()
    corrections = json.loads(Path(args.corrections).read_text(encoding="utf-8")) if args.corrections else {}
    info = release(*read_set(args.set), corrections, args.out)
    if args.summary:
        Path(args.summary).write_text(json.dumps(summary(info), indent=1, ensure_ascii=False), encoding="utf-8")
    print(f"{info['words']} words, {info['characters']} characters, {len(info['corrections'])} corrected -> {args.out}")


if __name__ == "__main__":
    main()
