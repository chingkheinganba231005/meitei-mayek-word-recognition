import io
import json
import random
import sys
import zipfile
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
import audit_tummhcd as audit  # noqa: E402

CLASSES = 5
WRITERS = 8


def glyph(label, writer, rng):
    """A character image whose size and stroke width depend on the writer, as real handwriting does."""
    side = 18 + 4 * writer + rng.randint(-1, 1)
    img = Image.new("L", (side, side), 235 - 5 * (writer % 3))
    d = ImageDraw.Draw(img)
    width = 1 + writer % 3
    m = 3
    shapes = [lambda: d.ellipse([m, m, side - m, side - m], outline=20, width=width),
              lambda: d.line([m, m, side - m, side - m], fill=20, width=width),
              lambda: d.rectangle([m, m, side - m, side - m], outline=20, width=width),
              lambda: d.line([m, side // 2, side - m, side // 2], fill=20, width=width),
              lambda: d.polygon([(side // 2, m), (side - m, side - m), (m, side - m)], outline=20)]
    shapes[label]()
    buf = io.BytesIO()
    img.save(buf, "PNG")
    return buf.getvalue()


def make_zip(path, name_of, per_writer=6, extra=None):
    """name_of(split, label, writer, index) -> path inside the class folder."""
    rng = random.Random(0)
    with zipfile.ZipFile(path, "w") as z:
        for split, reps in (("train", per_writer), ("test", 2)):
            for label in range(CLASSES):
                index = 0
                for writer in range(WRITERS):  # saved writer by writer
                    for _ in range(reps):
                        index += 1
                        inner = name_of(split, label, writer, index)
                        z.writestr(f"TUMMHCD/TUMMHCD{split}/{split}_{label:03d}/{inner}", glyph(label, writer, rng))
        for name, data in (extra or {}).items():
            z.writestr(name, data)
    return path


def test_names_folders_and_other_files(tmp_path):
    extra = {"TUMMHCD/readme.txt": b"TUMMHCD\nwriter list not included\n",
             "__MACOSX/._x.png": b"junk"}
    zp = make_zip(tmp_path / "a.zip", lambda s, l, w, i: f"img{i}.png", extra=extra)
    r = audit.audit(zip_path=zp)
    assert r["counts"]["class_images"] == 2 * CLASSES * WRITERS * 4
    assert r["counts"]["system_files_skipped"] == 1
    assert r["structure"]["images_in_subfolders_below_class_folder"] == 0
    assert r["names"]["templates"][0]["template"] == "img#"
    num = r["names"]["numbers"][0]
    assert num["max_repeats_within_a_class"] == 1 and num["median_fill_of_range_per_class"] == 1.0
    assert r["other_files"][0]["path"] == "TUMMHCD/readme.txt"
    assert r["other_files"][0]["first_lines"][1] == "writer list not included"


def test_writer_blocks_show_up_as_neighbour_correlation(tmp_path):
    zp = make_zip(tmp_path / "b.zip", lambda s, l, w, i: f"img{i}.png")
    orders = audit.audit(zip_path=zp)["style"]["orders"]
    assert orders["name"]["lag_1"]["mean"] > 0.5
    assert orders["name"]["lag_1"]["mean"] > orders["name"]["random_order_lag_1"]["mean"] + 0.3
    assert orders["archive"]["lag_1"]["mean"] > 0.5


def test_tabular_forms_show_up_as_same_number_correlation(tmp_path):
    zp = make_zip(tmp_path / "c.zip", lambda s, l, w, i: f"form{w}_{i}.png", per_writer=1)
    r = audit.audit(zip_path=zp)
    study = r["style"]["same_number_across_classes"]["number_0"]
    assert study["same_number"]["mean"] > 0.5
    assert study["same_number"]["mean"] > study["shuffled_numbers"]["mean"] + 0.3


def test_writer_subfolders_and_duplicates(tmp_path):
    zp = make_zip(tmp_path / "d.zip", lambda s, l, w, i: f"writer{w:02d}/{i}.png")
    r = audit.audit(zip_path=zp)
    assert r["structure"]["distinct_subfolder_paths"] == WRITERS
    # the glyphs repeat exactly across writers' samples and splits, so identical pixels are found
    assert r["duplicates"]["groups_of_identical_pixels"] > 0


def test_directory_input_and_json(tmp_path):
    zp = make_zip(tmp_path / "e.zip", lambda s, l, w, i: f"{i}.bmp")
    with zipfile.ZipFile(zp) as z:
        z.extractall(tmp_path / "raw")
    r = audit.to_json(audit.audit(dir_path=tmp_path / "raw", max_per_class=10))
    assert r["style"]["images"] == 2 * CLASSES * 10
    json.dumps(r, allow_nan=False)


def test_sizes_and_duplicate_breakdown(tmp_path):
    rng = np.random.default_rng(0)

    def png(h, w):
        buf = io.BytesIO()
        Image.fromarray((rng.random((h, w)) * 200).astype("uint8")).save(buf, "PNG")
        return buf.getvalue()

    twin_a, twin_b = png(24, 24), png(24, 24)
    zp = tmp_path / "f.zip"
    with zipfile.ZipFile(zp, "w") as z:
        for split in ("train", "test"):
            for label in range(3):
                for i in range(40):  # class 002 is stored at another fixed size
                    z.writestr(f"R/{split}_{label:03d}/{i + 1}.png", png(24, 12) if label == 2 else png(24, 24))
        z.writestr("R/test_000/41.png", twin_a)  # the same image under two labels
        z.writestr("R/train_001/41.png", twin_a)
        z.writestr("R/test_000/42.png", twin_b)  # the same image in test and train, same label
        z.writestr("R/train_000/42.png", twin_b)
    csv_path = tmp_path / "dups.csv"
    r = audit.audit(zip_path=zp, duplicates_csv=csv_path)
    assert r["sizes"]["most_common"] == "24x24"
    assert set(r["sizes"]["classes_with_other_sizes"]) == {"train_002", "test_002"}
    d = r["duplicates"]
    assert d["groups_of_identical_pixels"] == 2
    assert d["groups_with_different_labels"] == 1
    assert d["label_pairs_in_those_groups"] == [{"classes": "000/001", "groups": 1}]
    t = d["test_images_with_identical_train_image"]
    assert (t["total"], t["same_label"], t["other_label_only"]) == (2, 1, 1)
    assert len(csv_path.read_text().splitlines()) == 1 + 4
