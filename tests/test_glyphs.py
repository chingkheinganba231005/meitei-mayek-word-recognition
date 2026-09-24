import numpy as np
import pandas as pd
from PIL import Image

from mayek_words.charset import I_LETTER
from mayek_words.glyphs import GlyphStore, excluded_paths, ink_map, path_key


def stroke_image(x0=8, x1=12):
    im = np.full((24, 24), 250, np.uint8)
    im[3:21, x0:x1] = 40
    return im


def test_ink_map_keeps_polarity():
    alpha, _ = ink_map(np.zeros((24, 24), np.uint8))       # solid ink (apun stretched to the frame)
    assert alpha.min() == 1.0
    alpha, _ = ink_map(np.full((24, 24), 255, np.uint8))   # blank
    assert alpha.max() == 0.0
    im = np.full((24, 24), 60, np.uint8)
    im[:, :4] = 250                                         # mostly ink, a little paper
    alpha, feats = ink_map(im)
    assert alpha[:, 10].min() > 0.9 and alpha[:, 0].max() < 0.1
    alpha, feats = ink_map(stroke_image())
    assert alpha[10, 9] == 1.0 and alpha[10, 0] == 0.0 and feats[3] == 40.0


def test_font_store_and_pick():
    store = GlyphStore.from_font()
    assert len(store) == 55 and store.alpha.shape == (55, 24, 24)
    rng = np.random.default_rng(0)
    assert {int(store.labels[store.pick(I_LETTER, rng)]) for _ in range(50)} == {25, 44}


def test_style_matching_picks_the_nearest():
    images = [stroke_image(8, 8 + w) for w in (2, 3, 4, 5, 6, 7)]
    store = GlyphStore(images, [10] * 6)
    anchor = store.style[0]
    rng = np.random.default_rng(0)
    picks = {store.pick(chr(0xABC0), rng, anchor, k=2) for _ in range(40)}
    assert picks == {0, 1}                                 # the two thinnest, nearest to the thinnest


def test_from_split_leaves_out_duplicates(tmp_path):
    raw = tmp_path / "raw" / "TUMMHCD-TEST-TRAIN"
    rows = {"train": [], "test": []}
    for split, n in (("train", 3), ("test", 2)):
        d = raw / f"TUMMHCD{split}" / f"{split}_025"
        d.mkdir(parents=True)
        for k in range(n):
            Image.fromarray(stroke_image(6 + k, 10 + k)).save(d / f"mmhc26_{k}.tif")
            rows[split].append({"path": f"../raw/TUMMHCD-TEST-TRAIN/TUMMHCD{split}/{split}_025/mmhc26_{k}.tif",
                                "label": 25})
    splits = tmp_path / "splits"
    splits.mkdir()
    for split, r in rows.items():
        pd.DataFrame(r).to_csv(splits / f"{split}.csv", index=False)
    dup = tmp_path / "dups.csv"
    dup.write_text("group,pixel_md5,split,label,path\n"
                   "0,x,test,025,TUMMHCD-TEST-TRAIN/TUMMHCDtest/test_025/mmhc26_0.tif\n"
                   "0,x,train,025,TUMMHCD-TEST-TRAIN/TUMMHCDtrain/train_025/mmhc26_2.tif\n"
                   "1,y,train,002,TUMMHCD-TEST-TRAIN/TUMMHCDtrain/train_002/mmhc3_1.tif\n"
                   "1,y,train,009,TUMMHCD-TEST-TRAIN/TUMMHCDtrain/train_009/mmhc10_1.tif\n")
    assert excluded_paths(dup, "train") == {"TUMMHCDtrain/train_002/mmhc3_1.tif", "TUMMHCDtrain/train_009/mmhc10_1.tif"}
    train = GlyphStore.from_split(splits, "train", dup)
    test = GlyphStore.from_split(splits, "test", dup)
    assert len(train) == 3 and train.dropped == 0
    assert len(test) == 1 and test.dropped == 1 and test.keys == ["TUMMHCDtest/test_025/mmhc26_1.tif"]
    test.save(tmp_path / "t.npz")
    back = GlyphStore.load(tmp_path / "t.npz")
    assert back.keys == test.keys and np.array_equal(back.gray, test.gray)
    assert path_key("C:\\x\\TUMMHCDtest\\test_025\\a.tif") == "TUMMHCDtest/test_025/a.tif"
