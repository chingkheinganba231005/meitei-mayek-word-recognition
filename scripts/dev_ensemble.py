"""The first paper's ensemble as trained for its validation stage, in the released folder's
format, for checking the baseline (scripts/oracle_baseline.py) on the synthetic validation
set: the released networks were trained on TUMMHCD train including our validation part,
the development ones without it.

    python scripts/dev_ensemble.py --release work/weights \\
        --runs /content/drive/MyDrive/tummhcd98/runs --out /content/weights_dev/first_paper_dev

For every member of the released config.json (convnext_t, effv2_s, resnet50d_topo and their
_meta versions), the first project's runs/<member>/dev/final.pt is read ({"model": weights,
"cfg": settings, "history": ...}); its weights must have the same tensors (names and shapes)
as the released member's and must not be the released weights themselves. The settings
(architecture, input channels, size features) must agree too. The ensemble's other entries
(test-time views, member weights, normalisation constants) are the release's; the constants
are means and standard deviations over training images, which the validation part barely
moves.
"""

import argparse
import json
import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from mayek_htr.model import find_model_dir  # noqa: E402


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--release", required=True, help="the released model folder, or a folder above it")
    ap.add_argument("--runs", required=True, help="the first project's runs folder")
    ap.add_argument("--tag", default="dev", help="the stage trained without the validation part")
    ap.add_argument("--out", required=True, help="new model folder")
    args = ap.parse_args()

    release = find_model_dir(args.release)
    config = json.loads((release / "config.json").read_text(encoding="utf-8"))
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    report = []
    for m in config["members"]:
        path = Path(args.runs) / m["name"] / args.tag / "final.pt"
        if not path.exists():
            sys.exit(f"missing {path}: every member of the release needs its {args.tag} network")
        ck = torch.load(path, map_location="cpu", weights_only=False)
        state = ck["model"] if isinstance(ck, dict) and "model" in ck else ck
        released = torch.load(release / m["file"], map_location="cpu", weights_only=True)
        if set(state) != set(released) or any(state[k].shape != released[k].shape for k in state):
            sys.exit(f"{path} does not hold a {m['name']} network like the released one")
        cfg = (ck.get("cfg") or {}) if isinstance(ck, dict) else {}
        wrong = [k for k in ("arch", "channels") if k in cfg
                 and str(cfg[k]).split(".")[0] != str(m["cfg"].get(k)).split(".")[0]]
        wrong += ["meta"] if "meta" in cfg and bool(cfg["meta"]) != bool(m["cfg"].get("meta")) else []
        if wrong:
            sys.exit(f"{path}: settings {wrong} differ from the released member's {m['cfg']}")
        floats = [k for k in state if state[k].is_floating_point()]
        same = sum(torch.equal(state[k], released[k]) for k in floats)
        if same == len(floats):
            sys.exit(f"{path} holds the released weights themselves, not a network trained without validation")
        torch.save(state, out / m["file"])
        history = ck.get("history") if isinstance(ck, dict) else None
        last = history[-1] if isinstance(history, list) and history else None
        report.append({"member": m["name"], "from": f"{m['name']}/{args.tag}/final.pt",
                       "tensors_equal_to_released": f"{same} of {len(floats)}", "last_history_row": last})
        print(f"{m['name']}: {same} of {len(floats)} tensors equal to the released network; last row {last}")
    config["trained_on"] = ("TUMMHCD train without our validation part (the first project's "
                            f"runs/<member>/{args.tag}/final.pt)")
    config["members_source"] = report
    (out / "config.json").write_text(json.dumps(config, indent=2, default=str), encoding="utf-8")
    print(f"written to {out}")


if __name__ == "__main__":
    main()
