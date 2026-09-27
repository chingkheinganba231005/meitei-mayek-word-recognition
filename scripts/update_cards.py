"""Write the Hugging Face cards and the scores they quote again from the results files, and
optionally upload them. The trained network is not needed (scripts/build_demo.py builds
the whole model repository and Space).

    python scripts/update_cards.py --results results --out work/hf_cards            # write only
    python scripts/update_cards.py --results results --out work/hf_cards --upload   # and upload

Writes model/README.md and model/results.json (the model repository) and space/README.md
and space/about.txt (the Space: its card, and the paragraph under the demo page). With
--upload, the Space's config.json is downloaded, its "about" text replaced, and the four
files uploaded. The token is read from the HF_TOKEN environment variable (on Colab, a
secret), never from the command line.
"""

import argparse
import json
import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from mayek_htr.cards import HF_ID, about_text, cards, load_results, results_json, results_text  # noqa: E402


def write(results, out, hf_id=HF_ID, run=None):
    """The cards and results.json from the results files -> {name: path}."""
    run, val, test, real = load_results(results, run=run)
    text = results_text(test, real)
    model_card, space_card = cards(text, run, val.get("step") if val else None, hf_id)
    out = Path(out)
    (out / "model").mkdir(parents=True, exist_ok=True)
    (out / "space").mkdir(parents=True, exist_ok=True)
    files = {"model/README.md": model_card, "space/README.md": space_card, "space/about.txt": about_text(text),
             "model/results.json": json.dumps(results_json(run, val, test, real), indent=1, ensure_ascii=False)}
    for name, content in files.items():
        (out / name).write_text(content, encoding="utf-8")
    return {name: out / name for name in files}


def upload(paths, out, hf_id):
    from huggingface_hub import HfApi, hf_hub_download

    token = os.environ.get("HF_TOKEN")
    if not token:
        raise SystemExit("set HF_TOKEN (on Colab: a secret, read with google.colab.userdata) to upload")
    api = HfApi(token=token)
    config = json.loads(Path(hf_hub_download(hf_id, "config.json", repo_type="space", token=token))
                        .read_text(encoding="utf-8"))
    config["about"] = paths["space/about.txt"].read_text(encoding="utf-8")
    (Path(out) / "space" / "config.json").write_text(json.dumps(config, indent=1, ensure_ascii=False), encoding="utf-8")
    jobs = [("model/README.md", "README.md", "model"), ("model/results.json", "results.json", "model"),
            ("space/README.md", "README.md", "space"), ("space/config.json", "config.json", "space")]
    for local, remote, kind in jobs:
        api.upload_file(path_or_fileobj=str(Path(out) / local), path_in_repo=remote, repo_id=hf_id, repo_type=kind,
                        commit_message="Update the card and the quoted scores")
        print(f"uploaded {local} -> {kind} {hf_id}/{remote}")


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--results", required=True, help="folder with phase2_val_*, phase2_test_* and phase3_real_*.json")
    ap.add_argument("--out", required=True)
    ap.add_argument("--hf-id", default=HF_ID)
    ap.add_argument("--run", help="the demo's run (default: chosen on synthetic validation, as build_demo.py does)")
    ap.add_argument("--upload", action="store_true")
    args = ap.parse_args()
    paths = write(args.results, args.out, args.hf_id, args.run)
    print("written:", ", ".join(str(p) for p in paths.values()))
    if args.upload:
        upload(paths, args.out, args.hf_id)


if __name__ == "__main__":
    main()
