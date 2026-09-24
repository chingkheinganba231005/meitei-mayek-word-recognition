# Handwritten Meitei Mayek word recognition

Reading whole handwritten Meitei Mayek words instead of isolated characters, using a character
language model and the script's orthographic rules to fix errors that no isolated-character model
can fix (above all ꯢ, *i lonsum*, against ꯏ, *i*). A second stage will generate handwriting in
different writers' styles with a diffusion model. The project brief and every decision so far are
in [`CLAUDE.md`](CLAUDE.md).

This follows the character-level work in
[Handwritten-Meitei-Mayek-Recognition](https://github.com/chingkheinganba231005/Handwritten-Meitei-Mayek-Recognition)
(98.12% on TUMMHCD).

## Status

Phase 0, the literature and data audit: [`docs/phase0_audit.md`](docs/phase0_audit.md).

## Layout

```
scripts/
  audit_tummhcd.py   writer information in TUMMHCD: folders, file names, side files, hidden grouping
  corpus_stats.py    size of Meitei Mayek text corpora and how often the ꯢ / ꯏ rule holds
notebooks/
  phase0_data_audit.ipynb   runs both on Colab (TUMMHCD from Google Drive, corpora downloaded)
results/             JSON files written by the code; every reported number comes from here
docs/                Phase 0 report, AI use log
tests/               pytest
```

## Running

```bash
pip install -r requirements-dev.txt
pytest -q
python scripts/audit_tummhcd.py --zip TUMMHCD-TEST-TRAIN.zip          # -> results/tummhcd_audit.json
python scripts/corpus_stats.py wiki=mniwiki-latest-pages-articles.xml.bz2   # -> results/corpus_stats.json
```

TUMMHCD comes from its authors, <http://agnigarh.tezu.ernet.in/~sarat/resources.html>, under
their own terms.
