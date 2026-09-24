# AI use log

What Claude Code (Anthropic) did in this project, for the journal statement on AI use. One entry
per working session. The author checks and takes responsibility for everything listed.

| Date | What Claude Code did | Kind |
|---|---|---|
| 2026-09-24 | Phase 0 literature search on word- and line-level handwritten Meitei Mayek recognition, Indic handwriting recognition, synthetic word images and handwriting generation. Web search only; no full texts were read, so every reference is marked for checking (`docs/phase0_audit.md`). | literature search |
| 2026-09-24 | Audit of Meitei Mayek text corpora and their licences; read the FineWeb-2 language statistics and the AI4Bharat IndicTrans2 and IndicLLMSuite documentation on GitHub. | data audit |
| 2026-09-24 | Wrote `scripts/audit_tummhcd.py` (writer information in TUMMHCD), `scripts/corpus_stats.py` (corpus sizes and ꯢ/ꯏ rule statistics), their tests, and `notebooks/phase0_data_audit.ipynb`. Tested on synthetic data only; not yet run on the real data. | code |
| 2026-09-24 | Drafted `docs/phase0_audit.md` and the Phase 0 section of `CLAUDE.md`, including the draft novelty statement. | text drafting |
| 2026-09-24 | Made the Phase 0 notebook's clone step survive the branch merge and stop on failed steps; ran the whole notebook in the cloud session with Colab stubbed out, on synthetic data. | code |
| 2026-09-24 | Read the IIIT-Indic-HW-UC paper (Mondal and Jawahar; author PDF supplied by the author of this project) and recorded that its Manipuri part is in Bengali script, with its size, baseline results and lessons for Phases 2 and 3; updated the novelty statement in `CLAUDE.md` and `docs/phase0_audit.md` accordingly. | literature reading, text drafting |
| 2026-09-24 | Interpreted the first Colab run of the Phase 0 audit (run by the author): recorded that TUMMHCD carries no writer information, that its images are scaled to fill 24 × 24 frames, its duplicates, the corpus sizes, and that typed text breaks the ꯢ/ꯏ rule; drafted the revised decisions in `CLAUDE.md` and `docs/phase0_audit.md`. Added per-class image sizes, duplicate-label and train/test-twin counts to `scripts/audit_tummhcd.py`, and ꯢ/ꯏ counts by preceding character to `scripts/corpus_stats.py`, with tests. | data analysis, code, text drafting |
