# Phase 0: literature and data audit

Started 24 September 2026. Status: literature search and corpus licence audit done (first
pass); the IIIT Hyderabad question is settled (section 2); the first Colab run of the data audit
is done (section 10), and a second run with more detail is pending (section 11).

**How this was done, and what that means for citing.** The search ran in a cloud session whose
network allowed a web search engine and GitHub, but not publisher sites, arXiv, Hugging Face,
IIIT Hyderabad or the Tezpur server. So every paper below was read only through its search-result
abstract or snippet. The column "Checked" says what was seen: **A** = abstract or snippet only,
**F** = read from the source itself (the FineWeb-2 and AI4Bharat repository files, and the
IIIT-Indic-HW-UC paper, whose author PDF the owner supplied),
**M** = from memory, not searched in this session. Nothing marked A or M may be cited before it
has been checked against the full text (CLAUDE.md rule).

## 1. Findings in brief

1. **No end-to-end recogniser for handwritten Meitei Mayek words or lines was found.** The
   nearest work segments handwritten pages into lines and words but recognises only isolated
   characters (Inunganbi et al., 2020), or corrects a character classifier's output on segmented
   words with zones and the orthographic rule (Hijam and Saharia, 2024; Hijam's thesis adds a
   character LSTM language model).
2. **The one risk to the novelty claim is resolved.** IIIT Hyderabad's camera-captured
   handwritten dataset, IIIT-Indic-HW-UC (Mondal and Jawahar), includes Manipuri, but in
   **Bengali script**: the script column of its Table 1 says so, and its Manipuri word samples
   (Fig. 3) are Bengali-script handwriting (F). 101 writers, 200K words, 75,531 distinct words;
   CRNN baseline at 90.99% CRR and 83.38% WRR. There is no Meitei Mayek in it, so the novelty
   statement holds (section 7).
3. Meitei Mayek text recognition exists only for **print** (NE-OCR, 2026 preprint; a
   printed-character benchmark from 2022) and for isolated **scene-text characters**. EMBiL (2023)
   detects Meitei Mayek scene text and identifies its language but does not read it. IIIT
   Hyderabad's printed OCR (Mozhi) covers Manipuri, script not yet checked; the same group writes
   Manipuri in Bengali script in IIIT-Indic-HW-UC.
4. **No handwriting generation model for Meitei Mayek was found.** Indic diffusion work covers
   isolated Bangla characters and an unpublished Devanagari repository.
5. **Native Unicode Meitei Mayek text is scarce.** Wikipedia and FineWeb-2 together hold about
   1.45 million running words and 78,000 distinct words, with much overlap (section 10); FineWeb-2
   has 2.69M words of Manipuri in Bengali script (F). The MADLAD-400 authors report that most
   Meitei Mayek text on the web uses non-Unicode fonts (A).
6. **TUMMHCD carries no writer information, and its images carry no size or position**
   (section 10). File names are a class number and a running index; neighbouring and
   same-numbered files are unrelated in style; all 85,124 images are 24 × 24 with the ink filling
   the frame. 469 test images (3.7%) have a pixel-identical train image.
7. **Typed text mixes two spellings of the i after ꯥ, ꯣ and ꯨ.** Web text uses ꯢ only there,
   and page by page it either writes ꯢ there almost always or almost never (section 10). Pages
   of the first kind follow the thesis rule for 95–97% of their i's and have 4–6 ꯢ per ꯏ, close
   to the thesis figures (94.7%, 3.6). So the thesis spelling is a living practice, and about
   82,000 words of native text in it exist; the rest must be normalised before use.

## 2. Handwritten Meitei Mayek beyond isolated characters

| Work | What it does | Data and result | Relation to this project | Checked |
|---|---|---|---|---|
| Inunganbi, Choudhary, Manglem, "Meitei Mayek handwritten dataset: compilation, segmentation, and character recognition", *The Visual Computer* (online Jan 2020), doi 10.1007/s00371-020-01799-4 | Line and word segmentation of full handwritten pages by projection histograms; CNN on isolated letters | MM dataset: 189 handwritten pages, 809 lines; segmentation 91.84% (lines), 88.96% (words). Mayek27: 4,900 letters, 99.02% | Real handwritten *pages* of Meitei Mayek exist. Is MM public? If so, a source of real words (needs transcription). No word recognition | A |
| Hijam and Saharia, "Zone and rule assisted recognition of Meitei-Mayek handwritten characters", *Evolutionary Intelligence* 17:2963–2980 (2024), doi 10.1007/s12065-024-00920-z | CNN first stage; second stage uses zones and orthographic rules for confusable pairs, on segmented words | 100 handwritten words (565 characters), 88.50% → 91.86% (figures from CLAUDE.md) | Baseline to re-implement (Phase 2) | A |
| Hijam, PhD thesis, Tezpur University (2024), http://agnee.tezu.ernet.in:8082/jspui/handle/1994/1707 | Includes CNN + character-level LSTM language model on word images | Private test data (to check) | Must be clearly surpassed: public benchmark, sequence model, synthetic data at scale, real writer-disjoint test set | not reached |
| Mondal and Jawahar (IIIT Hyderabad), "Unconstrained Camera Captured Indic Offline Handwritten Dataset" (IIIT-Indic-HW-UC), ICPR 2024 proceedings (LNCS; venue from the Springer listing, the author PDF names none), doi 10.1007/978-3-031-78495-8_21; data: https://cvit.iiit.ac.in/usodi/ucciohd.php | Paragraphs of at most 50 words (prompts from web text corpora) handwritten on A4 and photographed with the writers' phones; page-level annotation, word images released; CRNN baseline (Gongidi et al.: transformation network, ResNet, 2-layer BLSTM, CTC; 96 × 256 input) | 13 languages, 1,220 writers, 91K pages, 2.6M words. **Manipuri is in Bengali script**: 101 writers, 200K words, 75,531 distinct; CRR 90.99%, WRR 83.38% (trained and tested on its own split) | No Meitei Mayek, so no overlap with contribution (1). Context for the paper (the only large handwritten Manipuri word set is in Bengali script) and lessons for Phases 2 and 3 (below) | F |
| NLTM OCR project, IIIT Hyderabad, https://ilocr.iiit.ac.in/ | Handwriting recognisers and a web API for 13 languages incl. Manipuri; printed for 22 | The IIIT-Indic-HW-UC paper describes the handwriting API | The API is presented with the Bengali-script data above; no sign of a Meitei Mayek model | A (project page), F (paper) |

**What IIIT-Indic-HW-UC tells us (F).**
(i) Its split is 75/10/15% of the word images of each language. Each writer wrote 100 to 200
paragraphs and the same paragraph could be written by several writers; the paper does not say
that the split is writer-disjoint or text-disjoint. Our real test set should be both, and say
so: a concrete difference in protocol.
(ii) Its collection protocol (typed prompts copied by hand on A4, photographed with a phone,
page-level boxes and reading order) is a tested template for Phase 3. Phone capture would also
keep our set closer to real use than flatbed scans.
(iii) Its baseline, the CRNN with CTC of Gongidi et al. (iiit-indic-hw-words), is the standard
Indic HTR baseline. Re-implement it on our data as a Phase 2 baseline next to the
pretrained-backbone model.
(iv) The paper states no licence; check the dataset page before any use. Its Bengali-script
Manipuri words could at most serve as same-language pretraining for the visual encoder, a
low-priority experiment.

Isolated-character work (for completeness, not competitors for word recognition): Mayek27
(above); Nongmeikapam, Kumar et al., ACM TALLIP 2019, doi 10.1145/3309497, dataset of more than
5,000 characters at 128×128 px on Mendeley Data, IEEE DataPort, Harvard Dataverse
(doi 10.7910/DVN/OMU2DV), Figshare, Dryad and Kaggle; Inunganbi et al., *Computational
Intelligence* 2021, doi 10.1111/coin.12392; "Bangla-Meitei Mayek scripts handwritten character
recognition using CNN", *Applied Intelligence*, doi 10.1007/s10489-020-01901-2; Hijam and
Saharia, TUMMHCD (2022) and multilevel fusion, *The Visual Computer* (2024),
doi 10.1007/s00371-023-02776-3. All A.

## 3. Meitei Mayek in print and scene text

| Work | What | Relevance | Checked |
|---|---|---|---|
| MWire Labs, "NE-OCR: Unified Optical Character Recognition for 10 Languages of Northeast India", Research Square preprint rs-9167777 (2026, date to check); model https://huggingface.co/MWirelabs/ne-ocr (CC BY 4.0) | ViTSTR-Base (86M) + CTC on 32×128 word/line crops, 1,056-character vocabulary, 12 language–script pairs incl. Meitei Mayek; trained on 1.34M images rendered from text corpora; mean character accuracy 94.99% | Printed, rendered text only. Shows word-level Meitei Mayek recognition from synthetic data works in print. Possible pretraining source or baseline to show the print/handwriting gap. Its Meitei Mayek text corpus may be reusable (check) | A |
| Mathew, Mondal, Jawahar, "Towards Deployable OCR Models for Indic Languages", ICPR 2024, doi 10.1007/978-3-031-78495-8_11, arXiv 2205.06740 | Mozhi: 1.2M printed word images, 13 languages incl. Manipuri; CTC models | Printed; script of Manipuri to check (the same group's handwriting set writes Manipuri in Bengali script) | A |
| "A benchmark dataset for printed Meitei/Meetei script character recognition", *Data in Brief* 45:108585 (2022), doi 10.1016/j.dib.2022.108585; Mendeley Data rw4b2zdk95 | 824 printed pages with binarised images, text files and XML; 51,460 isolated characters; CC BY | Its page transcriptions are native Meitei Mayek text (section 8) | A |
| Naosekpam, Islam, Chourasia, Sahu, "EMBiL", CAIP 2023, doi 10.1007/978-3-031-44237-7_7 | English–Manipuri scene text: 720 images, more than 28,500 text instances; detection and language identification (YOLOv5-based) | Detection and language identification only; it does not read the text | A |
| "Meetei Mayek natural scene character recognition using CNN", Springer 2023, doi 10.1007/978-3-031-27609-5_33 | Scene characters | Character level | A |

## 4. Indic word- and page-level handwriting recognition

These set the methods and evaluation practice we follow; none includes Meitei Mayek
(IIIT-Indic-HW-UC's Manipuri is in Bengali script, section 2).

| Work | What | Checked |
|---|---|---|
| Gongidi and Jawahar, "iiit-indic-hw-words", ICDAR 2021, doi 10.1007/978-3-030-86337-1_30 | 872K handwritten words, 135 writers, 8 scripts (10 with IIIT-HW-Dev and IIIT-HW-Telugu); CRNN baselines, pretraining study | A |
| ICDAR 2023 Competition on Indic Handwriting Text Recognition, doi 10.1007/978-3-031-41679-8_25 | Word-level Indic HTR competition | A |
| ICDAR 2025 Indic Handwritten Document Recognition (IHDR), https://ilocr.iiit.ac.in/icdar_2025_Indic_HDR/ | 10 languages (no Manipuri), 10,000 documents, about 400,000 words, about 1,000 writers | A |
| Kasuba et al., "PLATTER", arXiv 2502.06172 (2025) | Page-level HTR for 10 Indic languages (detection + recognition), six HTR models compared, CHIPS dataset | A |
| Dey, Alaei, Roy, "Handwritten Text Recognition for Low Resource Languages" (BharatOCR), arXiv 2512.01348 (2025) | ViT encoder, Transformer decoder, pretrained LM for refinement; paragraph-level Hindi and Urdu | A |
| GraDeT-HTR, EMNLP 2025 demos (2025.emnlp-demos.52), arXiv 2509.18081 | Bengali HTR with grapheme tokeniser and decoder-only Transformer; synthetic pretraining then real fine-tuning | A |
| DohaScript, arXiv 2602.18089 (2026) | Multi-writer continuous handwritten Hindi | A |
| "An Aid to Assamese Language Processing by Constructing an Offline Assamese Handwritten Dataset", ICON 2024 (2024.icon-1.10) | 410 pages, 300 writers aged 10–76. A neighbouring script at the scale of our Phase 3; useful for collection design | A |
| "HMM-based Indic handwritten word recognition using zone segmentation", arXiv 1708.00227 (authors and journal version to check) | Zone-wise word recognition for Bangla and Devanagari; the closest precedent for our zone-aware view | A |
| "End-to-end OCR for Bengali handwritten words", arXiv 2105.04020 | CNN + RNN + CTC on Bengali words | A |

## 5. Synthetic words and handwriting generation

**Composing words from character images.** Early handwriting synthesis concatenated character
glyphs: sample glyphs, deform them, align them on a baseline, join where needed. The review in
arXiv 2412.15853 ("Semi-supervised adaptation of diffusion models for handwritten text
generation") describes this line of work (A); the primary references (Arabic concatenation work
among them) still have to be collected. Roy, Mohta and Chaudhuri, "Synthetic data generation for
Indic handwritten text recognition", arXiv 1804.06254 (2018), deform digital text for Devanagari
and Bangla words (A). "Self-training of handwritten word recognition for synthetic-to-real
adaptation", arXiv 2206.03149 (A), is the reference point for closing the
synthetic-to-real gap. **No composition of Meitei Mayek words from isolated characters was
found**, and no work that places vowel signs and lonsum finals by zone.

**Generative models (Latin script mostly).** GAN and Transformer era: GANwriting (Kang et al.,
ECCV 2020), ScrabbleGAN (Fogel et al., CVPR 2020), Handwriting Transformers (Bhunia et al.,
ICCV 2021), VATr (Pippi et al., CVPR 2023) (all M). Diffusion: WordStylist (ICDAR 2023,
arXiv 2303.16576), DiffusionPen (ECCV 2024, arXiv 2409.06065; 5 style samples), One-DM
(ECCV 2024, arXiv 2409.04004; one style sample), DiffBrush (Dai et al., ICCV 2025,
arXiv 2508.03256; text lines), WriteViT (arXiv 2505.13235) (A). Evaluation metric to consider:
HWD (Pippi et al., BMVC 2023) (M).

**Indic scripts.** Okkhor-Diffusion, class-guided DDPM for isolated Bangla characters (IEEE
journal, 2024, document 10445466) (A); confidence-guided diffusion augmentation for Bangla
compound characters, arXiv 2605.10916 (2026) (A); a DiffBrush-style latent diffusion model for
Devanagari in a writer's style, GitHub `keysun8/Diffusion_HW_Hindi`, unpublished (A); DCGAN for
Devanagari handwriting (Springer chapter, doi 10.1007/978-981-95-0629-3_7) (A); BengaliDiff,
few-shot Bengali *font* generation (Springer chapter, doi 10.1007/978-3-032-09371-4_7) (A). One search summary said ScrabbleGAN has
been used to synthesise words for 10 Indic languages; source not identified (to check).
**Nothing for Meitei Mayek.**

## 6. Search gaps

Not yet searched: Google Scholar and Shodhganga for Indian theses on Meitei Mayek word or text
recognition since 2020 (Manipur University, NIT Manipur, IIIT Manipur); Bhashini's OCR services
(IIIT Hyderabad's Manipuri handwriting data is in Bengali script; is there any model for
Meitei Mayek?); the Hijam thesis itself; Indian conference
proceedings (NCVPRIPG, ICVGIP) for Meitei Mayek word recognition. Worth a second pass from a
machine with open internet.

## 7. Novelty statement (draft)

> To our knowledge, no published work recognises handwritten Meitei Mayek words or lines end to
> end. Earlier work classifies isolated characters, segments handwritten pages into lines and
> words without recognising them (Inunganbi et al., 2020), or corrects a character classifier's
> output on segmented words with zone information and the orthographic rule (Hijam and Saharia,
> 2024, on 100 words; Hijam's thesis adds a character LSTM language model). The only large
> handwritten Manipuri word dataset, IIIT-Indic-HW-UC (Mondal and Jawahar, 2024), is in Bengali
> script. Meitei Mayek text recognition exists only for print (scene-text work classifies
> isolated characters), and no handwriting generation model exists for the script. We contribute
> (1) the first segmentation-free handwritten Meitei Mayek word recogniser, built on pretrained
> visual encoders with a CTC or attention head and a character language model; (2) zone-aware
> synthetic word images composed from TUMMHCD characters at scale; (3) a public, consented,
> writer-disjoint set of real handwritten words from 50 or more writers, with a fixed protocol
> (CER, WER, ꯢ/ꯏ accuracy); (4) a controlled measurement of how much word context resolves ꯢ
> versus ꯏ, which isolated-character models cannot separate (68.2% against a 67.1% majority
> baseline); and later (5) the first diffusion model for Meitei Mayek handwriting, judged by
> whether it improves recognition of real handwriting.

**Distinct from Hijam's thesis:** public benchmark and protocol instead of a private word set;
segmentation-free sequence model instead of per-character CNN plus LSTM correction; synthetic
words at scale; writer-disjoint real test set; the ꯢ/ꯏ question measured directly.

**IIIT Hyderabad check: done (24 September 2026).** Its Manipuri handwriting is in Bengali
script (IIIT-Indic-HW-UC, Table 1 and Fig. 3), so the statement holds as written.

## 8. Text corpora

Size figures come from the sources' own documentation unless marked; the notebook recomputes
native Meitei Mayek word counts for everything it can download (`results/corpus_stats.json`).

| Source | Script | Size as documented | Licence | Access | Checked |
|---|---|---|---|---|---|
| Meitei Wikipedia (mni.wikipedia.org), dumps at dumps.wikimedia.org/mniwiki | Meitei Mayek | 10,222 articles (July 2022 figure) | CC BY-SA 4.0 | open; notebook downloads it | A |
| FineWeb-2 `mni_Mtei` (HuggingFaceFW/fineweb-2) | Meitei Mayek | 61,256 words, 169 documents (train) + 233 words (test); `wiki_ratio` 0.556 | ODC-By 1.0 (plus Common Crawl terms) | open; notebook downloads it | F (`fineweb2-language-distribution.csv`, commit d0defb2) |
| FineWeb-2 `mni_Mtei_removed` | Meitei Mayek | 3,718 documents, 7.3 MB, dropped by FineWeb-2's filters | ODC-By 1.0 | open; notebook downloads it; needs our own cleaning | F |
| FineWeb-2 `mni_Beng`, `mni_Latn` | Bengali, Latin | 2,688,747 and 1,700,950 words | ODC-By 1.0 | open | F |
| FLORES+ `mni_Mtei` (openlanguagedata/flores_plus) | Meitei Mayek | dev split only (FLORES dev is about 1,000 sentences) | CC BY-SA 4.0 | gated on Hugging Face (accept terms) | A |
| IN22-Gen / IN22-Conv (AI4Bharat) | Manipuri, script to check (IndicTrans2 supports both `mni_Beng` and `mni_Mtei`) | 1,024 / 1,503 sentences | CC BY 4.0 | open | F (IndicTrans2 README) |
| BPCC (AI4Bharat) | Manipuri, script and size to check | 230M pairs over 22 languages | BPCC-H-Wiki and -Daily: CC BY 4.0; mined part: CC0 packaging, source terms apply | open | F (licence), size not checked |
| Leipzig Corpora Collection; W2C (ÚFAL) | to check | to check | to check | open | lead only: IIIT-Indic-HW-UC built its prompts from these two collections |
| Sangraha (AI4Bharat) | Manipuri, script and size to check (paper Table 1, arXiv 2403.06350) | 251B tokens over 22 languages | CC BY 4.0 | open | A |
| MADLAD-400 `mni_Mtei` | Meitei Mayek | small; authors note most Meitei Mayek web text is in non-Unicode fonts | ODC-By (to check) | open | A |
| ILCI-II Hindi–Manipuri corpus (TDIL-DC, JNU) | Meitei Mayek | about 22,000 sentences, agriculture and entertainment, POS-tagged | TDIL-DC terms (registration; research use, redistribution to check) | registration | A |
| Printed Meitei/Meetei dataset text files (*Data in Brief* 2022) | Meitei Mayek | 824 pages; word count to compute | CC BY (to confirm version) | Mendeley Data | A |
| NE-OCR benchmark sets (MWire Labs) | Meitei Mayek among others | word/line crops up to 32 characters, 2,000 test samples per language–script pair | CC BY 4.0 (model card) | Hugging Face | A |
| MeiteiRoBERTa corpus (Nyalang, SIGTYP 2026, 2026.sigtyp-main.5) | Bengali | 76M words | "datasets released"; licence to check | to check | A |
| Transliteration resources: BMSC gold pairs (ACM TALLIP 2026, doi 10.1145/3806198); Aksharantar Manipuri (106k pairs, script to check); rule-based transliterators (Singh, 2012, W12-5016) | Bengali ↔ Meitei Mayek | 35,000 word pairs (BMSC) | to check | to check | A |
| Newspapers: Poknapham (online since 2008), Naharolgi Thoudang (e-paper), Hueiyen Lanpao, Sanaleibak, Marup | mixed; e-papers often images or non-Unicode fonts | — | copyright; needs written permission | — | A |
| Hugging Face `DayanandaThokchom/english-TO-meitei-mayek` | Meitei Mayek | unknown | unknown | — | A |

**Recommendations.**

1. *Evaluate on native text only.* Train and evaluate the language model, and measure ꯢ/ꯏ
   statistics, on native Meitei Mayek: Wikipedia, FineWeb-2, FLORES+, IN22 (if Meitei script),
   the printed dataset's transcriptions, and ILCI-II if TDIL's terms allow it. Deduplicate
   across sources (FineWeb-2 contains Wikipedia pages).
2. *Transliterated text is training data only, and flagged.* Bengali-script Manipuri is about 44
   times larger in FineWeb-2, but a transliterator decides ꯢ versus ꯏ by its own rules, so transliterated text
   carries the rule in by construction. Never use it to measure the rule or to test the language
   model on this pair; report results with and without it.
3. *Word list for synthetic words* from native text, weighted by frequency, with a held-out set
   of words that never appear in synthetic training data, so the test also measures unseen
   words.
4. *Prompts for the real test set (Phase 3).* Take sentences from a CC BY source (IN22, if its
   Manipuri is Meitei script) so the released set can itself be CC BY; CC BY-SA prompts (FLORES+,
   Wikipedia) would force share-alike. Add designed words that stress ꯢ/ꯏ and the other three
   confusable pairs.
5. *Replicate the rule figure.* `corpus_stats.py` reports how often the rule holds on each
   corpus, over running and distinct words. The thesis figure (94.7% on about 26,000 expert-checked
   words) and the frequency ratio (ꯢ about 3.6 times ꯏ) can be checked on open data this way.

*Revised after the first run (section 10):* native typed text is not reliable for the ꯢ/ꯏ pair,
so recommendation 1 no longer covers ꯢ/ꯏ statistics; those need text with checked spelling.

## 9. TUMMHCD writer information

**Outcome of the first run: no writer information (section 10).**

**What was known before.** 85,124 images, 55 classes, 72,330 train and 12,794 test (the first paper). About
500 writers, collected in two phases: unconstrained writing (answer sheets, classroom notes) and
tabular forms (dataset paper abstract, A). The archive holds `TUMMHCDtrain/train_NNN` and
`TUMMHCDtest/test_NNN` folders (`mayek/split.py` in the first project). Images are small scans,
mostly 24 × 24 px with strokes two to three pixels wide (`mayek/preprocess.py`): composed words
will be low resolution, and composition should work at that scale and upsample afterwards.

**What is not known.** Whether writer identity survives anywhere in the release, and whether the
official train/test split is writer-disjoint.

**What the audit measures** (`scripts/audit_tummhcd.py`, run from the notebook):

| Output | Reads as a writer signal when |
|---|---|
| `structure.images_in_subfolders_below_class_folder` | above zero: folders below the class level may be writers |
| `names.numbers[*]` | a writer ID has at most a few hundred values (about 500 writers), may repeat within a class, and recurs across classes; a running index fills its range per class (`median_fill_of_range_per_class` near 1), never repeats within a class, and its maximum is close to the class size |
| `other_files` | any list, spreadsheet or read-me that maps files to writers |
| `style.orders.{name,archive,time}.lag_k.mean` | clearly above the random-order value (which sits within about ±0.01 of zero at this size) at lag 1 and falling with the lag: files were saved writer by writer. The lag where it reaches the random level is roughly how many consecutive files one writer contributed |
| `style.same_number_across_classes` | above the shuffled value: the same number in different classes is the same writer (one tabular form per writer) |
| `duplicates.groups_spanning_train_and_test` | above zero: identical images in train and test (leakage; also relevant to the first paper) |

The style features (scan height and width, ink fraction, paper and ink grey levels, stroke width,
the four margins) are z-scored within each class, so character shape does not count as style.
On independent images the correlations sit at zero; on synthetic data with writer blocks or
tabular forms they are clearly positive (`tests/test_audit_tummhcd.py`).

**What each outcome means for Phase 1.**

- *Explicit writer IDs* (folders, names, a side file): compose writer-consistent words directly;
  split writers into train/validation/test for synthetic data.
- *Hidden grouping only:* cut each class's file sequence into pseudo-writer blocks where the
  style changes; link blocks across classes only if the same-number test is positive. Writer
  consistency is then approximate, and the paper must say so.
- *Nothing:* compose words from style-matched characters (nearest neighbours in size, stroke
  width and ink level) and normalise scale and stroke width. Writer-disjoint evaluation then comes
  from the Phase 3 real test set only, which is writer-disjoint from TUMMHCD by construction.
  Also ask Hijam and Saharia whether writer IDs exist for the form-based part.

## 10. Results of the Colab runs (24 September 2026)

Files: `results/tummhcd_audit.json` (all 85,124 images; archive sha256 7637cb9e…) and
`results/tummhcd_audit_duplicates.csv` from the second run, which repeats the first and adds image
sizes per class, the labels of duplicate groups and ꯢ/ꯏ counts by preceding character;
`results/corpus_stats.json` from the third run, which adds the count by document. Every number
below comes from these files.

### TUMMHCD: no writer information

| Where we looked | What we found |
|---|---|
| Folders and side files | no folder below the class folders, no other files; all 85,124 files are `.tif` |
| File names | one template, `mmhc<k>_<n>`. k runs 1–55 and is the class number plus one (each value sits in one class, in its train and test folders); n runs from 1 to the class size, fills that range and never repeats within a class: a running index, not a writer |
| Same n in different classes | style correlation 0.002 over 83,110 pairs, against 0.000 shuffled: the n-th files of two classes are not linked, so no tabular-form grouping survives |
| Neighbours in name order | 0.012 at lag 1 and 0.003 at lag 10, against −0.003 in random order: files were not saved writer by writer |
| Neighbours in file-time order | 0.064 at lag 1, falling slowly (0.025 at lag 100), carried by paper grey level (0.21) and ink grey level (0.11) more than by stroke width (0.06): scan batches, not writers. 682 distinct file times from 23 July to 19 September 2018, a median of 12 per class: each class was written out in about a dozen batches |

**The images carry no size or position.** All 85,124 images are 24 × 24 px (one size in the
whole archive). Each margin between the ink and the frame is at most one pixel in at least 95% of
the images, and zero at the top and left. So the characters were cropped to their ink and scaled
to fill the frame: their original size, aspect ratio and place in the word are gone.

**Duplicates.** 1,741 groups of pixel-identical images hold 3,512 images. Almost all are pairs
(1,712 pairs, 28 triples, one group of four), so they are duplicated files, not look-alike glyphs
(those would form large groups). They cluster in five classes, where about a third of all images
sit in a pair: ꯩ 049 (33.7%), ꯥ 045 (32.7%), ꯤ 048 (32.1%), ꯭ 054 (32.0%) and ꯧ 051 (31.3%);
then ꯖ 032 (12.9%) and ꯀ 010 (9.9%).

- 469 test images (3.7% of 12,794) have a pixel-identical train image: 456 with the same label,
  13 only with another label.
- 24 groups carry conflicting labels: ꯲/꯳ 001/002 (16), ꯲/꯹ 001/008 (4), and one each for
  ꯸/꯹, ꯆ/ꯇ, ꯗ/ꯘ and ꯙ/ꯚ. None is ꯢ/ꯏ (044/025), so label noise does not explain why that
  pair cannot be separated.
- The first run's `groups_spanning_classes` (479) counted a class's train and test folders as two
  classes; it was not a count of label conflicts. The fixed count is the 24 above.

**What this means.**

- Phase 1 cannot take a word's characters from one writer, and synthetic data cannot be split by
  writer. Compose words from characters matched in style (stroke width, ink and paper grey level;
  file-time batches as a weak session grouping). Keep TUMMHCD's own train/test split for the
  characters of synthetic training and test words, leaving out the test images that have a train
  twin, and take writer-disjoint evaluation from the Phase 3 real set only.
- Zone-aware placement needs a size and position model per class, because the images have none.
  One handle is the stroke width in the 24-px frame: for the same pen, a small glyph scaled up to
  fill the frame gets thicker strokes, so the median stroke width per class estimates each
  class's relative size. Check this in Phase 1.
- For the first paper (under review), two checks.
  - *Size features.* Image height and width are the same for every image and the ink box nearly
    always fills the frame, so of the five size numbers mostly the ink fraction can vary. Its
    size probe still separates 046/009 (86.5% against a 55.8% majority) and 011/047 (84.1%
    against 50.7%), so the signal is real, but it reaches the model through stroke thickness:
    a small glyph scaled up to 24 px gets thicker strokes and more ink. If the paper says the
    features see where a character sits in the word, or its size directly, correct that at
    revision.
  - *Duplicates.* 456 test images (3.6%) have an identical train image with the same label, and
    13 more one with another label. Score the final system without these 469
    (`results/tummhcd_audit_duplicates.csv` lists them); if none of the 241 errors is among them,
    accuracy would be 98.04% instead of 98.12%. The published results use the same test set, so
    the comparison stands; reporting both numbers pre-empts a reviewer. In the first project's
    notebook, with `pred` the final system's predicted classes in the row order of its
    `test.csv`:

    ```python
    import pandas as pd
    dup = pd.read_csv("tummhcd_audit_duplicates.csv")
    with_train = set(dup.loc[dup.split == "train", "group"])
    twins = set(dup[(dup.split == "test") & dup.group.isin(with_train)].path.str.split("TUMMHCD-TEST-TRAIN/").str[-1])
    test = pd.read_csv("data/splits/test.csv")
    keep = ~test.path.str.replace("\\", "/").str.split("TUMMHCD-TEST-TRAIN/").str[-1].isin(twins)
    print(keep.sum(), (pred == test.label).mean(), (pred[keep] == test.label[keep]).mean())
    ```

### Corpora: sizes

| Source | Running words | Distinct words | Non-space characters in the Meitei block |
|---|---:|---:|---:|
| Meitei Wikipedia dump | 1,036,283 | 72,437 | 19.5% (markup, English templates) |
| FineWeb-2 `mni_Mtei` | 52,395 | 11,141 | 91.1% |
| FineWeb-2 `mni_Mtei_removed` | 365,832 | 33,425 | 70.7% |
| All three (they overlap) | 1,454,510 | 77,580 | 25.0% |

FLORES+ did not download (gated: it needs `HF_TOKEN` after accepting its terms). Words here are
runs of Meitei Mayek letters, which is why FineWeb-2 counts fewer than its own 61,256. The removed
part seems to hold much Wikipedia interface text (ꯋꯤꯀꯤꯄꯦꯗꯤꯌꯥ, "Wikipedia", is among its 15 most
frequent words). Deduplicate before any use.

### ꯢ and ꯏ in typed text

| | Wikipedia | FineWeb-2 | FineWeb-2 removed |
|---|---:|---:|---:|
| ꯢ, all occurrences | 30,756 | 517 | 9,719 |
| ꯏ, all occurrences | 108,124 | 6,120 | 48,515 |
| ꯢ outside a vowel context | 0.42% | 1.9% | 0.7% |
| ꯏ among the i's after a vowel | 71.2% | 90.4% | 80.2% |
| i's after a vowel per other i | 3.29 | 3.98 | 5.15 |
| rule holds (running words) | 45.3% | 27.6% | 32.7% |

"After a vowel" means after a consonant letter (with its inherent a), a vowel letter or a vowel
sign; everything else is word start, a lonsum final, nung or apun.

Typed text uses ꯢ almost only after a vowel, as the rule says, but writes most i's after a vowel
as ꯏ. So the rule holds for only 28–45% of occurrences, and in Wikipedia ꯏ is 3.5 times as
frequent as ꯢ, the reverse of the thesis figure (ꯢ about 3.6 times ꯏ). The positions agree with
the thesis: Wikipedia has 3.3 i's after a vowel for every other i, close to its 3.6. The simplest
reading is that typists write ꯏ where the thesis's corpus (and presumably TUMMHCD's labels) have
ꯢ: a difference of spelling practice, not of language. The second run counts both letters by the
preceding character, which checks this context by context, including after consonant letters,
where our reading of "after a vowel" (the inherent a) could differ from the thesis's.

**By preceding character (second run).** Share of ꯢ among the i's in each context:

| Before the i | Wikipedia | FineWeb-2 | FineWeb-2 removed |
|---|---:|---:|---:|
| ꯥ | 31.8% (60,173 i's) | 10.5% (2,985) | 21.4% (32,253) |
| ꯣ | 26.6% (41,446) | 9.6% (1,877) | 17.3% (14,991) |
| ꯨ | 34.4% (1,092) | 20.5% (44) | 24.4% (484) |
| ꯦ, ꯧ, ꯤ or ꯩ | 1.4% (1,641) | 0.9% (212) | 6.3% (410) |
| a consonant letter (inherent a) | 0.6% (1,399) | 0.0% (121) | 3.5% (113) |
| a vowel letter | 1.9% (722) | 4.5% (66) | 3.5% (520) |
| word start | 0.2% (28,804) | 0.6% (1,152) | 0.2% (8,914) |
| a lonsum final, nung or apun | 1.8% (3,593) | 1.7% (180) | 10.2% (548) |

Of all ꯢ, 99.4% (Wikipedia), 97.1% and 98.8% follow ꯥ, ꯣ or ꯨ. So in typed practice ꯢ belongs
to the i after ꯥ, ꯣ or ꯨ (the diphthongs āi, oi, ui) and nowhere else, not even after a
consonant's inherent a; and after ꯥ, ꯣ or ꯨ most typists still write ꯏ. Our coding of the rule
counts the inherent a as a vowel, which typed text never follows with ꯢ; whether the thesis means
it must be checked in the thesis itself.

**What this means.**

- Typed corpora cannot teach or test the ꯢ/ꯏ distinction as they stand: a language model trained
  on them would prefer ꯏ after vowels, against the thesis orthography.
- The project needs a transcription convention, fixed and written down before Phase 1, and it has
  to say which vowels count. Standard orthography as in the thesis is the natural choice if
  TUMMHCD's labels follow it: check with a language expert, and check what the TUMMHCD paper says
  about labelling 044 and 025.
**By document (third run).** Share of ꯢ among the i's after ꯥ, ꯣ or ꯨ, for documents with at
least five such i's:

| Share of ꯢ | 0–0.1 | 0.1–0.2 | 0.2–0.3 | 0.3–0.4 | 0.4–0.5 | 0.5–0.6 | 0.6–0.7 | 0.7–0.8 | 0.8–0.9 | 0.9–1 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| Wikipedia pages | 1,071 | 392 | 152 | 90 | 71 | 103 | 181 | 399 | 1,459 | 605 |
| FineWeb-2 documents | 120 | 2 | 0 | 1 | 1 | 2 | 3 | 3 | 5 | 16 |
| FineWeb-2 removed | 1,109 | 211 | 103 | 51 | 46 | 38 | 93 | 54 | 102 | 286 |

Two spelling practices, not one noisy one: in Wikipedia 1,615 pages write ꯢ for fewer than 30%
of these i's and 2,463 for 70% or more, with only 445 in between. The web outside Wikipedia leans
to ꯏ (FineWeb-2: 122 against 24).

| Documents that write ꯢ for 90% or more | Wikipedia | FineWeb-2 | FineWeb-2 removed |
|---|---:|---:|---:|
| documents | 605 | 16 | 286 |
| running words | 55,589 | 2,404 | 23,705 |
| rule holds, running words | 95.1% (11,982 i's) | 96.9% (318) | 96.0% (4,407) |
| rule holds, distinct words | 92.7% (1,293) | 94.5% (165) | 92.0% (807) |
| ꯢ per ꯏ, running words | 5.9 | 4.1 | 4.8 |
| ꯢ per ꯏ, distinct words | 2.4 | 3.2 | 2.7 |

These pages match the thesis (94.7% on about 26,000 checked words; ꯢ about 3.6 times ꯏ) closely.
Part of the match is built in, because the pages were chosen for writing ꯢ after ꯥ, ꯣ and ꯨ; but
the other half of the rule holds on its own there: ꯏ at word start 1,002 times against 14 ꯢ, after
a lonsum final 164 against 9 (Wikipedia). No i follows a bare consonant letter anywhere in these
pages. In the Wikipedia pages the rule misses 587 of 11,982 i's: ꯏ after ꯥ (456, 6% of the i's
there), ꯏ after ꯣ or ꯨ (33), ꯏ after ꯦ, ꯧ, ꯤ or a vowel letter (75; the rule as coded predicts ꯢ
there, typed practice never writes it), and ꯢ at word start or after a final (23). The first two
groups and the last are the candidates for genuine exceptions, or typing slips.

**Exception candidates (from the word list of the 605 always-ꯢ Wikipedia pages).**
`scripts/i_exceptions.py` checks every i in the list against the recommended convention (ꯢ after
ꯥ, ꯣ or ꯨ, ꯏ elsewhere) and looks up each disagreeing word's twin, the same word with the other i
(`results/i_exception_summary.json`; review sheet `results/i_exception_candidates.csv`, 67 words
from the Meitei Wikipedia, CC BY-SA 4.0). The convention holds for 95.7% of the 11,982 i's and
for 94.8% of distinct words (thesis: 94.7%). Of the 518 misses, 398 are one word, ꯃꯆꯥꯈꯥꯏꯕ,
among the ten most frequent words of the whole dump (9,514 times) and probably inserted by a
template; without it the convention holds for 99.0%. The other 120:

- 74 are less frequent variants of words the same pages usually spell by the convention
  (ꯂꯥꯏꯅꯤꯡ 15 against ꯂꯥꯢꯅꯤꯡ 417; ꯑꯣꯏꯕ 1 against ꯑꯣꯢꯕ 967): slips or other authors;
- 36 are words seen once or twice in this spelling only: ꯢ at word start in forms that look like
  faulty conversion from legacy fonts (ꯢꯝ꯭ꯐꯥꯂ for ꯏꯝꯐꯥꯜ), ꯢ for the verb ending after a final
  consonant (ꯀꯣꯛꯢ; the same pages write ꯈꯨꯠꯏ 76 times), and ꯏ after ꯥ or ꯨ in rare words and
  loans (ꯄꯤꯑꯥꯏꯑꯦꯟ, ꯑꯥꯏꯑꯦꯁꯇꯤ);
- 10 have the other spelling about as often, among them the i after ꯧ (ꯀꯧꯢ 3, ꯀꯧꯏ 3), which the
  convention leaves open.

No word stands out as a genuine lexical exception. Questions for the language expert:
(1) ꯃꯆꯥꯈꯥꯏꯕ, ꯢ as the convention says? (2) the i after ꯧ and ꯦ; (3) the letter I in loans and
acronyms (ꯑꯥꯏ or ꯑꯥꯢ); (4) any exceptions the thesis itself lists.

The main ꯢ mode in Wikipedia lies at 80–90%, below the 90% cut: either the thesis spelling keeps ꯏ
after ꯥ in some words, or pages have several authors. The natural cut between the two practices
is the valley at 30–70%.

- The thesis spelling is a living practice, and about 82,000 running words of native text in it
  exist at the strict 90% cut (sources overlap), more at a 70% cut. That is the best text we have
  for training the language model on this pair and for testing it.
- Recommended convention (owner to confirm with a language expert): the thesis spelling, with ꯢ
  for the i after ꯥ, ꯣ or ꯨ and ꯏ everywhere else, plus a short list of exceptions taken from
  the ꯢ-writing pages and checked by the expert.
- Text preparation for Phase 1: the ꯢ-writing pages (70% or more) as they are; the ꯏ-writing pages
  normalised by the rule and the exception list, and flagged; ꯢ/ꯏ accuracy measured only on held-out
  ꯢ-writing pages and, later, on the real test set.
- Under that convention, normalise typed text before training (ꯏ after a vowel becomes ꯢ). This
  also erases the genuine exceptions (5.3% of the thesis's checked words), so what context adds
  beyond the rule can only be measured on text with checked spelling: the thesis's TDIL corpus,
  ILCI-II, the printed dataset's transcriptions, or our own test-set transcriptions made under the
  convention.

## 11. To do

1. Give the language expert `results/i_exception_candidates.csv` (67 words, a column for the
   verdict) and the four questions in section 10.
2. Confirm the ꯢ/ꯏ convention with a language expert (recommended: the thesis spelling, ꯢ for
   the i after ꯥ, ꯣ or ꯨ), and the exception list; find out how TUMMHCD labelled 044 and 025.
3. First paper: check the size-feature explanation, and score the final system without the 469
   test images that have a train twin.
4. Get text with checked spelling: ILCI-II (register on TDIL-DC), the printed dataset's text files
   (Mendeley Data); FLORES+ with `HF_TOKEN`. Check the Manipuri script and size in IN22, BPCC and
   Sangraha.
5. IIIT Hyderabad: the licence on the IIIT-Indic-HW-UC page, and the script of Manipuri in Mozhi.
6. Read the full texts of the entries marked A that we will cite, starting with Inunganbi et al.
   (2020): is the MM page dataset available? Read Hijam's thesis chapter on the CNN + LSTM word
   model (test set, size, results), and its exact wording of the ꯢ/ꯏ rule.
7. TUMMHCD paper: is the official test split meant to be writer-disjoint? 469 test images with an
   identical train image show that some images, and so some writers, are shared. Ask the authors
   about the duplicated vowel-sign files.

## Sources

Search results used in this report (A entries above link to the same pages):

- [Inunganbi et al. 2020, Springer](https://link.springer.com/article/10.1007/s00371-020-01799-4)
- [Hijam and Saharia 2024, Evolutionary Intelligence](https://link.springer.com/article/10.1007/s12065-024-00920-z)
- [TUMMHCD paper, Springer](https://link.springer.com/article/10.1007/s00371-020-02032-y)
- [Unconstrained Camera Captured Indic Offline Handwritten Dataset](https://link.springer.com/chapter/10.1007/978-3-031-78495-8_21) (author PDF supplied by the owner, read in full)
- [NLTM OCR, IIIT Hyderabad](https://ilocr.iiit.ac.in/)
- [NE-OCR preprint](https://www.researchsquare.com/article/rs-9167777/v1) and [model](https://huggingface.co/MWirelabs/ne-ocr)
- [Towards Deployable OCR Models for Indic Languages](https://arxiv.org/abs/2205.06740)
- [Printed Meitei/Meetei benchmark, Data in Brief](https://pmc.ncbi.nlm.nih.gov/articles/PMC9679442/)
- [EMBiL](https://link.springer.com/chapter/10.1007/978-3-031-44237-7_7)
- [iiit-indic-hw-words](https://link.springer.com/chapter/10.1007/978-3-030-86337-1_30)
- [ICDAR 2025 IHDR](https://ilocr.iiit.ac.in/icdar_2025_Indic_HDR/)
- [PLATTER](https://arxiv.org/abs/2502.06172)
- [BharatOCR](https://arxiv.org/abs/2512.01348)
- [GraDeT-HTR](https://aclanthology.org/2025.emnlp-demos.52/)
- [Assamese handwritten dataset, ICON 2024](https://aclanthology.org/2024.icon-1.10/)
- [Synthetic data generation for Indic HTR](https://arxiv.org/abs/1804.06254)
- [HMM-based Indic word recognition using zone segmentation](https://arxiv.org/abs/1708.00227)
- [Semi-supervised adaptation of diffusion models for HTG](https://arxiv.org/abs/2412.15853)
- [Self-training for synthetic-to-real adaptation](https://arxiv.org/abs/2206.03149)
- [WordStylist](https://arxiv.org/abs/2303.16576), [DiffusionPen](https://arxiv.org/abs/2409.06065), [One-DM](https://arxiv.org/abs/2409.04004), [DiffBrush](https://arxiv.org/abs/2508.03256)
- [Okkhor-Diffusion](https://ieeexplore.ieee.org/document/10445466/), [Bangla compound diffusion augmentation](https://arxiv.org/abs/2605.10916), [Diffusion_HW_Hindi](https://github.com/keysun8/Diffusion_HW_Hindi)
- [FineWeb-2 repository](https://github.com/huggingface/fineweb-2), [IndicTrans2 repository](https://github.com/AI4Bharat/IndicTrans2), [IndicLLMSuite repository](https://github.com/AI4Bharat/IndicLLMSuite)
- [FLORES+](https://huggingface.co/datasets/openlanguagedata/flores_plus), [MADLAD-400](https://arxiv.org/abs/2309.04662), [Meitei Wikipedia](https://simple.wikipedia.org/wiki/Meitei_Wikipedia)
- [ILCI-II Hindi–Manipuri corpus, TDIL-DC](https://tdil-dc.in/index.php?lang=en&option=com_download&task=showresourceDetails&toolid=1874)
- [MeiteiRoBERTa, SIGTYP 2026](https://aclanthology.org/2026.sigtyp-main.5/), [BMSC transliteration, TALLIP](https://doi.org/10.1145/3806198), [Aksharantar](https://aclanthology.org/2023.findings-emnlp.4/)
