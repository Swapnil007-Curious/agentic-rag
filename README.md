# Agnostic Agentic-RAG: Hybrid Vector-Semantic Document Intelligence for Niche Low-Resource Corpora

This repository contains the **document ingestion pipeline** of the project: PDF text extraction
and cleaning, legal-structure parsing, cross-reference extraction, three chunking strategies
behind one interface, corpus statistics, and the evaluation-set schema with a validator and a
section browser. Retrieval, embedding indexes, LLM calls, agents and UI are later stages and are
not part of this code yet.

"Agnostic" means the pipeline is domain-independent: document structure comes from a pluggable
`StructureParser`. The case-study corpus uses `IndianStatuteParser`; any other PDF can be
ingested with `GenericParser` (font-size / numbering based headings) by adding it to
`configs/ingest.yaml` with `parser: generic`.

## Corpus and sources

India's Consumer Protection Act, 2019 and four of its Rules. File hashes, sizes, page counts and
source URLs are recorded in [`data/raw/SOURCES.csv`](data/raw/SOURCES.csv) (the PDFs themselves
are not redistributed; put them in `data/raw/`).

| doc_id | file | document |
|---|---|---|
| `cpa2019` | `cpa2019.pdf` | The Consumer Protection Act, 2019 (India Code copy) |
| `ecom2020` | `ecommerce_rules_2020.pdf` | The Consumer Protection (E-Commerce) Rules, 2020 |
| `commissions2020` | `commission_rules_2020.pdf` | The Consumer Protection (Consumer Disputes Redressal Commissions) Rules, 2020 |
| `jurisdiction2021` | `jurisdiction_rules_2021.pdf` | The Consumer Protection (Jurisdiction of the District Commission, the State Commission and the National Commission) Rules, 2021 |
| `mediation2020` | `mediation_rules_2020.pdf` | The Consumer Protection (Mediation) Rules, 2020 |

Notes on the files (all have a text layer; no OCR is needed):

* The four Rules are bilingual Gazette of India copies: the Hindi version comes first, then the
  English one. Hindi pages and lines are removed during cleaning.
* `commission_rules_2020.pdf` is a Gazette issue carrying **two** notifications: G.S.R. 448(E)
  (the Commissions Rules, ingested as `commissions2020`) and G.S.R. 449(E) (the Consumer
  Protection (General) Rules, 2020). The second one is configured as `general2020` with
  `enabled: false` so the case-study corpus stays at five documents; set `enabled: true` in
  `configs/ingest.yaml` to include it (its English preamble shares a mostly-Hindi page that is
  dropped, so that document starts at rule 1).

## Setup (Windows 11, PowerShell)

Python 3.11 is required. Everything runs on CPU.

```powershell
py -3.11 -m venv .venv
.venv\Scripts\Activate.ps1
python --version                      # must print Python 3.11.x
python -m pip install --upgrade pip
# CPU-only torch first, from the PyTorch CPU wheel index (no CUDA download):
pip install torch==2.15.0 --index-url https://download.pytorch.org/whl/cpu
pip install -r requirements-dev.txt   # runtime + dev pins (requirements.txt is included)
pip install -e .
```

On Linux the same commands work with `python3.11 -m venv .venv` and `source .venv/bin/activate`.
The first run downloads the `BAAI/bge-small-en-v1.5` tokenizer and model (about 130 MB) into the
HuggingFace cache.

## Commands

```powershell
# 1. extract, clean, parse structure and cross-references for the configured PDFs
python scripts/ingest.py                    # --write-sources also refreshes data/raw/SOURCES.csv

# 2. chunk (fixed_256 | fixed_512 | semantic | structural | all)
python scripts/chunk.py --chunker all

# 3. browse sections while writing evaluation questions
python scripts/browse.py list --doc cpa2019
python scripts/browse.py show cpa2019:s35
python scripts/browse.py show "cpa2019:s2(7)"
python scripts/browse.py search "unfair trade practice"
python scripts/browse.py defs

# 4. validate the evaluation set
python scripts/validate_eval.py
```

All scripts accept path overrides (`--config`, `--raw-dir`, `--processed-dir`, ...); run any of
them with `--help`.

Outputs (all under `data/processed/`, which is git-ignored):

| path | content |
|---|---|
| `clean/<doc_id>.txt` | cleaned text of one document; every character offset in the outputs points into this file |
| `clean/<doc_id>.pages.json` | `(char_start, char_end, page)` spans mapping offsets to PDF pages |
| `structure/nodes.jsonl` | one structural node per line (see below) |
| `structure/xrefs.jsonl` | cross-reference edges `source_node -> target_node` with `raw_text` |
| `structure/footnotes.jsonl` | footnotes (amendment notes) linked to their page and sections |
| `structure/structure_report.json` | per-document statistics, numbering checks, warnings, config snapshot, run info |
| `chunks/<chunker>.jsonl` | chunks of one chunker |
| `chunks/chunk_report.json` | per-chunker counts, token statistics, boundary statistics, wall time |

Every report embeds the configuration used and the run environment (git commit and dirty flag,
Python, OS, CPU model, RAM, package versions, UTC time).

## What cleaning does

* keeps PyMuPDF line positions, font sizes and weights, and merges text objects printed on the
  same baseline (Gazette rule numbers are separate text objects from their headings);
* drops running headers/footers (text that recurs in the page margins at a similar height on at
  least half of all pages, or of the odd/even pages), standalone page numbers, rotated
  watermarks, superscript footnote markers, and the ARRANGEMENT OF SECTIONS block;
* drops pages that are predominantly Devanagari and any line whose letters are more than 30%
  Devanagari; residual Devanagari characters in English lines are removed, and Hindi clause
  labels such as `(क)` inside English text become `(a)`;
* cuts each document to its body with configurable start/end markers (front matter, file-number
  and signature blocks, printer imprint, other notifications in the same Gazette issue);
* keeps footnotes as separate records; de-hyphenates words broken across lines using the
  document's own vocabulary (`sub-section` stays hyphenated), joins wrapped lines into paragraphs,
  normalises quotes and dashes, and preserves clause markers, `Explanation.—` and `Provided that`.

## Node ID convention

Node IDs are stable for a given configuration and are what evaluation questions reference.

| unit | pattern | example |
|---|---|---|
| chapter | `<doc>:ch<n>` (Roman numeral as integer) | `cpa2019:ch4` |
| section (Act) | `<doc>:s<n>` | `cpa2019:s35` |
| sub-section | `<section>(<n>)` | `cpa2019:s35(1)`, `cpa2019:s55(1A)` |
| clause | `<parent>(<label>)` | `cpa2019:s35(1)(a)`, `cpa2019:s101(2)(zg)` |
| definition clause | clause of the definitions section | `cpa2019:s2(7)` ("consumer"), `ecom2020:r3(1)(b)` |
| sub-clause | `<clause>(<roman>)` | `cpa2019:s2(7)(i)`, `cpa2019:s58(1)(a)(iv)` |
| rule (Rules) | `<doc>:r<n>` | `ecom2020:r4` |
| sub-rule | `<rule>(<n>)` | `ecom2020:r4(3)` |
| proviso | `<parent>-prov<k>` (k-th proviso of that provision) | `cpa2019:s58(1)(a)(i)-prov1`, `cpa2019:s55(1)-prov2` |
| Explanation | `<parent>-expl<k>` | `cpa2019:s2(7)-expl1` |
| schedule | `<doc>:sch<n>` | `commissions2020:sch1` |
| preamble / enacting text | `<doc>:preamble` | `ecom2020:preamble` |
| other un-numbered headings | `<doc>:h-<slug>` | `cpa2019:h-statement-of-objects-and-reasons` |
| generic documents | `<doc>:h1`, `<doc>:h1.2`, ... | `report:h1.2` |

* `level` is one of `chapter, section, rule, subsection, clause, subclause, explanation, proviso,
  schedule, heading`. In Rules, sub-rules use the level `subsection`.
* Each node stores `text` (the node including its descendants), `parent_id`, `children`,
  `page_start/page_end` and `char_start/char_end` into `clean/<doc_id>.txt`; definition clauses
  also store `defined_term`.
* If the source itself repeats a label inside one provision (for example `ecom2020` rule 4(1)
  prints "(a)" twice), the second node gets a `~2` suffix (`ecom2020:r4(1)(a)~2`) and a warning is
  written to the structure report.
* **PowerShell:** quote IDs that contain parentheses, e.g. `python scripts/browse.py show
  "cpa2019:s35(1)"`. PowerShell otherwise splits or evaluates the parentheses.

## Chunkers

All chunkers are deterministic and configured in `configs/chunking.yaml`; every chunk records
`doc_id`, `node_ids`, `primary_node_id`, `section_ids`, `heading_path`, pages, character offsets
into the cleaned text, `n_tokens` (bge-small tokenizer) and `xref_targets`.

* `fixed_256` / `fixed_512`: sliding token windows (overlap 32 / 64) snapped to word boundaries.
* `semantic`: legal-aware sentence/clause units embedded with `BAAI/bge-small-en-v1.5` on CPU
  (cached in `data/processed/cache/`), breakpoints above the 90th percentile of adjacent cosine
  distances, then 64–512 token limits enforced.
* `structural`: one chunk per section/rule, split at sub-section then clause boundaries above
  384 tokens, tiny siblings (< 48 tokens) merged, every definition clause its own chunk; the
  embedded `text` is prefixed with the heading path, `display_text` is the exact cleaned text.

## Writing evaluation questions

Questions live in `data/eval/questions.jsonl`, one JSON object per line. The target is 90
answerable and 30 unanswerable questions.

| field | type | notes |
|---|---|---|
| `id` | string | `q001`, `q002`, ... (unique) |
| `question` | string | |
| `type` | string | `definition`, `factual`, `procedural`, `numeric`, `cross_reference`, `multi_section`, `unanswerable` |
| `answerable` | bool | `false` exactly when `type` is `unanswerable` |
| `gold_node_ids` | list of strings | node IDs from `nodes.jsonl`; empty for unanswerable questions |
| `gold_answer` | string | short reference answer; for unanswerable questions a one-line reason |
| `difficulty` | string | `easy`, `medium`, `hard` |
| `author` | string | `Swapnil` |
| `notes` | string, optional | |

Example line:

```json
{"id": "q001", "question": "What does \"e-commerce\" mean under the Consumer Protection Act, 2019?", "type": "definition", "answerable": true, "gold_node_ids": ["cpa2019:s2(16)"], "gold_answer": "Buying or selling of goods or services, including digital products, over a digital or electronic network.", "difficulty": "easy", "author": "Swapnil"}
```

Workflow:

1. find the provision: `python scripts/browse.py search "<words>"` or `python scripts/browse.py list --doc ecom2020`;
2. read it with sub-node IDs inline: `python scripts/browse.py show "ecom2020:r4"` (also lists what it refers to and what refers to it);
3. pick the most specific node(s) that contain the answer as `gold_node_ids`;
4. run `python scripts/validate_eval.py`. It checks the schema, unique ids, that every gold node
   exists (suggesting the closest IDs when one does not) and answerable/unanswerable
   consistency, prints counts by type, difficulty and document, and shows progress toward
   90 + 30. It exits non-zero on errors.

The three questions shipped in the file are examples built from real nodes; their `notes` say
"example — replace or keep after review", and the validator warns until that note is removed.

## Tests and CI

```powershell
pytest -q              # synthetic fixtures only (corpus tests are deselected by default)
pytest -q -m corpus    # integration test on the real PDFs in data/raw (skipped if absent)
ruff check . ; ruff format --check . ; mypy --strict src
```

Unit tests generate small PDFs with PyMuPDF (an Act-like document, a Rules-like notification and
a non-legal report). GitHub Actions runs lint, type checks and tests on `ubuntu-latest` and
`windows-latest` with Python 3.11, installs CPU torch from the PyTorch CPU index and caches the
HuggingFace model directory.

## Limitations

* Single-column layouts are assumed; text objects on the same baseline are merged, so true
  multi-column pages would interleave.
* Tables are not reconstructed. The fee table in `commissions2020` rule 7 and the quarterly
  report form in its Schedule I are kept as plain text of the rule/schedule (rows become lines).
* Clause hierarchy is inferred from labels, numbering continuity and indentation. Ambiguous
  `(i)`/`(v)`/`(x)` labels are resolved from the preceding siblings and indentation; text that
  follows a list without a label is attached to the provision that introduced the list.
* A proviso or Explanation is attached to the provision it follows; scope words such as "this
  sub-section" or "this rule" are used when present.
* Cross-references written loosely in the source ("of that section", "the said section", "under
  the said sub-section") may stay unresolved; references to other statutes are kept as
  `external`. When a cited provision does not exist (e.g. a drafting error such as
  "sub-clause (zg) of sub-section (1) of section 101"), the edge points to the deepest existing
  ancestor and is marked `partial`.
* Amendment brackets of the India Code copy (`[ [(1A) ...]]`) are kept in the text; amendment
  footnotes are kept as footnote records, not merged into the provisions.
* Hindi text is removed, not translated; scanned (image-only) PDFs are skipped because OCR is
  out of scope.

## License

MIT, see [LICENSE](LICENSE).
