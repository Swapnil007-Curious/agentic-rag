# Agnostic Agentic-RAG: Hybrid Vector-Semantic Document Intelligence for Niche Low-Resource Corpora

This repository contains three parts of the project.

* **Document ingestion:**
  * PDF text extraction and cleaning;
  * legal-structure parsing and cross-reference extraction;
  * three chunking strategies behind one interface, with corpus statistics;
  * the evaluation-set schema, with a validator and a section browser.
* **Retrieval:**
  * BM25, dense embeddings, rank fusion and cross-encoder reranking;
  * structure-aware cross-reference expansion;
  * a retrieval evaluation harness (dev/test split, ablation grid, paired bootstrap);
  * an interactive search tool;
  * an OpenAI-compatible LLM client, used to build the evaluation set automatically (with
    automatic quality filters) and to draft questions for human review.
* **Answering:**
  * a Synthesizer agent that answers only from retrieved passages, with a citation after every
    sentence, and a Critic agent with one bounded revision round;
  * deterministic hallucination guards (citation validity, number grounding) and calibrated
    abstention (a retrieval-score threshold tuned on dev);
  * an answer-quality evaluation harness (NLI-based faithfulness and correctness, citation
    quality, abstention, cost) with the same dev/test discipline as retrieval;
  * a local web UI with clickable citations.

"Agnostic" means the pipeline is domain-independent: document structure comes from a pluggable
`StructureParser`. The case-study corpus uses `IndianStatuteParser`; any other PDF can be
ingested with `GenericParser` (font-size / numbering based headings) by adding it to
`configs/ingest.yaml` with `parser: generic`.

The system serves several **collections** (document sets), each with its own configs, data,
caches, evaluation set and results: `consumer_law` (the case study above, at its original
locations) and `cs_ai` (selected chapters of three open-licensed computer science textbooks).
See [Collections](#collections).

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

## Collections

A collection is a named document set defined in
[`configs/collections.yaml`](configs/collections.yaml): its four config files (ingest,
chunking, retrieval, generation), a domain profile (prompt wording, question-type mix and
out-of-scope topics, in `src/ragsys/profiles.py`), UI texts and four root directories. Every
other location (embeddings, the LLM response cache, the rerank-score and NLI caches, result
tables, the test-access log, the abstention threshold, UI feedback) is derived from those roots,
so collections never share an output or a cache. The test suite checks that the paths written in
each collection's config files agree with its roots and that no two collections share a
location.

| collection | configs | raw / processed / eval / results |
|---|---|---|
| `consumer_law` (default) | `configs/{ingest,chunking,retrieval,generation}.yaml` | `data/raw`, `data/processed`, `data/eval`, `results` |
| `cs_ai` | `configs/collections/cs_ai/*.yaml` | `collections/cs_ai/{raw,processed,eval,results}` |

`consumer_law` maps to the original locations and config files, so every command written
without `--collection` behaves exactly as before (a test compares the resolved paths with the
original ones, the consumer-law prompts and system fingerprints with a snapshot taken before
collections existed, and the files in `results/tables/` with their recorded hashes).

Every script takes `--collection NAME` (default `consumer_law`); explicit `--config` and path
flags still override the collection's files:

```powershell
python scripts/ingest.py --collection cs_ai
python scripts/chunk.py --collection cs_ai --chunker all          # fixed_512 + structural
python scripts/browse.py --collection cs_ai list --doc d2l
python scripts/search.py --collection cs_ai "how does git rebase work" --config-id "structural|bm25"
python scripts/ask.py --collection cs_ai "What problem does the attention mechanism solve in sequence models?" --system G2
python scripts/serve.py                                           # every collection, selector in the page
```

### The `cs_ai` corpus

Selected chapters of three open-licensed textbooks; put the PDFs in `collections/cs_ai/raw/`
(they are not redistributed). Chapters are selected by title in
[`configs/collections/cs_ai/ingest.yaml`](configs/collections/cs_ai/ingest.yaml); their page
ranges are resolved from each PDF's outline (bookmarks), or by heading detection when a title is
not in the outline, and recorded in the structure report and in
[`collections/cs_ai/raw/SOURCES.csv`](collections/cs_ai/raw/SOURCES.csv), which also holds the
authors, edition, licence (with where it was read), source URL, SHA-256 and page counts. The web
UI shows this attribution in its footer.

| doc_id | file | book | chapters | licence |
|---|---|---|---|---|
| `d2l` | `d2l.pdf` | Aston Zhang, Zachary C. Lipton, Mu Li, Alexander J. Smola: *Dive into Deep Learning* (PyTorch edition) | Multilayer Perceptrons; Convolutional Neural Networks; Recurrent Neural Networks; Attention Mechanisms and Transformers; Optimization Algorithms | CC BY-SA 4.0 (book text; the sample code is under a modified MIT licence) |
| `progit` | `progit.pdf` | Scott Chacon, Ben Straub: *Pro Git*, second edition | Getting Started; Git Basics; Git Branching | CC BY-NC-SA 3.0 Unported |
| `ods` | `ods.pdf` | Pat Morin: *Open Data Structures* (pseudocode edition) | Array-Based Lists; Linked Lists; Hash Tables; Binary Trees; Heaps; Sorting Algorithms; Graphs | CC BY 2.5 Canada |

The licence is read from the PDF's licence page when the PDF states one with a version (Pro
Git). The d2l PDF has no licence page and the ODS PDF names "a Creative Commons Attribution
license" without a version; for these two the licence comes from the publishers' own pages
(the d2l-en repository README; the licence link on opendatastructures.org), and SOURCES.csv says
so in its `licence_source` column.

Textbooks are ingested with `doc_type: textbook` and `parser: generic`:

* **cleaning** (`src/ragsys/ingest/textbook.py`): running headers and footers are removed by
  learning the offset between PDF pages and printed page numbers (plus repetition and
  configured patterns, e.g. ODS's "§2.1 Array-Based Lists"); lines set in a monospace font
  become code blocks that keep their line breaks and indentation (ODS's pseudocode, set in the
  text font, is recognised by indentation); math is kept as extracted text, with small raised
  or lowered characters written as `^2` / `_i`; sections titled Exercises, References,
  Bibliography, Index, Contents, Solutions or Further Reading are dropped with their
  subsections, as are exercise lists inside a section ("Exercise 2.1. ..."); front matter and
  unselected chapters are never extracted;
* **structure**: headings are lines set larger than the body text or in a heading-only font
  family with section numbering; levels come from the numbering ("5" chapter, "5.1" section,
  "5.1.2" subsection), otherwise from the font size. Numbered headings get IDs
  `<doc>:<number>` (`d2l:11.5`, `ods:5.2.1`), unnumbered ones `<doc>:h-<slug>`
  (`progit:h-rebasing`; a slug used more than once is qualified by its parent:
  `progit:h-git-basics-summary`), code blocks `<parent>/code-<k>` inside their section;
* **cross-references**: "Section 4.2", "Sections 4.2 and 4.3", "Chapter 3" resolve to nodes
  (`exact`, or `partial` to the nearest existing ancestor); references into chapters that were
  not ingested are `external`; figure and table references stay `unresolved`; citations of
  other books ("[46, Section 2.2.2]") are ignored;
* **chunking**: `fixed_512` and `structural` only, with the consumer-law size rules; structural
  chunks also cover a chapter's introduction before its first section;
* **indexes**: `bge-small` and `bge-base` only (bge-large is too slow on a laptop CPU for this
  corpus); embedding runs in batches with a progress line and an ETA and is resumable;
* **evaluation set**: `build_auto_eval.py --collection cs_ai` with the `cs_ai` profile: 60
  answerable questions (definition 15, factual/conceptual 15, procedural 12, numeric 8,
  multi-section 10; cross-reference questions are added only when enough resolved references
  between eligible passages exist, and the decision is recorded in `eval_construction.json`) and
  20 unanswerable ones on realistic topics the selected chapters do not cover (Kubernetes,
  quantum computing, reinforcement learning, facts about recent models, company practices,
  chapters that were not ingested), with the same filters and retrieval check as consumer_law;
* **retrieval grid** (16 configs): chunker {fixed_512, structural} x {bm25, dense bge-small,
  dense bge-base, hybrid_rrf bm25+bge-base} x rerank {off, MiniLM}, xref expansion off; the same
  selection rule and baselines (naive RAG = `fixed_512|dense|bge-base`);
* **generation**: G1 (naive), G2 (`retrieval: "selected:best_overall"`, the best dev retrieval
  config written by `select_configs.py`), G3 (G2 + guards, derived), G4 (G2 + guards + critic).
  G5 and G6 are not run for cs_ai (G5 underperformed on consumer_law and the cs_ai dev split is
  small; G6 is the slowest system); the reason is recorded in the generation config and tables.
  Before the selection exists, `ask.py` and the web UI use the first built config of
  `selection_fallback` and say which one; evaluations never fall back.

### Overnight runner

```powershell
.\scripts\run_collection_pipeline.ps1 -Collection cs_ai          # or:
python scripts\run_collection_pipeline.py --collection cs_ai
python scripts\run_collection_pipeline.py --collection cs_ai --dry-run   # plan + time estimate only
```

Runs, in order: ingest -> chunk -> build_index -> build_auto_eval -> validate_eval (stops on
failure) -> `ollama stop` (frees the LLM's memory) -> retrieval dev grid -> select_configs ->
eval_retrieval on test -> run_generation on dev (G2, G4) -> derive_systems on dev (G3) ->
eval_generation on test (G1-G4, deriving G3) -> `SUMMARY.md` with the key tables in
`<results_dir>/tables/`. Before starting it prints a time estimate per step, computed from the
corpus size and the speeds recorded in this repository's own logs (embedding times, LLM latency
per phase, reranking time and end-to-end latency per question). Each step is skipped when it is
already complete with unchanged inputs; interrupted steps resume from their own caches, so after
a crash or a reboot the same command continues. Every step's output, exit code, wall time and
peak memory go to `<results_dir>/pipeline_log.txt`; the run ends with `PIPELINE COMPLETE` or
names the failing step and the command to resume. While it runs, Windows is kept from sleeping.
Ollama must be running for the evaluation-set and generation steps.

### Adding a collection

1. Put the PDFs in `collections/<name>/raw/`.
2. Copy `configs/collections/cs_ai/` to `configs/collections/<name>/` and edit the documents
   (file, doc_id, title, authors, chapters to keep, licence fallback) and every
   `collections/cs_ai/...` path.
3. Add an entry to `configs/collections.yaml` (roots, UI texts, profile, auto-eval counts,
   pipeline systems). A new domain gets its own profile in `src/ragsys/profiles.py` (prompt
   wording, question-type mix, out-of-scope topics); the `cs_ai` profile suits other technical
   textbooks.
4. Run `python scripts/run_collection_pipeline.py --collection <name>` (or step by step with
   `--collection <name>`), and `pytest -q` to check that the configs agree with the roots.

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
HuggingFace cache. Building the retrieval indexes also downloads:

* `BAAI/bge-base-en-v1.5` (about 440 MB);
* `BAAI/bge-large-en-v1.5` (about 1.3 GB);
* `cross-encoder/ms-marco-MiniLM-L-6-v2` (about 90 MB).

The answer evaluation downloads `cross-encoder/nli-deberta-v3-base` (about 740 MB) on first use.

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
| `author` | string | `Swapnil`; `seed example` for q001-q003; automatic items: `auto (<model>) - filtered` |
| `notes` | string, optional | |
| `origin` | string, optional | `human` (default), `llm_draft_human_reviewed` or `llm_auto_filtered` |
| `gen_meta` | object, optional | automatic items only: method, filters passed, source nodes, lexical overlap, model, seed, ... |

Example line:

```json
{"id": "q001", "question": "What does \"e-commerce\" mean under the Consumer Protection Act, 2019?", "type": "definition", "answerable": true, "gold_node_ids": ["cpa2019:s2(16)"], "gold_answer": "Buying or selling of goods or services, including digital products, over a digital or electronic network.", "difficulty": "easy", "author": "seed example", "notes": "seed example created during tooling development"}
```

Workflow:

1. find the provision: `python scripts/browse.py search "<words>"` or `python scripts/browse.py list --doc ecom2020`;
2. read it with sub-node IDs inline: `python scripts/browse.py show "ecom2020:r4"` (also lists what it refers to and what refers to it);
3. pick the most specific node(s) that contain the answer as `gold_node_ids`;
4. run `python scripts/validate_eval.py`. It checks the schema, unique ids, that every gold node
   exists (suggesting the closest IDs when one does not) and answerable/unanswerable
   consistency, prints counts by type, difficulty and document, and shows progress toward
   90 + 30. It exits non-zero on errors.

The first three questions (q001-q003) are seed examples built from real nodes while the tooling
was developed; their author is `seed example` and their `notes` say so. The validator warns about
any other author name, and about items whose notes still carry the old "replace or keep after
review" marker.

## Retrieval

### Architecture

```
query
  |-- BM25Okapi over chunk text (legal-aware tokenizer) ----------\
  |                                                                >-- fusion: RRF (k=60), or linear:
  '-- bge encoder (query instruction) -> exact cosine (NumPy) ----/    alpha*dense + (1-alpha)*BM25
                                                                       (min-max normalised)
first-stage ranking (first_stage_k = 50)
  |
  |-- xref expansion: follow the resolved cross-references of the top M (5) chunks and add
  |   the chunks covering the referenced nodes (at most 10), provenance "xref_from:<chunk_id>"
  |
  |-- rerank on : window (top rerank_n = 20) + expanded chunks -> cross-encoder -> sorted
  |   rerank off: window (top final_k) with each expanded chunk right after its source
  v
ranked list (rest of the first stage appended)  ->  results = top final_k (10)
```

Every result keeps its first-stage rank and score, the rank and score it had in each component
(BM25, dense), the rerank score and its provenance; every query records per-stage timings in ms.

Code lives in `src/ragsys/retrieval/`:

| module | content |
|---|---|
| `config.py` | `RetrievalConfig` (one retriever; unknown keys raise), `RetrievalSettings` (configs/retrieval.yaml), config ids |
| `bm25.py` | tokenizer (statute references such as `section 2(7)` also emit `s2(7)` and `2(7)`; built-in stopwords; optional light suffix stripping) and `BM25Index` |
| `embed.py` | sentence-transformers encoders on CPU (normalised embeddings) and the corpus embedding cache |
| `store.py` | `VectorStore` protocol, `NumpyStore` (exact cosine, used for all experiments), `ChromaStore` (chromadb `PersistentClient`, cosine space, precomputed embeddings only) |
| `fusion.py` | Reciprocal Rank Fusion and weighted linear fusion |
| `rerank.py` | cross-encoder reranker (`cross-encoder/ms-marco-MiniLM-L-6-v2`, optionally `BAAI/bge-reranker-base`) over chunk `text` (structural chunks include their heading prefix) |
| `xref_expand.py` | structure-aware cross-reference expansion; target nodes are mapped to chunks through `node_ids`, so it works for every chunker |
| `pipeline.py` | `Retriever` and `RetrievalResources` (indexes and models shared across a grid) |
| `fakes.py` | hashing encoder and word-overlap cross-encoder (model names `fake:...`) used by tests |

### Configuration and config ids

`configs/retrieval.yaml` holds the paths, the model registry (`bge-small`, `bge-base`,
`bge-large`, `minilm`, `bge-reranker-base`), the defaults used by `search.py`, the evaluation
settings, the dev grid and the baselines. A config id names one retriever, for example
`structural|hybrid_rrf|bge-base|xref|rerank-minilm`, `fixed_512|dense|bge-base` or
`fixed_512|bm25`. Parameters that are not spelled out in the id (BM25 k1/b, RRF k, depths, ...)
add a `cfg-<hash>` part when they differ from the defaults. Result folders use `__` instead of
`|`.

### Build the indexes

```powershell
python scripts/build_index.py --chunker all --models bge-small bge-base bge-large
```

For every chunker this builds the BM25 index. For every model it then computes or loads the corpus
embeddings in `data/processed/emb/<model>/<chunker>.npy`. A JSON sidecar stores the chunk_id order
and a hash of the embedded texts, and a stale cache is rebuilt automatically. Finally it builds a
Chroma collection under `data/processed/chroma/` and checks that, with every chunk used as a
query, Chroma's top-10 equals exact NumPy search. Timings and sizes are written to
`data/processed/emb/index_report.json`.

### Search

```powershell
python scripts/search.py "Who can file a complaint about defective goods?"
python scripts/search.py "Who can file a complaint about defective goods?" --config-id "structural|hybrid_rrf|bge-base|xref|rerank-minilm"
python scripts/search.py "Who can file a complaint about defective goods?" --chunker fixed_512 --retriever dense --model bge-base --no-xref --no-rerank
```

Quote config ids in PowerShell, because they contain `|`. Output is ASCII-folded, as in
`browse.py`.

## Retrieval evaluation

### Protocol

* **Dev/test split by hash.** A question is in `dev` if the first 8 hex digits of
  `sha256("split-v1:" + question id)`, divided by 2^32, are below 0.30; otherwise it is in
  `test`. The split is computed, never stored, so adding questions never moves existing ones.
  `validate_eval.py` prints counts per split x type x origin.
* **Test split guard.** Only `scripts/eval_retrieval.py` (and, for answer generation,
  `scripts/eval_generation.py`) can read test questions:
  * the harness needs a grant, and `grant_test_access()` only issues it to those scripts;
  * every grant is appended to `results/test_access_log.json` (UTC time, git commit, dirty flag,
    configs, number of questions);
  * a test scans the source tree to check that no other file asks for a grant;
  * the grid runner only accepts `--split dev`.

  All tuning and selection use dev.
* **Relevance.** A chunk *covers* a node when it contains part of that node's own text, i.e. the
  part not inside a child. A chunk is relevant to a gold node if a node it covers is:
  * the gold node itself;
  * a descendant of the gold node;
  * an ancestor of the gold node below chapter level.

  Ancestry comes from the parent links in nodes.jsonl, not from ID prefixes, because IDs contain
  `~2`, `-prov1` and `-expl1`. Unanswerable questions have no relevant chunks.
* **Metrics.** Computed per question and averaged over answerable questions:
  * Recall@{1,3,5,10}: the fraction of gold nodes covered by the top k chunks;
  * Hit@k, MRR@10, and nDCG@10 with a binary gain per chunk;
  * **token-budget coverage** at 512 / 1024 / 2048 tokens: the fraction of gold nodes covered
    when ranked chunks are taken in order until the next one would exceed the budget (counted
    with chunk `n_tokens`).

  Token-budget coverage is the headline metric and is reported next to Recall@k. Chunkers
  produce chunks of very different sizes (a structural chunk is often a quarter of a fixed_512
  chunk), so Recall@k alone favours large chunks; coverage at a fixed budget compares what a
  generator would actually receive. Budget metrics use the whole ranked list (the reranked head
  plus the first-stage tail), not just the top 10.
* **Latency** per stage (BM25, query encoding, dense search, fusion, xref, rerank, total): p50
  and p95 in ms, measured after a warm-up query. Rerank scores (SQLite, across runs) and query
  embeddings (in memory, within a run) are cached; on a cache hit the latency measured when the
  value was first computed is reported for that stage and included in `total`, so latency
  figures do not depend on whether a cache was warm.
* **Unanswerable questions.** The top-1 first-stage score, and the top-1 rerank score when
  reranking is on, are summarised separately for answerable and unanswerable questions. The AUROC
  of the top-1 score for separating the two groups is reported as input for later abstention
  thresholds.
* **Breakdowns** by question type, document and origin.
* **Paired bootstrap.** 1000 seeded resamples of the difference to the naive RAG baseline
  (`fixed_512|dense|bge-base`) on Recall@5 and coverage@1024, giving the mean difference and a 95%
  interval.

### Workflow

```powershell
# 1. dev grid (resumable). 104 configs:
#    4 chunkers x {bm25, dense bge-small/base/large, hybrid_rrf bge-base,
#                  hybrid_linear bge-base a=0.5} x xref {off,on} x rerank {off,minilm},
#    plus hybrid_rrf bge-large per chunker with rerank off/on
python scripts/run_retrieval_grid.py --split dev
# 2. choose configs on dev: best by coverage@1024 (ties: Recall@5, then lower latency),
#    overall and per chunker, plus the baselines
python scripts/select_configs.py
# 3. once, at the end: evaluate the selected configs on test
python scripts/eval_retrieval.py --split test --configs-from results/tables/selected_configs.json
```

Outputs:

| path | content |
|---|---|
| `results/retrieval/<split>/<config>/` | `per_query.jsonl` and `summary.json` (with config snapshot and run info) |
| `results/tables/retrieval_dev.csv` | one row per config: all metrics, latency, question counts |
| `results/tables/retrieval_dev_*.csv` | breakdowns `_by_type`, `_by_document`, `_by_origin`, and `_bootstrap` |
| `results/tables/selected_configs.json` | the configs chosen on dev and the rule used |
| `results/tables/retrieval_test*.csv` | the same tables for the test evaluation |

Every table has a `.meta.json` with the settings and run info.

Memory and speed (both `run_retrieval_grid.py` and `eval_retrieval.py`):
* configs run grouped by dense model: BM25-only configs first (no encoder loaded), then one
  group per encoder; at most one sentence-transformers encoder is in memory, and it is released
  (`del` + `gc.collect()`) before the next group. Tables keep the order of the grid.
* the cross-encoder is loaded once and shared by all configs; its scores are cached in
  `data/processed/rerank_cache/<model>.sqlite`, keyed by (model, sha256(query), chunk_id,
  sha256(chunk text)), together with the per-pair latency measured when the pair was first
  scored. `--no-rerank-cache` always rescores.
* query embeddings are cached in memory per (model, query) for the run.
* torch uses 4 CPU threads (`--threads`). The RSS is printed at group boundaries, an ETA after
  every config and the peak RSS at the end.

Options:
* the grid skips configs whose outputs exist for the same dev question set, unless you pass
  `--overwrite`;
* `--only TEXT` runs a subset of the grid; the tables still cover every config with current
  outputs;
* `--smoke` writes to `results/smoke/` and labels every row "smoke - not results".

`eval_retrieval.py` refuses to run when there are fewer than 20 answerable questions. It also
refuses a second test evaluation unless you pass `--allow-repeat`, and the repeat is logged.

## LLM client

`src/ragsys/llm/client.py` is a single client for any OpenAI-compatible chat endpoint (the `openai`
SDK with a `base_url`), configured in `configs/llm.yaml`. It builds and drafts evaluation
questions and runs the answering agents (with the overrides of `configs/generation.yaml`:
temperature 0, max_tokens 512).

* **Ollama (default):** install Ollama, run `ollama pull qwen2.5:3b` and keep it running. The
  default `base_url` is `http://localhost:11434/v1`; no key is needed.
* **Any OpenAI-compatible provider:** set `base_url` and `model` in `configs/llm.yaml` and put the
  key in the environment variable `LLM_API_KEY` (PowerShell: `$env:LLM_API_KEY = "..."`).
  * The key is read from the environment only.
  * It is never logged, printed, cached or written to reports.
  * `.env` and key files are git-ignored.

Features:
* temperature, max_tokens and seed;
* JSON mode, with a tolerant JSON extractor as fallback;
* retries with exponential backoff (at most 5) on rate limits, timeouts, connection errors and
  5xx responses;
* per-call latency and token usage;
* an on-disk response cache in `data/processed/llm_cache/`, keyed by (base_url, model, messages,
  parameters), so reruns are free and reproducible. A cache hit reports `latency_ms = 0` (no
  model time spent now) and carries the latency measured on the original call in
  `stored_latency_ms`; `reported_latency_ms` is what latency figures use, so they do not depend
  on whether the cache was warm.

Tests use `FakeLLM` and a fake HTTP transport; no test calls a network endpoint.

## Evaluation set construction

The evaluation set is built automatically with the local model of `configs/llm.yaml`
(qwen2.5:3b through Ollama by default). The model is small and makes factual mistakes, so a
generated item is kept only if it passes every automatic check. The three hand-written examples
in `questions.jsonl` (and any other `human` or `llm_draft_human_reviewed` item) are never changed.

```powershell
python scripts/build_auto_eval.py          # 1-3 hours on a laptop CPU; resumable
python scripts/validate_eval.py
python scripts/spot_check.py --n 20 --seed 0   # optional manual check (see below)
```

* Every LLM call goes through the on-disk response cache (`data/processed/llm_cache/`). If the
  run is interrupted, run the same command again: finished calls are answered from the cache.
  Progress lines show the elapsed time and an ETA per phase.
* Nothing is written until the end. The script refuses to run when `questions.jsonl` already
  contains `llm_auto_filtered` items; `--rebuild` replaces exactly those items and keeps every
  other line byte-for-byte.
* The bge-small encoder used for duplicates and the retrieval check is released before the
  script ends. Do not run the retrieval grid at the same time; `ollama stop qwen2.5:3b` frees
  the model's memory afterwards.

### Answerable questions

1. **Candidates.** 200 targets from the drafting code (same type mix as `draft_questions.py`,
   stratified by document and node level, seed 0; nodes used by existing questions are
   excluded), one drafting call each. `gold_node_ids` are the source node(s).
2. **Filters**, applied in this order; each is logged with pass/fail counts and reasons:

   | filter | rule |
   |---|---|
   | `format` | valid JSON with a question and an answer; question 6-40 words; answer at most 40 words |
   | `provision_number` | no section, sub-section, rule or clause number in the question (allowed for `cross_reference`) |
   | `copy` | lexical overlap with the source text at most 0.6; otherwise the LLM paraphrases the question once in plain citizen language and the paraphrase is checked again (overlap, length, provision numbers) |
   | `numbers` | deterministic: every number in the gold answer, written as digits or English words ("30", "thirty", "Rs. 50,00,000", "fifty lakh"), occurs in the source in some form (compared by value); numbers inside a provision reference may also match the source node ids |
   | `support` | LLM judge on source text + question + answer, strict JSON `{"supported": true\|false, "corrected_answer": "..."}`; only `supported: true` is kept; a corrected answer is recorded but never used |
   | `answerable_from_source` | the LLM answers the question from the source text alone; the item is dropped when that answer states different numbers than the gold answer, or names a different entity of the same kind (capitalised terms such as "State Commission" vs "District Commission") |
   | `near_duplicate` | bge-small cosine similarity above 0.90 to an existing question or to an earlier candidate: keep the first |

3. **Selection.** Up to 90 survivors, stratified by type (the drafting mix; a type with too
   few survivors passes its share to the others) and spread as evenly over documents as
   availability allows. A shortfall is reported, not filled.

### Unanswerable questions

1. **Candidates** (90): 45 out-of-scope questions (3 per call over 15 topics the corpus does not
   contain: state-specific fees and procedures, other statutes such as IPC/BNS, the Motor
   Vehicles Act, the RTI Act, the Income-tax Act and the Contract Act, the repealed Consumer
   Protection Act, 1986, case law, office addresses and phone numbers, timelines of real cases,
   banking ombudsman schemes) and 45 perturbations of accepted answerable questions (the key
   entity or scope moved to another law, a specific state, a specific company or a specific
   year).
2. **Filters:** `format` (6-40 words); `corpus_provision_number` (the question names no section
   or rule number that exists in this corpus; applied before the LLM step to save calls);
   `near_duplicate` (also against all answerable questions); `retrieval_check`: the hybrid
   retriever `structural|hybrid_rrf|bge-small` (BM25 + bge-small, RRF) returns its top 5 chunks
   and the LLM must answer strict JSON `{"answered": false}`.
3. **Selection:** 30, balanced across the two methods (and over topics / perturbation kinds
   within a method). The gold answer is a one-line reason derived from the generation method,
   e.g. "Not in the corpus: it asks about the Right to Information Act, 2005, which the Consumer
   Protection Act, 2019 and its four Rules do not cover."

### Outputs

* `data/eval/questions.jsonl`: the selected items are appended with ids continuing from the
  existing ones, `origin: "llm_auto_filtered"`, `author: "auto (qwen2.5:3b) - filtered"` and
  `gen_meta` (method, filters passed, source nodes, lexical overlap, model, seed, prompt
  version; for answerable items also the paraphrase history, the judge's verdict and the
  independent answer; for unanswerable items the topic or perturbation target and the ids of the
  five retrieved chunks).
* `results/tables/eval_construction.json`: candidates generated, pass/fail counts and reasons per
  filter (with examples), paraphrase statistics, selection and shortfall, final counts by type,
  document and split, mean lexical overlap, LLM calls (cached / uncached, model time), wall
  time, peak memory, model name and run info.

Code: `src/ragsys/eval/autogen.py` (pipeline and prompts), `src/ragsys/eval/checks.py`
(deterministic checks), `src/ragsys/eval/spot_check.py`.

### Spot check (optional)

```powershell
python scripts/spot_check.py --n 20 --seed 0     # y = correct, n = wrong, s = skip, q = quit
python scripts/spot_check.py --apply-removals    # only if you decide to remove the wrong items
python scripts/validate_eval.py
```

* The sample is a seeded random draw of `llm_auto_filtered` items, stratified into answerable
  and unanswerable items in proportion to their numbers (at least one of each).
* Answerable items are shown with the text of their gold nodes. Judge an item correct when the
  question is answerable from that text and the gold answer is right.
* Unanswerable items are shown with the five passages retrieved during construction. Judge an
  item correct when the corpus really does not answer the question.
* `results/tables/eval_spot_check.json` records the counts, the error rate among judged items
  with a 95% Wilson score interval (overall and per stratum) and the ids marked wrong. The
  session itself never edits `questions.jsonl`.
* `--apply-removals` removes the items marked wrong. It only removes `llm_auto_filtered` items
  whose question text is still the one that was judged. Removing items changes the dev/test
  question sets, so the grid re-runs the affected configs automatically. Do the spot check
  before the one-time test evaluation.

### Disclosure wording for the paper

Fill in the numbers from `eval_construction.json`, `validate_eval.py` and (if done)
`eval_spot_check.json`:

> The evaluation set contains <N> questions (<A> answerable, <U> unanswerable). <H> questions
> were written by the author; the remaining <M> were generated automatically by a small
> open-weight language model (qwen2.5:3b, run locally, temperature 0.2, seed 0) and kept only if
> they passed a fixed sequence of automatic filters. They were not individually reviewed by a
> person. Answerable questions were generated from provisions sampled by question type and
> document, with the source provision(s) as gold evidence. Items were removed when the question
> copied the source wording (lexical overlap above 0.6 after one paraphrase attempt), cited
> provision numbers, contained numbers in the answer that do not occur in the source, were
> judged unsupported by the source by the same model, received a conflicting answer (different
> numbers or entities) when the model answered from the source alone, or were near-duplicates
> (cosine similarity above 0.90 with bge-small). Unanswerable questions were generated for
> topics outside the corpus and by changing the entity or scope of accepted questions; they were
> kept only if the model judged that the top five passages of a hybrid BM25 + dense retriever did
> not answer them. Every item records its origin and generation metadata, and results are
> reported separately by origin. [If a spot check was done:] A seeded random sample of <n>
> automatically generated items, stratified by answerability, was checked manually; <k> were
> judged incorrect (error rate <p>, 95% Wilson interval [<lo>, <hi>]).

## Drafting and reviewing questions

```powershell
python scripts/draft_questions.py --n 150 --seed 0     # appends pending drafts to data/eval/drafts.jsonl
python scripts/review_drafts.py                        # a=accept e=edit r=reject s=skip q=quit
python scripts/review_drafts.py --add-unanswerable     # type unanswerable questions by hand
python scripts/validate_eval.py
```

### draft_questions.py

* **Type mix:** definition 15, factual 20, procedural 15, numeric 15, cross_reference 15,
  multi_section 10, scaled to `--n`. Restrict it with `--types`.
* **Sampling:** target nodes are stratified by document (each document's quota is proportional to
  the square root of its candidate count) and by node level, with a seeded RNG.
* **Pairs:**
  * cross-reference drafts use resolved cross-reference pairs;
  * multi-section drafts pair a Rule provision with an Act provision, using explicit references
    first and otherwise the best BM25 match.
* **Prompt:** the model is asked for:
  * a question that is answerable only from the given text;
  * paraphrased, citizen-style wording;
  * no provision numbers, except for cross_reference;
  * a gold answer under 40 words.
* **Each draft records:**
  * `gold_node_ids` from the source nodes, the source text and the model name;
  * a lexical-overlap score (the share of question words found in the source text); drafts
    above 0.6 are flagged as copy-like.
* **Other rules:**
  * nodes already used by a draft or a question are skipped;
  * the script never writes `questions.jsonl` and never drafts unanswerable questions;
  * `--fake` runs it with the deterministic FakeLLM.

### review_drafts.py

* Shows each pending draft together with its source text.
* **Accept** appends the draft to `questions.jsonl` with the next free id,
  `origin: "llm_draft_human_reviewed"` and author Swapnil.
* **Edit** lets you change the question, answer, gold IDs (checked against nodes.jsonl), type and
  difficulty (Enter keeps a value), then shows the draft again.
* Progress is saved after every action.
* **`--add-unanswerable`** asks for a question and the reason the corpus cannot answer it, shows
  the top 5 BM25 hits so you can confirm that, and appends it with `answerable: false` and
  `origin: "human"`.

### Disclosure

Every question carries an `origin` field:

* `human`: written by hand (the default for entries without the field);
* `llm_draft_human_reviewed`: drafted by a language model from the provision text, then read,
  possibly edited, and accepted by a person. The `notes` of such questions name the draft and the
  model.
* `llm_auto_filtered`: generated by `scripts/build_auto_eval.py` and kept only because it passed
  every automatic filter; not individually reviewed by a person (see below).

Results can be broken down by origin (`retrieval_*_by_origin.csv`).

## Answering

### Agent architecture

```
question
  |
  |-- retriever (RetrievalConfig of the system) --> ranked chunks + abstention score
  |
  |-- [G5] abstention score < dev-tuned threshold? --> ABSTAIN (low_retrieval_score), no LLM call
  |
  |-- context builder: ranked chunks in order until the next would exceed 1024 tokens
  |     -> passages [C1] [C2] ... (heading path, section ids, text) inside delimiters
  v
Synthesizer (LLM) --> {"abstain", "answer", "citations"}   abstain --> ABSTAIN (synth_abstain)
  |
  v
deterministic guards: citation_validity + number_grounding
  |   G1/G2: recorded only        G3: failure --> ABSTAIN (guard_fail)
  v
Critic (LLM; sees the question, the cited passages and the answer)       [G4, G5, G6]
  |   "abstain" --> ABSTAIN (critic_abstain)
  |   "accept" and guards passed --> ANSWER
  |   guards failed or "revise":
  v
revision round (once): context extended with the next chunks up to 2048 tokens; the
synthesizer gets the guard reasons, the critic feedback and its previous answer
  |
  v
guards + critic again --> ANSWER, or ABSTAIN (synth_abstain | guard_fail | critic_abstain)
```

* At most 2 synthesizer rounds and 2 critic rounds per question. Each round may use one
  JSON-repair retry (the reply and a short repair instruction are sent back once); repairs are
  counted in the LLM-call totals. An unreadable synthesizer reply after the repair counts as
  `synth_abstain`; an unreadable critic reply is recorded and treated as no objection (the
  guards have already passed at that point).
* When round 1 fails the guards, the critic is skipped and the revision uses the guard reasons
  (`loop.critic_on_guard_fail: false`, saves one LLM call). When the critic still asks for a
  revision in the final round, the system abstains (`loop.on_final_revise: abstain`).
* Every step is recorded (name, round, status, details, timing, LLM calls with token usage and
  latency). Cached LLM responses report the latency measured on the original call, and the
  retrieval step reports the retriever's cache-adjusted total, so end-to-end latency does not
  depend on warm caches.
* A warning is recorded when a prompt comes close to the model's context window
  (`llm_context_window`, 4096 for qwen2.5:3b under Ollama here). Round-1 prompts are about 1.2k
  tokens and revision prompts about 2.5k, so they fit; if you raise the budgets, also raise
  `OLLAMA_CONTEXT_LENGTH`.

**Untrusted document text.** Passages go into the user message only, each wrapped in
`<<<PASSAGE Cn>>> ... <<<END Cn>>>`. Delimiter look-alikes inside document text (`<<<`, `>>>`)
are neutralised so a passage cannot close its block, and the fixed system prompts say that
passages are data and that instructions inside them must be ignored. System prompts never
contain document text.

**Plain-language mode** (UI and `ask.py --plain` only) adds one rule to the synthesizer's system
prompt: simpler wording, still one citation per sentence. Evaluation always uses the default
mode.

Code lives in `src/ragsys/agents/`:

| module | content |
|---|---|
| `config.py` | `GenerationSettings` (`configs/generation.yaml`): budgets, loop, systems, abstention, NLI, UI |
| `context.py` | `ContextBuilder` (packing, extension), `Passage` (delimited block), `PackedContext` |
| `synthesizer.py` / `critic.py` | prompts, strict-JSON parsing and validation |
| `llmcall.py` | JSON reply with one repair retry; per-call latency (stored latency on cache hits) and tokens |
| `guards.py` | `citation_validity`, `number_grounding` |
| `text.py` | citation tags, sentence splitting (legal abbreviations, citations after the full stop) |
| `orchestrator.py` | `Orchestrator`, `PipelineSpec`, `AnswerResult` (trace, cost, record) |
| `abstention.py` | threshold tuning and the threshold file |
| `derive.py` | derived systems G3 and G5 from base records |
| `engine.py` | builds orchestrators per system on shared retrieval resources |
| `fake.py` | deterministic offline LLM stand-in (`--fake-llm`, tests) |

### Guards (no LLM)

* **citation_validity**: the answer is not empty, every sentence ends with at least one
  citation, and every cited label (inline tags and the `citations` field) exists in the context.
* **number_grounding**: every number in the answer (digits or English words, including lakh /
  crore amounts and durations such as "forty-five days") occurs, compared by value, in at least
  one cited passage (its heading path, section ids or text; page numbers do not count). "30 days"
  matches "thirty days" and "50 lakh" matches "Rs. 50,00,000". A bare "one" (as in "any one of")
  is not treated as a number.

A failed guard marks the answer `unsupported` with its reasons.

### Abstention

The abstention score is the top-1 dense cosine of the first-stage retriever (the dense
component of hybrid rankings), or the top-1 rerank score when reranking is on.
`scripts/tune_abstention.py` computes it for every dev question with the retriever of
`abstention.tuned_system` (G5, i.e. the retriever of G4). It tries every distinct dev score as the
threshold (abstain when score < threshold) and keeps the one with the highest abstention F1
(positive class = unanswerable); ties go to the lower threshold. Because the candidates are
observed scores, "abstain on everything" is never chosen. The result goes to
`results/tables/abstention_threshold.json` (threshold, dev precision / recall / F1, the whole
curve, per-question dev scores). Questions below the threshold are abstained before any LLM
call.

### Systems (`configs/generation.yaml`)

| system | retrieval | guards | critic loop | threshold | how it is produced |
|---|---|---|---|---|---|
| G1 naive | `fixed_512\|dense\|bge-base` | recorded only | - | - | run |
| G2 structural | `structural\|dense\|bge-base` (best dev retrieval config) | recorded only | - | - | run |
| G3 | as G2 | failure -> abstain | - | - | derived from G2's records, no LLM calls |
| G4 | as G2 | yes | yes | - | run (round 1 reuses G2's synthesizer call from the LLM cache) |
| G5 | as G2 | yes | yes | yes | derived from G4's records + threshold; questions below it lose their LLM cost |
| G6 | `fixed_256\|hybrid_linear\|bge-base\|a0.5\|rerank-minilm` (best dev recall@5) | yes | yes | - | run (reranking comparison) |

Derived systems can also run live (UI, `ask.py`): G3 as G2 with the guards applied, G5 as G4
with the threshold check first. Without a threshold file, G5 runs like G4 and every result says
so.

### Answer evaluation

`src/ragsys/eval/answers.py` is independent of the generator LLM. The NLI model is
`cross-encoder/nli-deberta-v3-base` on CPU (about 740 MB, downloaded on first use; if it cannot
be loaded, `cross-encoder/nli-distilroberta-base` is used and the reason is written to the
table metadata). P(entailment) is cached in `data/processed/nli_cache/nli_scores.sqlite`. Long
premises are split into 400-token windows (stride 300) and the maximum over windows is used.

| metric | definition | over |
|---|---|---|
| faithfulness | supported sentences / sentences; a sentence (citation tags removed) is supported when the max over its cited passages of P(entailment) >= 0.5; a sentence without a valid citation is unsupported | answered items (mean per answer; `faithfulness_micro` pools sentences) |
| unsupported-claim rate | 1 - faithfulness | answered items |
| answers with an unsupported sentence | share of answers with at least one unsupported sentence | answered items |
| correct | P(entailment \| premise = answer, hypothesis = gold answer) >= 0.5, or, for `numeric` questions, every number of the gold answer occurs in the answer (by value) | answerable items that were answered (`correct_rate_answered`); all answerable items (`correct_rate_answerable`) |
| correct and faithful | correct and every sentence supported | all answerable items |
| citation precision | cited passages relevant to a gold node (retrieval relevance definition) / cited labels; labels not in the context count as not relevant; no citation gives 0 | answerable items that were answered |
| citation recall | gold nodes covered by the cited passages / gold nodes | answerable items that were answered |
| correct refusal | abstained | unanswerable items |
| false abstention | abstained | answerable items |
| abstention precision / recall / F1 | positive class = unanswerable | all items |
| overall accuracy | (answerable and answered correctly) + (unanswerable and abstained) | all items |
| cost | LLM calls, prompt / completion tokens (means), end-to-end latency p50 / p95 | all items |

Abstentions are also counted by reason. A paired bootstrap (1000 seeded resamples) compares
every system with G1 on overall accuracy (all questions), faithfulness (questions both systems
answered) and correct refusal (unanswerable questions). Tables are broken down by question type.

### Protocol

* Dev and test are the retrieval split (hash of the question id). Everything that is chosen
  (the systems' retrieval configs, the prompts, the loop settings, the threshold) is chosen on
  dev.
* `scripts/eval_generation.py` is the only generation code path that reads test questions:
  * it needs a grant from `grant_test_access()`, which is issued to `scripts/eval_retrieval.py`
    and `scripts/eval_generation.py` only (a test checks the source tree);
  * every start is appended to `results/test_access_log.json`;
  * it refuses to start without `abstention_threshold.json`;
  * it refuses a second complete run unless `--allow-repeat` is given (and logged).

  `run_generation.py`, `derive_systems.py` and `tune_abstention.py` only accept dev.
* Per-question records are reused only when the question (id, text, type, answerability, gold
  nodes and answer) and the system fingerprint (system definition, retrieval config, budgets,
  loop settings, prompt texts, LLM model and parameters) match; otherwise they are recomputed.

### Commands

```powershell
ollama pull qwen2.5:3b                  # once; keep Ollama running

# one question, with citations, step trace and timings
python scripts/ask.py "Within how many days must a complaint be decided if no testing of goods is needed?" --system G4
python scripts/ask.py "Who can file a complaint?" --system G5 --plain

# local web UI on http://127.0.0.1:8000 (loopback only; Ctrl+C to stop)
python scripts/serve.py

# 1. dev runs (resumable; one record per question as it is answered)
python scripts/run_generation.py --split dev --systems G1 G2 G4 G6
# 2. abstention threshold on dev (retrieval only, about a minute)
python scripts/tune_abstention.py
# 3. derived systems G3 / G5 on dev and the complete dev tables
python scripts/derive_systems.py --split dev
# 4. once, at the end: test evaluation of all six systems
python scripts/eval_generation.py --split test --systems G1 G2 G3 G4 G5 G6
# optional: LLM-judge cross-check of the NLI verdicts with another model family
python scripts/judge_sample.py --model llama3.2:3b --results results/generation/dev/G4
```

On a 2-core laptop CPU an LLM call takes roughly 10-45 s, so a synthesizer-only system needs
about half a minute to a minute per question and a critic system about one to two minutes.
Expect a few hours for the dev runs and several hours for the test evaluation. Both resume
after an interruption: rerun the same command, and finished LLM calls come from the cache.
Memory: one bge-base encoder, the MiniLM reranker and later the NLI model (the retrieval models
are released before the NLI model is loaded); Ollama runs in its own process.

Outputs:

| path | content |
|---|---|
| `results/generation/<split>/<system>/per_question.jsonl` | answer, outcome and reason, sentences and citations, context passages (ids, sections, pages), retrieval scores, every round (synthesizer, guards, critic), step trace, cost |
| `results/generation/<split>/<system>/eval.jsonl` | per-question answer metrics, with the NLI score of every sentence |
| `results/tables/generation_<split>.csv` | one row per system: all metrics above, abstentions by reason |
| `results/tables/generation_<split>_by_type.csv`, `_bootstrap.csv`, `.meta.json` | breakdown by question type, paired bootstrap vs G1, settings, NLI model, threshold, run info |
| `results/tables/abstention_threshold.json` | dev-tuned threshold with its curve |

### Web UI

`scripts/serve.py` starts a FastAPI app (`src/ragsys/app/`) bound to 127.0.0.1, port 8000 by
default, serving every collection:

* a collection selector (shown first: `cs_ai`, also "Indian Consumer Law"); per collection the
  heading, the documents with titles, authors, licence and source links (from its
  `SOURCES.csv`), five example questions from its dev split as clickable chips, its disclaimer
  (consumer_law: "Not legal advice"; cs_ai: "Answers come only from the listed open
  textbooks") and its systems (consumer_law: G2, G4, G5, G6, default G5; cs_ai: G2, G4, default
  G4). A collection's models are loaded on its first question; switching releases the other
  collection's encoders. A collection that is not built yet is listed as such;
* ask box, system selector and a plain-language switch;
* the answer with a superscript citation chip after each sentence; a chip opens a side panel
  with the passage's heading path, section id, pages and text, with the cited sentence's key
  terms highlighted;
* when the system abstains: "Not found in these documents" and the three closest sections,
  each opening its passage;
* a collapsible step trace (retrieval, context, threshold, synthesis, guards, critic, revision)
  with timings and LLM usage;
* thumbs up / down stored in the collection's feedback file (`data/feedback.jsonl`,
  `collections/cs_ai/feedback.jsonl`; git-ignored, stays on this machine);
* footer: the collection's disclaimer and documents with their versions (file hash and pages).

Endpoints: `GET /`, `GET /api/collections`, `GET /api/meta?collection=`, `POST /ask`
(`{"question", "system", "plain", "collection"}`), `GET /passage?chunk_id=...&collection=`,
`POST /feedback`. The page loads no external resources and has no inline script (strict
Content-Security-Policy), and all document and model text is inserted as text. Requests are
answered one at a time, and only loopback host names are accepted. The page is keyboard
accessible (labelled controls, visible focus, Esc closes the passage panel) and works at 360 px
width.

## Tests and CI

```powershell
pytest -q              # synthetic fixtures, fake models, FakeLLM, fake NLI; no network (corpus tests deselected)
pytest -q -m corpus    # integration test on the real PDFs in data/raw (skipped if absent)
ruff check . ; ruff format --check . ; mypy --strict src
```

Unit tests generate small PDFs with PyMuPDF (an Act-like document, a Rules-like notification, a
non-legal report and a textbook with an outline, numbered and unnumbered headings, a code block,
an equation line, running headers, an Exercises section, references and an index). Collection
tests check the consumer-law paths, prompts, fingerprints and result tables against the state
before collections existed, the isolation of caches and test-access logs between collections,
the runner's skip/resume logic with fake steps and collection switching in the web UI.
`pytest -q -m corpus` also re-ingests and re-chunks the consumer-law PDFs and compares the
outputs byte for byte with `data/processed/`. GitHub Actions runs lint, type checks and tests on `ubuntu-latest` and
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
* Answering uses a small local generator (qwen2.5:3b, 4-bit, CPU). It follows the citation
  format well but can cite a passage that does not answer the question, or copy a related
  provision. The critic is the same small model and is lenient about relevance, and the guards
  only check citation form and numbers, not the meaning of a sentence.
* Faithfulness and correctness are NLI-based, an automatic proxy rather than human judgement: a
  correct paraphrase can score below 0.5, and gold answers are short, so the correctness test is
  approximate. `judge_sample.py` offers a cross-check with another model family.
* Most evaluation questions were generated and filtered automatically (see above) and the dev
  split is small, so the threshold and the system comparisons on dev are noisy; use the
  bootstrap intervals.
* Sentence splitting is rule-based (legal abbreviations, citations after the full stop); unusual
  formatting can merge or split sentences.

* Textbooks: heading detection relies on font size, a heading-only font family or numbering;
  bold run-in headings set at body size are not detected. Equations are kept as extracted
  text (fractions and matrices lose their layout), figures are dropped (their captions are
  kept), and code that is not set in a monospace font (ODS pseudocode) is recognised only by
  its indentation. Node page numbers are PDF pages, not the printed page numbers.
* Pro Git refers to other sections by title rather than by number, so its cross-references are
  not resolved; d2l and ODS references into chapters that were not ingested are `external`.
* The `cs_ai` licences of d2l and ODS are not stated (with a version) in the PDFs themselves;
  they come from the publishers' pages, as recorded in SOURCES.csv.

## License

MIT, see [LICENSE](LICENSE).
