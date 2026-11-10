# Agnostic Agentic-RAG: Hybrid Vector-Semantic Document Intelligence for Niche Low-Resource Corpora

This repository contains two parts of the project.

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
  * an OpenAI-compatible LLM client, used for drafting evaluation questions that a person then
    reviews.

Answer generation, agents and UI are later stages and are not part of this code yet.

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
HuggingFace cache. Building the retrieval indexes also downloads:

* `BAAI/bge-base-en-v1.5` (about 440 MB);
* `BAAI/bge-large-en-v1.5` (about 1.3 GB);
* `cross-encoder/ms-marco-MiniLM-L-6-v2` (about 90 MB).

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
| `origin` | string, optional | `human` (default) or `llm_draft_human_reviewed` |

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
* **Test split guard.** Only `scripts/eval_retrieval.py` can read test questions:
  * the harness needs a grant, and `grant_test_access()` only issues it to that script;
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
  and p95 in ms. It is measured after a warm-up query, with no caching across configs.
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
SDK with a `base_url`), configured in `configs/llm.yaml`. It is used only to draft evaluation
questions.

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
  parameters), so reruns are free and reproducible.

Tests use `FakeLLM` and a fake HTTP transport; no test calls a network endpoint.

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

Results can be broken down by origin (`retrieval_*_by_origin.csv`).

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
