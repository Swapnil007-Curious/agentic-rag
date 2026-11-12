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
* **Assistant:**
  * three answer modes: Grounded (the pipeline above), Hybrid (grounded first, then a clearly
    labelled general-knowledge answer when the documents do not answer) and Chat & Code (a
    conversation with optional document passages and a coding assistant);
  * two generator backends: a local model (Ollama) and a hosted OpenAI-compatible model, with
    rate limiting and no silent fallback during evaluations;
  * five collections, two of them evaluated; app-only documents never touch a frozen
    evaluation corpus; a generator-scale ablation (G2H / G4H) on the evaluated collections.

"Agnostic" means the pipeline is domain-independent: document structure comes from a pluggable
`StructureParser`. The case-study corpus uses `IndianStatuteParser`; any other PDF can be
ingested with `GenericParser` (font-size / numbering based headings) by adding it to
`configs/ingest.yaml` with `parser: generic`.

The system serves several **collections** (document sets), each with its own configs, data,
caches and (when evaluated) evaluation set and results: `consumer_law` (the case study above, at
its original locations), `cs_ai` (open-licensed computer science textbooks, plus two Python books
in the app), and three collections served in the app only: `indian_law`, `human_biology` and
`physics_chemistry`. See [Collections](#collections) and
[Research page: modes, backends and app collections](#research-page-modes-backends-and-app-collections).
The web app is a chat assistant that grounds its answers in these collections automatically
(see [Assistant](#assistant) and [Public demo](#public-demo)).

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

| collection | configs | raw / processed / eval / results | evaluated |
|---|---|---|---|
| `consumer_law` (default) | `configs/{ingest,chunking,retrieval,generation}.yaml` | `data/raw`, `data/processed`, `data/eval`, `results` | yes |
| `cs_ai` | `configs/collections/cs_ai/*.yaml` (app index: `configs/collections/cs_ai/app/`) | `collections/cs_ai/{raw,processed,eval,results}` (app index: `processed_app`) | yes |
| `indian_law` | `configs/collections/indian_law/*.yaml` | `collections/indian_law/{raw,processed}` | no |
| `human_biology` | `configs/collections/human_biology/*.yaml` | `collections/human_biology/{raw,processed}` | no |
| `physics_chemistry` | `configs/collections/physics_chemistry/*.yaml` | `collections/physics_chemistry/{raw,processed}` | no |

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
2. Copy the configs of a similar collection to `configs/collections/<name>/`:
   `human_biology/` for textbooks (OpenStax-style numbered headings, whole books selected with
   `outline_chapters`), `indian_law/` for statutes, `cs_ai/` for an evaluated collection with
   selected chapters. Edit the documents (file, doc_id, title, authors, edition, source URL,
   licence and where it was read, chapters to keep) and every `collections/<old>/...` path.
3. Add an entry to `configs/collections.yaml` (roots, UI texts including the safety note, the
   chat domain and six example questions, profile, `evaluated`, `eval_docs` / `app_docs`,
   pipeline settings) and add it to `ui_order`. A new domain gets its own profile in
   `src/ragsys/profiles.py` (prompt wording; for evaluated collections also the question-type
   mix and out-of-scope topics).
4. App only: `python scripts/run_app_collections.py --no-ablation` builds its index (ingest ->
   chunk -> bge-small index). Evaluated: `python scripts/run_collection_pipeline.py --collection
   <name>`. Then `pytest -q` checks that the configs agree with the roots and that the
   documents listed in `eval_docs` / `app_docs` match the ingest configs.

## Assistant

`python scripts/serve.py` opens a chat assistant at http://127.0.0.1:8000 (code:
`src/ragsys/assistant/`, page: `src/ragsys/app/static/chat.html`). The grounded research page
with its systems and modes moved to http://127.0.0.1:8000/research; the research pipelines,
evaluations and results are not changed by the assistant.

### Chat-first behaviour

* **One LLM call per message, streamed token by token** (Server-Sent Events from
  `POST /api/chat`; the hosted backend streams with `stream=True`, Ollama through its
  OpenAI-compatible streaming). Stop, Regenerate, Copy and edit-and-resend of the last message
  are in the page. Nothing is cached on disk and message contents are never logged.
* **Automatic grounding without an extra LLM call.** The latest message (joined with the
  previous user message when it is a short follow-up such as "and why?") is embedded with
  bge-small and searched in the selected collection's app index and in the session's uploaded
  PDFs. When the top-1 cosine of a source reaches its gate threshold, the best passages of that
  source are added to the message as labelled, delimited data ([C1], [C2], ...; 2048 index
  tokens for the hosted backend, 1024 for the local one) and the model is asked to use and cite
  them when relevant, or to answer from general knowledge and say so. Below the gate the
  message is plain chat. Thresholds live in `configs/assistant.yaml`: for the two evaluated
  collections the 5th percentile of the top-1 dense scores of their answerable DEV questions
  (structural chunks, bge-small; `tests/test_assistant_gate.py` recomputes them from the DEV
  outputs), 0.62 for the other collections and for uploads.
* **Source line.** Under every reply: "Answered with documents (N sources)" with clickable
  citation chips that open the passage, or "General knowledge — no matching documents". It is
  decided from the citations the reply actually contains; labels that name no passage given to
  the model are removed.
* **Verify** (on demand, per grounded reply): the grounded pipeline's deterministic guards
  (citation validity, number grounding) and the critic check the reply against its cited
  passages, and every sentence is marked supported or unsupported with the reason.
* **Grounded only** (switch): the grounded pipeline (synthesizer, guards, critic, one revision
  round) answers strictly from the documents or says "Not found in these documents" with the
  closest sections.
* **Conversation context:** the last 10 turns within the backend's history budget; when they do
  not fit, the oldest messages are summarised by one short LLM call and the summary is reused
  for the following turns. Conversations are stored in the browser's localStorage only (the page
  works when storage is blocked).
* **System prompt:** a general assistant for students and developers (concepts; code with stated
  assumptions, runnable examples and complexity notes; step-by-step math and science in LaTeX),
  with the collection's notice (law collections: "Not legal advice"; human biology:
  "Educational, not medical advice; for symptoms or emergencies contact a doctor or local
  emergency services") and the provider's safety policies.
* **Rendering:** safe Markdown built from text nodes (HTML in a reply is shown as text, only
  http(s) and mailto links), code blocks with syntax highlighting and copy buttons, LaTeX with
  KaTeX. highlight.js and KaTeX are vendored under `src/ragsys/app/static/vendor/` with their
  licence files; the page loads nothing from other sites and keeps the strict
  Content-Security-Policy. Light and dark themes follow the system setting; the page is
  keyboard accessible and works at 360 px.

### Uploading your own PDFs

Up to 3 PDFs per browser session (at most 20 MB and 300 pages each). They are parsed in memory
with the ingest pipeline's line extraction and cleaner, structured by the GenericParser, cut into
structural chunks and embedded with bge-small, with a progress indicator; they are then searched
next to the selected collection (also with "None (general chat)"). Nothing is written to disk
or logs; a session's documents are deleted when the page closes or after 2 hours without
activity. Non-PDF, encrypted, oversized and scanned (text-less) files are refused with a message.

### Privacy: hosted and local

| backend | what leaves this computer |
|---|---|
| local (Ollama, qwen2.5:3b) | nothing: messages, passages and uploads stay on this computer |
| hosted (Groq) | each message, the recent conversation (or its summary) and the passages added to it are sent to the provider; uploads themselves are not sent, only their matching passages |

The page shows the active backend, its model and this note. The assistant never switches to
another backend on its own: when the hosted API is rate-limited the reply says "The model is
busy right now. Please try again in a minute."

### Speed

`python scripts/bench_chat.py` runs 10 prompts over the collections through the assistant and
prints the time to first token, the full answer time, a projection for a 300-token reply and the
gate time (embedding + search) with p50 and p95 (`--fake-llm` runs offline).

## Public demo

A free public Hugging Face Space runs the assistant with the hosted backend only (no local
model): **https://huggingface.co/spaces/USERNAME/agentic-rag-assistant** (placeholder until it
is deployed). In public mode the research page is off and the API key is protected by limits
from `configs/assistant.yaml`:

* 20 messages per 10 minutes and 200 per day per visitor (Verify and Grounded-only count as
  messages), 10 uploads per hour;
* messages of at most 4000 characters, replies of at most 1500 tokens;
* a global daily request cap with a friendly "busy, try later" message;
* no logging of message contents.

The server binds to 127.0.0.1 unless the `HOST` environment variable is set (the Space's image
sets `HOST=0.0.0.0`, `PORT=7860`); see `src/ragsys/assistant/hosting.py` for `ALLOWED_HOSTS`,
`FRAME_ANCESTORS`, `TRUST_PROXY` and `PROXY_HOPS`.

### Deployment

```powershell
# 1. assemble deploy/hf_space/build/ (code, configs, the five app indexes, SOURCES.csv files,
#    vendored assets, Dockerfile, Space README); prints a size summary and checks that it
#    contains no PDFs, no .env files and no LLM_API_KEY / HF_TOKEN values
python scripts/build_space.py

# 2. create or update <username>/agentic-rag-assistant (username from the token)
$env:HF_TOKEN = "hf_..."              # write token; read from the environment only
python scripts/deploy_space.py --dry-run
python scripts/deploy_space.py --set-llm-secret   # also stores this shell's LLM_API_KEY as the Space secret

# without --set-llm-secret: Space -> Settings -> Variables and secrets -> New secret LLM_API_KEY
```

The Space builds the Docker image itself (python:3.11-slim, CPU torch from the PyTorch CPU index,
bge-small downloaded at build time, non-root user, port 7860). Docker is not needed locally.
Several collections are licensed for non-commercial use, so the demo is non-commercial; the app's
"About & sources" panel lists every document's licence as recorded in its `SOURCES.csv`.

## Research page: modes, backends and app collections

The research page (`/research`) keeps the original grounded interface with the evaluated systems.

### Answer modes

| mode | what it does | grounding |
|---|---|---|
| Grounded | the pipeline described under [Answering](#answering) (synthesizer + guards + critic, G4 by default) | only from the passages, one citation per sentence; otherwise "Not found in these documents" with the three closest sections |
| Hybrid | grounded first; only when it abstains, a short general-knowledge answer from the same backend | the general answer is shown separately under the banner "General knowledge — not from these documents. May be wrong; verify." with the closest passages as "Related reading"; it is never merged into a grounded answer and carries no citations |
| Chat & Code | a multi-turn conversation in Markdown with fenced, copyable code blocks | none by default; with "Use documents", passages retrieved for the latest message are added (delimited as untrusted data) and the reply cites them as [C1], [C2], ...; unknown labels are reported |

Chat keeps the last 10 turns (a turn is a user message and its reply) and then drops the oldest
until the history fits the backend's `chat_history_tokens` (estimated at four characters per
token); the latest message is always sent. The coding assistant prompt (explain, write, review
and debug code; state assumptions; prefer the standard library; complexity notes) is used for
`cs_ai` and whenever the "Code" switch is on. **Code is never executed.** System prompts carry
the collection's safety note (law collections: "Not legal advice"; `human_biology`:
"Educational information, not medical advice; for symptoms or emergencies contact a doctor or
local emergency services") and ask the model to refuse clearly harmful requests according to the
provider's usage policies; there is no custom jailbreak handling.

Citation tags written with fullwidth brackets ("【C3】", a habit of some hosted models) are read
as [C3] in chat replies and in grounded answers; no answer of the local model contains them, so
existing results are unaffected.

Code: `src/ragsys/agents/modes.py` (modes, history capping), `src/ragsys/agents/chat_prompts.py`
(system prompts), `src/ragsys/app/markdown.py` (the safe Markdown subset).

### Generator backends and privacy

`configs/llm.yaml` defines named backends; every script that calls a generator takes
`--backend` (`ask.py`, `serve.py`, the generation and question-drafting scripts) and the web UI
has a backend selector:

| backend | model | where the text goes |
|---|---|---|
| `local` (default) | `qwen2.5:3b` through Ollama on this computer | questions, passages and answers stay on this computer |
| `hosted` | an OpenAI-compatible provider (Groq); the model id, the date it was chosen and how it was chosen are recorded in `configs/llm.yaml` | **questions and the retrieved passages (and, in chat, the conversation) are sent to the API provider** |

* The API key is read from the environment variable `LLM_API_KEY` only (PowerShell:
  `$env:LLM_API_KEY = "..."`). It is sent to the hosted endpoint and nowhere else: it is not
  sent to the local server, logged, cached, written to reports or shown in the UI; a test checks
  logs, cache files and reprs.
* Rate limits: a client-side token bucket (`requests_per_minute`, `tokens_per_minute`, shared by
  all clients of the same endpoint and model), the provider's `Retry-After` on 429 responses and
  exponential backoff; a requested wait longer than `max_retry_wait_s` (an exhausted daily
  quota), or a rate limit that outlasts every retry, stops the run with a message (exit code 75)
  instead of sleeping for hours; `eval_generation.py --max-retry-wait SECONDS` raises that limit
  for unattended runs. The provider's daily quota (`tokens_per_day`, `requests_per_day`) is
  recorded for time estimates. Resumable runs continue from the response cache when started
  again. A hosted call that fails in the web UI is reported as an error naming the backend.
* Responses are cached on disk by (backend, base URL, model, messages, parameters); the local
  backend's keys are the ones used before backends existed, so its caches stay valid.
* Context budgets: the hosted backend uses 2048 tokens for the first round and 3072 for the
  revision round; the local backend keeps 1024 / 2048 (its 4096-token context window).
* No silent fallback: evaluations check every backend they need before opening the test split
  and stop when one is unavailable; a system's backend is part of its definition (`--backend`
  only checks it). The web UI picks hosted as its default when the key is set and the endpoint
  answers, else local, and names the model under every answer; a request for an unavailable
  backend gets an error, not another model.

### Frozen evaluation corpora and app-only documents

`configs/collections.yaml` lists, per evaluated collection, `eval_docs` (the exact documents of
every existing evaluation, frozen) and `app_docs` (what ask.py and the web UI serve). When they
differ, the collection has a second index:

| index | used by | cs_ai documents | location |
|---|---|---|---|
| eval (frozen) | every evaluation script | d2l, progit, ods | `collections/cs_ai/processed` |
| app | ask.py, the web UI | + the Python Tutorial (whole tutorial) and Think Python 2e (whole book without front/back matter) | `collections/cs_ai/processed_app` |

The app index has its own configs (`configs/collections/cs_ai/app/`; its first three document
entries are identical to the evaluation config, which a test checks) and its retrieval config
says `index: app`. Every evaluation script (`run_retrieval_grid`, `eval_retrieval`,
`select_configs`, `build_auto_eval`, `spot_check`, `tune_abstention`, `run_generation`,
`derive_systems`, `eval_generation`, `judge_sample`) refuses an app index and the collections that
are not evaluated, and the retrieval and generation harnesses refuse it again at library level.
Until the app index is built, ask.py and the UI serve the evaluation index and say so. The app
ingest writes its provenance to `collections/cs_ai/raw/SOURCES_app.csv` (`paths.sources_file`),
so the evaluation corpus's `SOURCES.csv` is never rewritten; the UI lists the app file while it
serves the app index. `consumer_law` serves its evaluation documents (`app_docs = eval_docs`).

`tests/test_frozen_results.py` proves that the existing results are untouched: every result
table, per-question / per-query file and test-access log of `consumer_law` and `cs_ai` is
recorded in `tests/data/frozen_results.json` and must be byte-identical (line endings
normalised); the generator-scale ablation may only append rows to the `generation_test` tables
and entries to the test-access log (their recorded bytes / entries must stay a prefix), and
regenerating `SUMMARY.md` must keep every existing section. Re-ingesting and re-chunking both
corpora with the current code reproduces their processed files byte for byte.

### App-only collections (not evaluated)

These collections are labelled "Not evaluated" in the UI: they have no evaluation set, no
retrieval grid and no generation evaluation. Each uses the structural chunker and bge-small
embeddings only (the fastest encoder on this CPU, and the best dev retriever of `cs_ai`); before
the index is built, retrieval falls back to BM25 and says so. Example questions are written from
real section headings and shown as chips. Licences, editions, source URLs, SHA-256 and pages are
in each collection's `SOURCES.csv`, shown in the UI footer.

| collection | documents | parser | licence |
|---|---|---|---|
| `indian_law` (Indian Law, Constitution & Tax) | Constitution of India (as on 10 September 2026); Income-tax Act, 2025 (as amended by the Finance Act, 2026); CGST Act, 2017; Bharatiya Nyaya Sanhita, 2023 | `constitution` (Constitution), `statute` (the Acts) | Acts of Parliament may be reproduced (Copyright Act, 1957, s. 52(1)(q)); the PDFs state no licence |
| `human_biology` | OpenStax Anatomy and Physiology 2e (whole book without front matter, end-of-chapter glossaries and review material, references and index) | generic, textbook mode | CC BY-NC-SA 4.0 (read from the PDF) |
| `physics_chemistry` | OpenStax College Physics 2e, University Physics Volume 3, Chemistry 2e, Organic Chemistry (same exclusions; appendices and answer keys left out) | generic, textbook mode | CC BY-NC-SA 4.0 (read from the PDFs) |
| `cs_ai` app documents | Python Tutorial (release 3.13.0a0, `python_tutorial/tutorial.pdf` of the Python documentation PDF set); Think Python 2e (Allen B. Downey) | generic, textbook mode | PSF License Agreement (documentation; code examples also Zero-Clause BSD); CC BY-NC 3.0 Unported (read from the PDF) |

Statute parsing for the Constitution (`src/ragsys/structure/constitution.py`): Parts with their
titles, Chapters inside Parts, Articles including lettered ones (21A, 243ZH; the printed
"243Z-I" is stored as 243ZI), omitted articles (kept with their "Omitted by ..." note, whichever
side of the number the amendment bracket is printed), clauses, sub-clauses, provisos,
Explanations, group headings ("Right to Equality") and the twelve Schedules; the appendices are
back matter, and the editorial brackets of substituted headings are removed. Node IDs:
`coi:part3`, `coi:part5-ch1`, `coi:a21A`, `coi:a21A(1)`, `coi:a19(1)(a)`, `coi:sch7`,
`coi:preamble`. The Income-tax Act print sets each section heading on its own bold line above
"1. (1) ..."; `section_heading_above: true` folds it in (also when the heading carries an
amendment marker, "18[Deduction in respect of ...]"). Its amendment footnotes ("7. Sub. by Act
No. 4 of 2026, w.e.f. ...") are set in the body font, so `footnote_start_pattern` finds them: per
page, the first matching line below a gap of two line heights starts the footnote area. Textbooks are selected chapter by chapter
from the PDF outline (`outline_chapters: '^Chapter\s+\d+\b'`); end-of-chapter matter is dropped
from its first heading to the next chapter (`drop_to_chapter_end`: "Key Terms", "Glossary",
"Chapter Review"), chapter-opening lists of section headings are dropped
(`drop_outline_headings`) and wrapped chapter titles are joined from the outline. With
`outline_chapters`, the outline names every chapter: a box heading set in the chapter style
(Organic Chemistry's "CHEMISTRY MATTERS") stays a section of its chapter, and a Sphinx chapter
opening ("CHAPTER" / "ONE" / "WHETTING YOUR APPETITE", the Python Tutorial) becomes chapter 1
with its outline title. Licence sentences hyphenated across a line break ("Cre-/ative Commons")
are read. All these options are off by default, so the existing collections ingest exactly as
before. The ingest report lists, per document, the pages kept, sections/articles, numbering gaps
and warnings.

### Generator-scale ablation (evaluated collections, TEST split, no tuning)

| system | definition |
|---|---|
| G2H | G2 with the hosted backend: same retrieval config, hosted context budgets |
| G4H | G4 with the hosted backend: same retrieval config, guards + critic, hosted context budgets |

The hosted model was chosen before any result was seen (`configs/llm.yaml`: model id, date and
rationale). `eval_generation.py --split test --collection <consumer_law|cs_ai> --systems G2H G4H`
needs the standard test evaluation to be complete (its G1, G2 and G4 outputs are read, never
rewritten), logs the test access, uses the same NLI metrics and guards, and appends rows to
`generation_test.csv`, `_by_type.csv` and `_bootstrap.csv` (existing rows untouched; the
`label` column names the backend, model and selection date). The paired bootstrap compares each
ablation system with G1, with G2 (same retrieval, local generator) and with the system it mirrors
(G4 for G4H). `generation_test_ablation.meta.json` records the backend, the context budgets, the
NLI model, tokens and API latency (cached calls report the latency measured when they were first
made), and `SUMMARY.md` gains a "Generator-scale ablation" section.

### App runner

```powershell
$env:LLM_API_KEY = "..."                       # for the ablation steps only
.\scripts\run_app_collections.ps1               # or: python scripts\run_app_collections.py
python scripts\run_app_collections.py --dry-run # plan + time and token estimate only
python scripts\run_app_collections.py --no-ablation   # app indexes only, no API calls
.\scripts\run_app_collections.ps1 -WaitForQuota # wait for the daily API quota (unattended)
```

Steps: ingest -> chunk -> bge-small structural index for `indian_law`, `human_biology` and
`physics_chemistry`; the `cs_ai` app index; the G2H / G4H test evaluations of `consumer_law` and
`cs_ai`; their `SUMMARY.md`. Each step is skipped when complete with unchanged inputs and resumes
from its caches (embedding batches, LLM responses, per-question records). The dry run prints a
time estimate per step and the expected API tokens of the ablation (per-call tokens measured on
the hosted model, the hosted context budgets and the share of revisions of the local G4 run).

The ablation needs more tokens than a free API account's daily token quota, so it spans several
days. Without `--wait-for-quota` an ablation step stops when the day's quota is used up (exit code
75, nothing falls back to the local model) and the same command continues it once the quota has
renewed. With `--wait-for-quota` (`-WaitForQuota`) the run waits instead: each call waits out the
provider's `Retry-After` (up to 6 hours) and a step stopped by the quota is started again every 30
minutes for up to 5 days; the computer must stay on (the runner keeps it awake). The log is
`collections/app_pipeline_log.txt`; the run ends with `ALL DONE` or names the failing step and the
resume command.

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
SDK with a `base_url`); `configs/llm.yaml` defines its named backends
(`src/ragsys/llm/backends.py`, see
[Generator backends and privacy](#generator-backends-and-privacy)). It builds and drafts
evaluation questions and runs the answering agents (with the overrides of
`configs/generation.yaml`: temperature 0, max_tokens 512 for the local backend).

* **Ollama (`local`, default):** install Ollama, run `ollama pull qwen2.5:3b` and keep it
  running. The `base_url` is `http://localhost:11434/v1`; no key is needed or sent.
* **Hosted (`hosted`):** an OpenAI-compatible provider; put the key in the environment variable
  `LLM_API_KEY` (PowerShell: `$env:LLM_API_KEY = "..."`).
  * The key is read from the environment only.
  * It is never logged, printed, cached or written to reports.
  * `.env` and key files are git-ignored.

Features:
* temperature, max_tokens and seed;
* JSON mode, with a tolerant JSON extractor as fallback;
* retries with exponential backoff (at most 5) on rate limits, timeouts, connection errors and
  5xx responses; the provider's `Retry-After` is respected, and a longer wait than
  `max_retry_wait_s` stops the call (`QuotaExceeded`);
* an optional client-side token bucket (requests and tokens per minute) and the rate-limit
  headers of the last response;
* per-call latency and token usage;
* an on-disk response cache in `<processed_dir>/llm_cache/`, keyed by (backend, base_url, model,
  messages, parameters; the `local` backend's keys carry no backend name, as before backends
  existed), so reruns are free and reproducible. A cache hit reports `latency_ms = 0` (no
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

# web app on http://127.0.0.1:8000: the assistant at /, the research page at /research
# (loopback only; Ctrl+C to stop)
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
default. The chat assistant is served at `/` (see [Assistant](#assistant)); the research page
described here is served at `/research` and covers every collection:

* a collection selector in this order: "Computer Science, AI & Coding" (`cs_ai`, shown first),
  "Indian Consumer Law", "Indian Law, Constitution & Tax", "Human Biology", "Physics &
  Chemistry", each with an "Evaluated" badge (linking to its `SUMMARY.md` rendered as a page,
  `/summary?collection=...`) or a "Not evaluated" badge; per collection the heading, the
  documents with titles, authors, licence and source links (from its `SOURCES.csv`), example
  questions as clickable chips (dev-split questions for evaluated collections, six written from
  real section headings for the others), its disclaimer and safety note and its systems. A
  collection's models are loaded on its first question; switching releases the other
  collection's encoders. A collection that is not built yet is listed as such;
* a mode selector (Grounded / Hybrid / Chat & Code), a backend selector with the model and its
  privacy note, and "Answered by <backend> — model <id>" under every answer;
* Chat & Code: a conversation panel with the chat history, "New chat", a "Use documents" switch
  (citations open the passage panel) and a "Code" switch; Markdown is rendered safely (the server
  parses a small Markdown subset into a block tree and the page builds it from text nodes: no
  HTML, no links, `javascript:` targets stay text), code blocks are monospace with a copy button;
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

Endpoints of the research page: `GET /research`, `GET /api/collections`, `GET /api/backends`, `GET /api/meta?collection=`,
`POST /ask` (`{"question", "system", "plain", "collection", "mode": "grounded"|"hybrid",
"backend"}`), `POST /chat` (`{"messages", "collection", "backend", "use_documents", "code"}`),
`GET /passage?chunk_id=...&collection=`, `POST /feedback`, `GET /summary?collection=` and
`GET /api/summary?collection=`. The page loads no external resources and has no inline script
(strict Content-Security-Policy), and all document and model text is inserted as text. Requests
are answered one at a time, and only loopback host names are accepted. The page is keyboard
accessible (labelled controls, a radio group for the mode, visible focus, Ctrl+Enter sends, Esc
closes the passage panel) and works at 360 px width.

```powershell
python scripts/serve.py                     # backend: hosted if LLM_API_KEY is set and reachable, else local
python scripts/serve.py --backend local     # everything stays on this computer
python scripts/ask.py --collection indian_law "What does Article 21 of the Constitution protect?" --mode grounded --backend hosted
python scripts/ask.py --collection cs_ai "Write a Python function that checks whether a string is a palindrome" --mode chat --backend hosted
python scripts/ask.py --collection human_biology "How does the body regulate acid-base balance?" --mode hybrid
```

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
the runner's skip/resume logic with fake steps and collection switching in the web UI. Further
tests cover backend selection and the absence of fallbacks in evaluations, the rate limiter and
Retry-After, the API key never appearing in logs or caches, the three modes (the hybrid banner
path, chat history capping, "Use documents" citations, the coding prompt), the separation and
refusal of app indexes, the frozen results (`tests/data/frozen_results.json`), the Constitution
parser on a synthetic PDF, the Markdown sanitiser and every UI endpoint in every mode; FakeLLM
stands in for both backends.
The assistant's tests (`tests/test_assistant_*.py`, `tests/test_build_space.py`) use a fake
streaming LLM and the hashing encoder: token order and Stop, the gate above and below its
threshold, citation parsing and stripping, Verify, Grounded only, upload limits, rejections and
in-memory cleanup, rate limits, the HOST binding, history capping and the summary trigger, the
Markdown sanitiser of the chat page (run under Node when it is installed) and a Space build
without PDFs or secrets.
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
* App-only collections are not evaluated: their retrieval and answers have no measured quality.
  Statute parsing leaves numbering gaps where a heading is not recognised (reported per document
  by `ingest.py`); amendment notes printed in the body font of the Income-tax Act print stay in
  the text of the provision they follow; tables in the Acts and schedules are kept as plain
  text. OpenStax end-of-chapter material (glossaries, summaries, review questions) is dropped, as
  are two-column layouts that would interleave; equations are kept as extracted text.
* Hybrid and chat answers from general knowledge are not grounded and can be wrong; the hosted
  backend sends questions and passages to the API provider and is limited by its quotas.

## License

MIT, see [LICENSE](LICENSE).
