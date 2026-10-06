# TODO — RAG against the machine

Step-by-step guideline derived from `CLAUDE.md` (roadmap, section 5) and the subject
(`en.subject _rag.pdf`, v2.0). Tick boxes as we go; this file is the single source of
truth for "where are we". It is git-ignored (`*.md` except `README.md`): it is a working
note, not a deliverable.

Legend: `[ ]` not started · `[~]` in progress · `[x]` done and understood.

Every stage ends with a **Concept check** (section 3.4 of `CLAUDE.md`): the stage is not
done until those questions are answered without notes, whatever the code says.

---

## Progress overview

| # | Stage | Status |
|---|-------|--------|
| F | Foundations (vocabulary and mental model) | `[x]` |
| 0 | Project setup | `[x]` |
| 1 | Explore corpus and datasets | `[x]` (concept check pending in "To revisit") |
| 2 | Pydantic data models | `[x]` |
| 3 | Ingestion and chunking | `[x]` (commit pending) |
| 4 | Lexical index and scorer | `[ ]` |
| 5 | `search_dataset` + `evaluate` (recall@k) | `[ ]` |
| 6 | Iteration loop (tune retrieval) | `[ ]` |
| 7 | Generation with Qwen3-0.6B | `[ ]` |
| 8 | CLI hardening | `[ ]` |
| 9 | README | `[ ]` |
| 10 | Bonuses | `[ ]` |

**Last measured recall@5:** docs — · code —
**Last measured indexing time:** 2.2 s (12 554 chunks, max 2000, 2026-09-29)
**Last measured retrieval time (200 questions):** —

---

## Stage F — Foundations (no code, only understanding)

Goal: I can explain the whole pipeline in my own words before writing a single line.
Each box is ticked only after I have explained the notion back and been corrected.

### F.1 Language models
- [ ] F.1.1 What an LLM is: a model that predicts the next token from the previous ones.
  What a **token** is (≈ a word piece), what the **context window** is (the maximum
  number of tokens the model can read at once).
- [x] F.1.2 Why a model is "frozen in time": it only knows its training data. Why
  retraining is not the answer (cost, time, private data).
- [ ] F.1.3 What a **hallucination** is: a fluent answer invented because the model has
  no evidence. Why a small model (0.6B parameters) hallucinates more than a big one.

### F.2 What RAG is
- [x] F.2.1 RAG = Retrieval-Augmented Generation: instead of putting knowledge *inside*
  the model, find the right pages at answer time and paste them into the prompt.
- [ ] F.2.2 The four stages and what each one produces:
  **Index** (files → chunks → searchable structure), **Retrieve** (question → top-k
  chunks), **Augment** (chunks → prompt within the token budget), **Generate**
  (prompt → answer).
- [x] F.2.3 Why retrieval quality dominates: if the right chunk is not retrieved, no
  prompt trick will produce a correct answer. This is why the grade is on recall@k.

### F.3 Chunks and index
- [x] F.3.1 What a **chunk** is: a contiguous slice of a file, identified by
  (`file_path`, `first_character_index`, `last_character_index`). Why we slice files
  instead of indexing whole files.
- [x] F.3.2 What an **index** is: a data structure that answers "which chunks contain
  this word?" in milliseconds (inverted index = word → list of chunks).
- [x] F.3.3 What **lexical** retrieval means: matching on words, not on meaning.
  TF-IDF / BM25 intuition: a chunk scores high if the question's words appear in it
  often (TF) and those words are rare across the corpus (IDF).
- [ ] F.3.4 What lexical retrieval cannot do (synonyms, paraphrase) and what the
  **semantic** bonus adds (embeddings = vectors that encode meaning).

### F.4 The metric and the grader
- [x] F.4.1 **recall@k** on this project: for each question, did one of my top-k results
  land in the *same file* as the ground-truth source with overlap IoU > 0.05?
  Average over the questions.
- [x] F.4.2 **IoU** (intersection over union) on two character ranges, computed by hand
  on one real example from the dataset.
- [ ] F.4.3 Why recall@1 ≤ recall@5 ≤ recall@10, and why "return more" is not a
  strategy (context budget, noise for the generator).
- [x] F.4.4 The three ways to score zero without a bug: wrong `file_path` prefix, a
  chunk longer than 2000 characters, offsets shifted by a text transformation.

### F.5 The deliverable in one sentence
- [x] F.5.1 A Python CLI (`uv run python -m src <command>`) with six commands:
  `index`, `search`, `search_dataset`, `answer`, `answer_dataset`, `evaluate`.
- [x] F.5.2 Two chunkers (Python, Markdown), one lexical scorer (BM25 or TF-IDF), one
  small local generator (Qwen3-0.6B), one evaluator (my own recall@k).

**Concept check**
- Explain RAG to someone who has never heard of it, in 60 seconds.
- Take the LoRA question from the docs dataset and describe, stage by stage, what
  happens between the question and the answer.
- Why is a result in the right file but the wrong span worth zero?

---

## Stage 0 — Project setup

Goal: a clean skeleton where `uv sync` and `make lint` are green from a fresh clone.

- [x] 0.1 Inputs in the mandatory layout (not committed, see `.gitignore`):
  - [x] `data/raw/vllm-0.10.1/` extracted from `vllm-0.10.1.zip` (3 226 files)
  - [x] `data/datasets/AnsweredQuestions/` and `data/datasets/UnansweredQuestions/`
    extracted from `datasets_public.zip`
  - [x] Move the zips, the PDF and `README (1).md` (moulinette README) out of the
    working tree into a `notes/` folder (git-ignored) so the repo root stays clean
- [x] 0.2 `uv init` / write `pyproject.toml`: Python `>=3.10`, project name, `src` package
- [x] 0.3 Declare dependencies: `pydantic`, `fire`, `tqdm` (runtime); `flake8`, `mypy`,
  `pytest` (dev). Generate `uv.lock` and commit it.
- [x] 0.4 Create `src/__init__.py` and `src/__main__.py` with a Fire entry point exposing
  a command class (six commands stubbed, each printing "not implemented")
- [x] 0.5 Makefile: `install`, `run`, `debug` (`python -m pdb -m src`), `clean`, `lint`
  (exact flags from subject V.2), `lint-strict`
- [x] 0.6 `.flake8` config (max-line-length, excludes for `.venv`, `data/`)
- [x] 0.7 `mypy` config in `pyproject.toml` (exclude `data/`, `.venv/`)
- [x] 0.8 Verify from a clean clone: `uv sync && make lint && uv run python -m src`
- [x] 0.9 First commit (2026-09-24)

**Concept check**
- Why does the reviewer only run `uv sync`? What breaks if `uv.lock` is missing?
- What does `--disallow-untyped-defs` force me to do on every function?

---

## Stage 1 — Explore the corpus and the datasets by hand

Goal: I can describe what a question looks like and what a "correct source" looks like.

Known facts (measured on 2026-09-24):
- Corpus: 3 226 files, of which 1 763 `.py` and 178 `.md`. Everything else (yaml, cmake,
  cu, txt, ...) is a decision to make.
- Datasets: 100 docs questions, 99 code questions. Each answered question has exactly
  **one** ground-truth source.
- Ground-truth span sizes (characters): docs min 40 / median 1162 / max 1997;
  code min 12 / median 878 / max 1578. The max is below 2000: the reference itself
  respects the ceiling.
- Docs spans start on a Markdown heading (`### Using API Endpoints`); code spans start
  on a decorator / `def` / a block of constants. This is a strong hint for the chunkers.
- Public datasets share filenames across `AnsweredQuestions/` and `UnansweredQuestions/`.
- `last_character_index` is **exclusive** (Python slice convention): the LoRA span ends
  right before the `\n\n### Using Plugins` that follows.
- Docs: 79/100 sources start on a Markdown heading, 53/100 end right before the next
  heading. 22 sources are under 500 chars.
- Code: only 41/99 sources start on `def` / `class` / decorator; 50 start mid-function
  (an `else:`, an assignment). 13 are under 500 chars, 2 under 100 (93 and 12 chars).
- **IoU trap**: a reference fully inside my chunk gives IoU = ref_len / chunk_len. With
  2000-char chunks, 4 docs + 2 code references are lost by construction (IoU <= 0.05);
  with 1000-char chunks, 1 + 1; with 500, 0 + 1. The 93-char reference inside the
  14 258-char `attn_fwd` kernel is the canonical example. Argument for chunks < 2000.

- [x] 1.1 Open 5 docs questions and 5 code questions. For each, open the ground-truth file
  at the given `[first_character_index:last_character_index]` and read the span.
- [x] 1.2 Plot / tabulate the span-size distribution → drives the chunk-size decision in
  stage 3
- [x] 1.3 Note what the ground-truth spans are aligned on: a function? a class? a heading?
  a paragraph? (drives the chunking boundaries)
- [x] 1.4 Classify the questions: paraphrase vs identifier-quoting (2026-09-25, after
  excluding the word `vLLM` which appears in 97/100 questions): **docs 18/100 IDENT,
  code 77/99 IDENT**. Decision: tokeniser keeps each identifier whole AND its split parts.
- [x] 1.5 Which files to index (2026-09-25). Ground-truth sources: 99 `.py`, 97 `.md`,
  3 `.txt` (all `CMakeLists.txt`, docs dataset). Zero sources under `tests/`,
  `benchmarks/`, `examples/`. `tests/` = 696 py/md files vs 879 in `vllm/`, and reuses
  the same identifiers (ranking noise). **Decision: index `.py`, `.md`, `.txt`; exclude
  `tests/` and `benchmarks/`; keep `examples/` (measure its effect in stage 6).**
  → 1 205 files kept.
- [x] 1.6 Encoding traps (2026-09-25) on the 1 205 kept files: 1 CRLF (a 235-char
  `__init__.py`, not a source), 0 BOM, 0 non-UTF-8. **Decision: read UTF-8 strict, text
  mode, no `errors=`; skip unreadable files with a warning instead of shifting offsets.**
- [x] 1.7 `notes/corpus.md` written (2026-09-25, by Claude at my request — I must be
  able to restate it; the three concept-check questions were skipped → "To revisit").

`notes/explore.py` sub-commands: `show <dataset> <i>`, `classify <dataset>`,
`origin <dataset>...`, `corpus <root>`, `encoding <root>`.

**Session 2026-09-25**: IoU acquired (interval schema on `lora.md`). Stage 0 questions
answered by Claude and moved to "To revisit". 1.4–1.6 done.

**Concept check**
- What is the IoU > 0.05 rule, concretely, on one of the spans I read?
  (mental check: a 400-char reference inside a 2000-char chunk, found or lost?)
- Why does a result in the right file but the wrong span count as zero?
- Why is a docs question "easier" than a code question for a lexical system?

---

## Stage 2 — Pydantic data models

Goal: models round-trip a real dataset JSON without validation errors.

- [x] 2.1 Concept pass: validation at the boundaries (input datasets, output files)
- [x] 2.2 `src/models.py`: the 8 models of subject VI.4, field names checked by a test
- [ ] 2.3 Internal models `Chunk` / `ScoredChunk` → moved to stage 3/4, written when
  first needed
- [x] 2.4 Validator on `MinimalSource`: `0 <= first < last`. **The 2000 ceiling is NOT
  in the model**: it depends on `--max_chunk_size` and the datasets go through the same
  model. It is checked in `search_dataset` before writing (task 5.3).
- [x] 2.5 Round-trip test on both `AnsweredQuestions` files, compared on the subject
  fields only (`difficulty` and `is_valid` are dropped: never used, not added)
- [x] 2.6 Union test: pydantic 2 "smart" mode picks the right variant **whatever the
  order** of the union (tested both orders). Found trap: a broken answered question
  silently falls back to `UnansweredQuestion` and loses its sources → added
  `AnsweredDataset` (answered-only) for `evaluate` to load the ground truth.
- [x] 2.7 Commit (tests: 20 passed, `make lint` green on 2026-09-29)

**Concept check**
- In `RagDataset`, why is the union ordered `AnsweredQuestion | UnansweredQuestion` and
  what happens if I reverse it?
- Where in the pipeline does a malformed JSON get caught, and what does the user see?

---

## Stage 3 — Ingestion and chunking

Goal: `index --max_chunk_size 2000` runs on the full corpus under 5 minutes and persists
to `data/processed/`, with exact character offsets.

### 3.a Concept pass (before any code)
- [ ] 3.a.1 Why chunk at all instead of indexing whole files
- [ ] 3.a.2 Chunk-size trade-off: signal dilution vs lost context, effect on recall@5
- [ ] 3.a.3 Why Python and Markdown need different splitters
- [ ] 3.a.4 Character offsets: why they must refer to the *original* file content, and
  why an off-by-one is invisible in my tests but fatal against the grader
- [ ] 3.a.5 Explain all four back in my own words

### 3.b Ingestion
- [x] 3.b.1 `src/ingest.py`: walk `data/raw/`, filter by extension list from stage 1.5,
  read each file with a context manager as UTF-8 (decide `errors=` policy and document
  its impact on offsets)
- [x] 3.b.2 Produce `file_path` **relative to the repo root** exactly as
  `data/raw/vllm-0.10.1/...`, with `/` separators regardless of OS
- [x] 3.b.3 tqdm over files

### 3.c Markdown / text chunker
- [x] 3.c.1 Design: split on headings, then paragraphs, then hard-split any block still
  above `max_chunk_size` (choose: overlap or not, and why)
- [x] 3.c.2 Implement with offsets tracked on the original string (no `.strip()`, no
  normalisation before offsets are recorded)
- [x] 3.c.3 Unit test: for every chunk, `original[first:last] == chunk.text`
- [x] 3.c.4 Unit test: every chunk length `<= max_chunk_size`

Increment 1 (2026-09-29): `src/ingest.py`, `src/chunking.py`, `Chunk` model,
`tests/test_chunking.py`. Findings: fence regex must use `[ \t]` not `\s`; docs
references never start/end with whitespace (0/100) → chunks are trimmed, LoRA reference
`[4695:6098]` is exactly one chunk. Best-case ceiling (a chunk with IoU > 0.05 exists),
`.py` still on the text splitter: 2000 → 10 014 chunks, docs 100/100, code 97/99;
1000 → 20 401, 100/100, 98/99; 500 → 41 849, 100/100, 98/99. Hidden dirs such as
`.buildkite/` are indexed: candidate exclusion, to measure in stage 6.

### 3.d Python code chunker
- [x] 3.d.1 Design decision: `ast` (function/class boundaries, robust to syntax errors?)
  vs line-based heuristics (`def`/`class` at column 0). Note the trade-off.
- [x] 3.d.2 Implement; handle: module-level code between definitions, decorators,
  nested classes, oversized functions (hard-split), files that fail to parse (fallback
  to the text chunker, never crash)
- [x] 3.d.3 Same two unit tests as 3.c.3 / 3.c.4
- [x] 3.d.4 Handle offset subtleties: `ast` gives line/col, I need character index —
  precompute line start offsets; beware `\r\n` and tabs

Increment 2 (2026-09-29): `split_python` with `ast`, recursive (module → class →
method → body statements), decorators + comment lines above kept with their def, small
statements packed, fallback to the text splitter on SyntaxError. 0 parse failures on
the corpus. Code refs: 42 in a function <= 2000 chars, 37 in a bigger one, 20 outside any
function. Best-IoU median vs text splitter: 2000 → 0.50 vs 0.45; 1000 → 0.84 vs 0.70;
500 → 0.47 vs 0.48. Whole corpus chunked in 1.8 s (12 554 chunks at 2000).
BM25 library question: allowed by the subject ("you may use any library you like");
decide in stage 4 (recode exercise argues for our own ~60 lines).

### 3.e Persistence
- [x] 3.e.1 Choose a format for `data/processed/` (JSON lines / pickle / both) and
  justify (load time matters for the 90 s retrieval budget)
- [x] 3.e.2 Store `max_chunk_size` in the index metadata so `search` can refuse a
  mismatched index
- [x] 3.e.3 Assert at write time that no chunk exceeds `max_chunk_size`
- [x] 3.e.4 Wire the `index` CLI command; time the full run (must be < 5 min)
- [x] 3.e.5 Commit

Increment 3 (2026-09-29): `src/indexer.py` (build, check, atomic save, load),
`ChunkIndex` model (chunks + max_chunk_size), `index` command with `--raw_dir` and
`--processed_dir`, `CliError` + clean exit in `src/__main__.py` (1 on user error, 130 on
Ctrl+C). Format: one JSON file, 17 MB, loads in 0.12 s. Size accepted: 100..2000
(memory: 280 MB at 200, 650 MB at 50). 1 205 files listed, 1 153 produce chunks (the
rest are empty `__init__.py`). Stage 8 note: fire runs `index` with defaults when an
option is misspelled (`--max_chunk_sise`), and only complains afterwards.

**Concept check**
- Show me where the character offsets are computed and prove they are correct.
- What happens to my offsets if I lowercase the text before chunking?
- Why would a 500-character chunk beat a 2000-character chunk on recall@5, and when
  would it lose?

---

## Stage 4 — Lexical index and scorer

Goal: `search "..." --k 5` returns plausible source locations in milliseconds.

### 4.a Concept pass
- [ ] 4.a.1 Term frequency, document frequency, why IDF is a logarithm
- [ ] 4.a.2 TF-IDF as sparse vectors + cosine similarity
- [ ] 4.a.3 BM25: saturation (`k1`), length normalisation (`b`), what each changes here
- [ ] 4.a.4 Inverted index and posting lists: why lookup is not a scan
- [ ] 4.a.5 The vocabulary-mismatch tension (paraphrase vs verbatim identifier)
- [ ] 4.a.6 **Decision:** TF-IDF or BM25 first (record the reasoning; the other can come
  in stage 6 as a comparison)

### 4.b Tokeniser (`src/tokenizer.py`)
- [ ] 4.b.1 Baseline: lowercase, split on non-alphanumerics
- [ ] 4.b.2 Identifier splitting: `snake_case` and `CamelCase` → sub-tokens, **while
  keeping the original token too**
- [ ] 4.b.3 Decide on stopwords (cost on docs vs code), stemming (yes/no, cost)
- [ ] 4.b.4 Same tokeniser used at index time and query time (single function, tested)

### 4.c Index (`src/index.py`)
- [ ] 4.c.1 Build the inverted index: `term -> list[(chunk_id, tf)]`, plus
  `df[term]`, chunk lengths, average chunk length
- [ ] 4.c.2 Persist alongside the chunks in `data/processed/`
- [ ] 4.c.3 tqdm on tokenising

### 4.d Scorer (`src/retriever.py`)
- [ ] 4.d.1 Implement the chosen scorer (BM25 or TF-IDF) over posting lists
- [ ] 4.d.2 Return top-k `ScoredChunk`, ties broken deterministically
- [ ] 4.d.3 Handle: empty query, query with zero known terms, `k=0`, `k > n_chunks`
- [ ] 4.d.4 Wire the `search` CLI command; output format like the subject screenshot
  (`path [first:last]`)
- [ ] 4.d.5 Sanity-check on 5 questions from stage 1 by eye
- [ ] 4.d.6 Commit

**Concept check**
- Why is IDF a logarithm? What would happen with a linear IDF on this corpus?
- What does `k1` control? What value would I try first here and why?
- What can a lexical scorer fundamentally not do?

---

## Stage 5 — `search_dataset` and `evaluate` (THE KEYSTONE)

Goal: I get a recall@k number I trust, in seconds, without the moulinette.
**Do not touch chunking or scoring parameters before this stage is done.**

- [ ] 5.1 Concept pass: recall@k vs precision@k, why recall here, why
  recall@1 ≤ recall@5 ≤ recall@10 by construction
- [ ] 5.2 `search_dataset --dataset_path --k --save_directory`: load `RagDataset`,
  run retrieval on every question (tqdm), write a `StudentSearchResults` JSON to
  `<save_directory>/<dataset filename>`; create the directory if missing
- [ ] 5.3 Enforce output invariants before writing: path prefix `data/raw/`, every
  source `<= 2000` chars, at most `k` sources per question
- [ ] 5.4 `evaluate --student_search_results_path --dataset_path`: my own recall@k
  implementation with the grader rule — same file **and** IoU > 0.05 on
  `[first, last]` — reporting recall@1/3/5/10
- [ ] 5.5 Write the IoU function with a unit test on: identical spans, disjoint spans,
  touching spans, one span containing the other, spans in different files
- [ ] 5.6 Match questions by `question_id`, warn (not crash) on missing ids
- [ ] 5.7 Cross-check my `evaluate` against the moulinette on the same output file: the
  numbers must match. If they don't, my implementation is wrong, not theirs.
- [ ] 5.8 Time `search_dataset` on 200 questions (must be < 90 s including index load)
- [ ] 5.9 Record baseline recall@5 (docs / code) at the top of this file
- [ ] 5.10 Commit

**Concept check**
- What exactly does recall@5 measure here, and why not precision?
- Why does returning k=10 never hurt recall, and why is that not the answer?
- Why is the code threshold 50 % when the docs threshold is 80 %?

---

## Stage 6 — Iteration loop (measured, not guessed)

Goal: ≥ 80 % recall@5 on docs, ≥ 50 % on code, and I can explain every change.

Method for each experiment: one variable at a time → re-index → `search_dataset` →
`evaluate` → log the number in an experiments table → keep or revert.

- [ ] 6.1 Create `notes/experiments.md` table: date, change, docs recall@5, code
  recall@5, indexing time, retrieval time
- [ ] 6.2 Error analysis first: list the 20 worst misses per dataset. For each: right
  file wrong span? wrong file? term missing from vocabulary? chunk too big?
- [ ] 6.3 Tokenisation experiments: with/without identifier splitting, stopwords,
  stemming, keeping digits, keeping dotted paths (`vllm.engine.arg_utils`)
- [ ] 6.4 Chunk-size experiments: 2000 / 1000 / 500 characters; overlap or not
- [ ] 6.5 Chunk-boundary experiments: heading level for md, function vs class for py,
  prepend file path / class name / heading breadcrumb to the chunk text
- [ ] 6.6 Scorer experiments: `k1` and `b` sweep (BM25), or the other scorer as a
  comparison
- [ ] 6.7 Query-side experiments: boost verbatim identifiers found in the question,
  weight docs vs code chunks per question type
- [ ] 6.8 Lock the winning configuration as defaults; re-check indexing < 5 min and
  retrieval < 90 s
- [ ] 6.9 Run the real moulinette once on both datasets and record the numbers
- [ ] 6.10 Commit; update the numbers at the top of this file

**Concept check**
- Which single change moved recall the most, and why does it make sense?
- What is the biggest remaining weakness of my retrieval, and how would I fix it with
  another week?

---

## Stage 7 — Generation with `Qwen/Qwen3-0.6B`

Goal: `answer` and `answer_dataset` produce valid, grounded, structured output.

### 7.a Concept pass
- [ ] 7.a.1 Grounding: what it means and how the prompt enforces it
- [ ] 7.a.2 Context budget: token counting, what to do when top-k exceeds it
- [ ] 7.a.3 Why a 0.6B model needs a much more constrained prompt than a frontier model
- [ ] 7.a.4 Hallucination: where it comes from, which design choices reduce it
- [ ] 7.a.5 Explain back in my own words

### 7.b Implementation
- [ ] 7.b.1 Add `transformers` + `torch` (CPU) to dependencies; check disk space
  (several GB); model download must be lazy and only happen on `answer*` commands,
  never on `index`/`search`
- [ ] 7.b.2 `src/generator.py`: load model and tokenizer once (context manager or
  cached singleton), CPU-only, graceful error if the download fails or is offline
- [ ] 7.b.3 Prompt template: system instruction ("answer only from the sources, say
  'I don't know' otherwise"), numbered sources with file paths, the question
- [ ] 7.b.4 Context budget: count tokens per source, include sources in rank order until
  the budget is hit, truncate the last one if needed
- [ ] 7.b.5 Qwen3 specifics: disable the thinking mode or strip `<think>` blocks from
  the output; set max new tokens; deterministic decoding
- [ ] 7.b.6 `answer <query> --k`: retrieve then generate, print the answer and the
  sources used
- [ ] 7.b.7 `answer_dataset --student_search_results_path --save_directory`: read a
  `StudentSearchResults`, re-read each source span from disk, generate (tqdm), write
  `StudentSearchResultsAndAnswer` JSON scoped by dataset
- [ ] 7.b.8 Manually review 10 answers per dataset for coherence / grounding / on point
- [ ] 7.b.9 Commit

**Concept check**
- What does my system do with a question that has no answer in the corpus?
- Walk me through what happens between typing a query and getting an answer.

---

## Stage 8 — CLI hardening

Goal: I try to break my own CLI and fail. Zero unhandled tracebacks.

- [ ] 8.1 Write an abuse checklist and run it on every command:
  - empty query `""`, whitespace-only query, nonsensical query `"xqzv plorp"`
  - `--k 0`, `--k -1`, `--k 100000`, `--k abc`
  - missing dataset file, malformed JSON, valid JSON with wrong schema
  - `search` before `index` (no `data/processed/`), corrupted index files
  - `--save_directory` that does not exist, that is a file, that is read-only
  - `--max_chunk_size 0`, `-1`, `10`, `50000`
  - `answer` with no model / no network
  - Ctrl+C during indexing / generation
- [ ] 8.2 One top-level `try/except` in `__main__` that prints a clean message and exits
  non-zero, plus targeted handling inside each command
- [ ] 8.3 tqdm on every long loop (files, chunks, questions, generation)
- [ ] 8.4 Context managers on every file / model resource
- [ ] 8.5 `make lint` green; try `make lint-strict`
- [ ] 8.6 pytest suite covering the checklist (not graded, but it is my safety net for
  the recode exercise during the defense)
- [ ] 8.7 Fresh-clone rehearsal: `uv sync` → `index` → `search_dataset` → moulinette →
  `answer_dataset`, exactly as in subject VI.7.2
- [ ] 8.8 Commit

**Concept check**
- Where is each degenerate input caught, and what does the user see?

---

## Stage 9 — README (English)

Goal: covers every required section with real numbers.

- [ ] 9.1 First line, italicised: *This project has been created as part of the 42
  curriculum by acombier*
- [ ] 9.2 Description
- [ ] 9.3 Instructions (install, run, every CLI command with an example)
- [ ] 9.4 Resources: references (BM25 paper, TF-IDF, RAG paper, pydantic/Fire docs)
- [ ] 9.5 Resources: **how AI was used** — honest, per task and per part of the project
  (what was explained, what was drafted by AI and reviewed, what I wrote myself)
- [ ] 9.6 System architecture (pipeline diagram: ingest → chunk → index → retrieve →
  augment → generate → evaluate)
- [ ] 9.7 Chunking strategy (both strategies, offsets, chunk-size effect on recall)
- [ ] 9.8 Retrieval method (tokeniser, scorer, parameters, inverted index)
- [ ] 9.9 Performance analysis (recall@1/3/5/10 on both datasets, indexing time,
  retrieval time, the experiments table)
- [ ] 9.10 Design decisions (every choice with its trade-off)
- [ ] 9.11 Challenges faced
- [ ] 9.12 Example usage (real command outputs)
- [ ] 9.13 Commit

---

## Stage 10 — Bonuses (only once mandatory fully validates)

Priority order chosen for interview value: 1 → 2 → 4 → 3 → 5. All CPU-only.

- [ ] 10.1 Semantic embeddings: `sentence-transformers` + `all-MiniLM-L6-v2`, embed
  every chunk at index time, cosine search (numpy or faiss-cpu); measure recall@5 alone
- [ ] 10.2 Hybrid retrieval: merge lexical and semantic rankings (reciprocal rank fusion
  or weighted scores); measure; explain why it beats each one alone
- [ ] 10.3 Caching: on-disk cache of the loaded index (cold start) and of query results
  (repeated queries), with invalidation when the index changes
- [ ] 10.4 Incremental indexing: hash per file, re-chunk and re-index only changed files,
  rebuild posting lists for those chunks only
- [ ] 10.5 Local HTTP API: small FastAPI/Flask server exposing `/search` and `/answer`
- [ ] 10.6 Each bonus: flake8 + mypy clean, documented in README, demonstrable live

---

## To revisit (notions skipped with `skip`, see CLAUDE.md 3.1bis)

Re-ask these unannounced in a later session. Tick when answered without notes.

- [ ] (2026-09-25) `uv.lock` vs `pyproject.toml`: what the lock guarantees that the
  constraints alone do not (exact pinned versions → identical env at the reviewer's).
- [ ] (2026-09-25) `--disallow-untyped-defs`: what mypy does with `def f(x):` without
  the flag (skips the body silently) and with it (error, forces annotations).
- [ ] (2026-09-25) Offset drift with undetected `\r\n` files: my offsets shrink by one per
  line, chunks point too early, IoU < 0.05 on long files, and my own `evaluate` cannot
  see it because it shares the bias.
- [ ] (2026-09-25, stage 1 check) 12-char reference inside a 1 500-char chunk: lost,
  IoU = 12/1500 = 0.008. Some references are lost by construction (4 docs + 2 code at
  2 000 chars).
- [ ] (2026-09-25, stage 1 check) Why index `.txt`: 3/100 docs sources are
  `CMakeLists.txt`; what the grader expects matters, not the corpus file count.
- [ ] (2026-09-25, stage 1 check) Heading-based md chunking: 79/100 spans start on a
  heading, 53/100 end before the next. Code: only 41/99 start on def/class/decorator,
  50 start mid-function → def-based splitting is not enough.

Since 2026-09-29 (CLAUDE.md 3.4): no quizzes during the build. This list is a revision
sheet to read before the defense.

---

## Defense drills (run unannounced throughout)

- [ ] Explain RAG to someone who has never heard of it, in 60 seconds
- [ ] Why chunk the corpus? What if I indexed whole files?
- [ ] Walk through query → answer, step by step
- [ ] Why BM25 over TF-IDF (or the reverse)? What does `k1` control?
- [ ] Why is IDF a logarithm?
- [ ] What does recall@5 measure here, and why not precision?
- [ ] Why is the code threshold 50 % when docs is 80 %?
- [ ] Show where offsets are computed and prove they are correct
- [ ] What does the system do with a question that has no answer in the corpus?
- [ ] Biggest weakness of my retrieval and the one-week fix
- [ ] Corporate wiki instead of vLLM: what changes?

---

## Definition of done (from `CLAUDE.md` section 12)

1. [ ] I can answer every defense drill without notes
2. [ ] Targets met: indexing ≤ 5 min, 200 questions ≤ 90 s, recall@5 ≥ 80 % docs and
   ≥ 50 % code (verified with the moulinette)
3. [ ] `make lint` green and the CLI survives the stage 8 abuse checklist
4. [ ] README written and honest
5. [ ] I could restart the pipeline from an empty directory on another corpus
