# Relocation note — resume here (written 2026-10-14)

Read this once in the new clone, then delete it or keep it as a log. `CLAUDE.md`
section 13 and `TODO.md` are also up to date as of this note.

## Why we moved

The home disk of the Lenovo (18 GB) cannot hold the CUDA build of `torch` (about
4.5 GB with the NVIDIA libraries) next to the `uv` cache and the Hugging Face model.
Decision: the project now lives in the goinfre, with both caches redirected there.
`pyproject.toml` no longer pins the CPU wheel of `torch`: `uv sync` installs the
standard PyPI build with CUDA, and `src/generator.py` falls back to CPU when no GPU is
visible. The project is always run and defended on this machine (RTX 4050, 6 GB).

Checklist for the new clone (all steps, in order):

```
mkdir -p /goinfre/acombier/uv-cache /goinfre/acombier/huggingface
mv ~/.cache/huggingface/hub /goinfre/acombier/huggingface/hub   # keeps the 1.5 GB model
echo 'export UV_CACHE_DIR=/goinfre/acombier/uv-cache' >> ~/.zshrc
echo 'export HF_HOME=/goinfre/acombier/huggingface' >> ~/.zshrc
source ~/.zshrc
git clone <url> /goinfre/acombier/RAG
cp -r ~/Documents/RAG/notes ~/Documents/RAG/data /goinfre/acombier/RAG/   # git-ignored
cd /goinfre/acombier/RAG && uv sync
uv run python -c "import torch; print(torch.__version__, torch.cuda.is_available())"
```

Expected: `2.14.1 True`. Then `make lint` (first run ≈ 40 s to fill the mypy cache)
and `uv run pytest -q` (97 tests). The old clone in `~/Documents/RAG` has a
half-synced `.venv`: do not run `uv run` there without `--no-sync`.

Caveat: the goinfre is not backed up. Commit and push often; keep a copy of
`notes/` (corpus notes, experiments log, the moulinette binaries, the zips) in the
home.

## State of the project

Commit `726abaf` "stage 7" is the last one, pushed. Stages 0–6 done, stage 7 in
progress, lint green, 97 tests.

| Stage | Status |
|---|---|
| 0–3 setup, exploration, models, chunking | done |
| 4 tokeniser + BM25 (`rank_bm25`) + `search` (+ `--render_path` debug) | done |
| 5 `search_dataset`, `evaluate` (matches the moulinette exactly), `--show_misses` | done |
| 6 iteration loop | done enough: docs 85.0 %, code 75.8 % recall@5 (thresholds 80 / 50) |
| 7 generation | `answer` works, `answer_dataset` works (checked on 3 questions); full run and prompt review pending |
| 8 CLI hardening, 9 README, 10 bonuses | not started |

### Numbers to remember

- Baseline recall@5: docs 80.0, code 68.7. After stopwords, `MIN_PIECE = 2` and
  `b = 0.5`: docs **85.0**, code **75.8** (k=10 files, chunks 2000, k1=1.5). All in
  `notes/experiments.md` with the rejected experiments (chunks 1000, MIN_PIECE 3,
  b 0.6 / 0.3).
- Remaining misses at k=5: docs 15 (6 same_file, 4 wrong_file, 4 late, 1 near), code
  24 (10 same_file, 7 wrong_file, 7 late). Candidates not tried: pack small Markdown
  sibling sections, acronym split rule (`APIs` → ap+is, `MXFP4` → mxf+p4), `k1` sweep.
- Indexing 2.2 s (4 s on this machine); retrieval ≈ 15 s for 200 questions on the
  other machine, ≈ 45 s here (this CPU is 5x slower on tokenising: 19 s per load).
- Generation on this CPU (AVX2, float32): prompt ≈ 3 000 tokens, prefill 43 s,
  decode 0.77 s/token, ≈ 72 s per answer → 2 h per dataset. Hence the GPU.

### Stage 7 as built

- `src/generator.py`: `fit_sources` (token budget, pure function), `build_messages`
  (system rules + numbered sources with paths + question), `Generator` (lazy torch
  import, Qwen3-0.6B, float32, GPU if available, greedy decoding, thinking disabled,
  `max_new_tokens` 256, context budget 3 000 tokens). `NO_ANSWER = "I don't know."`.
- `src/datasets.py`: `save_model` (atomic JSON), `load_chunk` (re-reads
  `file_path[first:last]` from the corpus, refuses out-of-range spans).
- `src/cli.py`: `answer <query> --k` (retrieve, then load the model, prints answer and
  the sources kept), `answer_dataset --student_search_results_path --save_directory
  [--model_name] [--limit N]` writing `StudentSearchResultsAndAnswer` under
  `data/output/search_results_and_answer/<Scope>/<same file name>`.
- `tests/test_generator.py` (6 tests on budget and prompt, no model),
  `tests/test_datasets.py` (+2 on `load_chunk`).
- `pyproject.toml`: mypy overrides skip `torch`/`transformers` (they ship `py.typed`,
  following them took minutes); torch from PyPI with CUDA.

### Findings from the first answers (docs dataset, `--limit 3`)

1. LoRA endpoint: source at rank 1, answer right.
2. `lm_eval` command: source at rank 3, answer right **after** the prompt change
   "copy commands in full inside a code block, never shorten with '...'" (the first
   version elided the command). Model name taken from int4.md instead of int8.md:
   equivalent.
3. `LLM.embed`: a `wrong_file` miss (no chunk of `pooling_models.md` in the top 10);
   the model answered `llm.generate` with confidence instead of "I don't know."
   **Grounding failure on abstention**: a 0.6B model does not abstain on its own.
   Two fixes to test: a harder instruction, or a code-side guard (no generation when
   the best BM25 score is under a threshold).
- The citation rule is weak: the model cites `[1]` only, or nothing. Suggested
  rephrase: "Cite every source you used by its number, for example [3] or [1][4]."
- Lesson recorded: the demo question "How do I enable LoRA adapters" is NOT in the
  dataset; its true answer (`enable_lora=True`) is in `lora.md [0:1882]`, found at
  rank 2. The span [4695:6098] belongs to another question. Never diagnose without
  the ground truth.
- The generator cannot repair retrieval: recall@5 is the ceiling of answer quality.
  No automatic grading of answers exists (moulinette's `evaluate_student_answers` is
  not implemented); quality is judged by reading answers next to the `answer` field
  of `data/datasets/AnsweredQuestions/*.json`.

## Next steps, in order

1. In the goinfre clone: verify CUDA, re-run
   `uv run python -m src answer "What command can be used to evaluate the accuracy of a quantized model using lm_eval with vLLM?" --k 5`
   and compare the time with 76 s on CPU.
2. Abuse cases for `answer` not run yet: `answer "" --k 5` and
   `answer "xqzv plorp" --k 5` (expected `I don't know.`), and `answer_dataset` with a
   dataset file instead of a results file (expected "does not match
   StudentSearchResults" before loading the model).
3. Full run: `answer_dataset` on `data/output/search_results/UnansweredQuestions/dataset_docs_public.json`
   with `--save_directory data/output/search_results_and_answer/UnansweredQuestions`,
   then the same for code. Note the time per answer on GPU.
4. Read 10 answers per dataset next to the reference (TODO 7.b.8); then decide on the
   abstention guard and the citation rephrase; re-run with `--limit 10` to compare.
5. Update `TODO.md` stage 7 boxes and `CLAUDE.md` section 13, commit.
6. Stage 8 (abuse checklist, every command), stage 9 README with the experiments
   table and an honest "how AI was used" section.

## Working mode (unchanged)

Claude explains the notion briefly, then writes the code directly in the repo when
asked ("edit la codebase pour moi"); the user runs the commands himself and pastes the
output. No quizzes. French in the chat, English in code, comments and docs.
