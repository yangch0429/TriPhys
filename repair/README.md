# Question repair scripts (provenance)

This directory holds the scripts that repaired the generated questions into the released benchmark, **version 08**
(410 QAs / 200 videos; MPR 228 / KPR 182; train 300 (MPR 167 / KPR 133), test 110 (MPR 61 / KPR 49)).
"A" / "A-class" in the code means MPR and "B" / "B-class" means KPR.

**These files are a provenance record.** They are the scripts that ran, with only the following release edits:

- API keys, endpoints, model ids and machine paths are read from environment variables (listed below).
- Each file starts with a short provenance header that gives its original location.

The logic is unchanged. **The scripts are not guaranteed to run end to end:**

- several of their dependencies and intermediate files lived on a training server that is no longer available (see "Missing dependencies");
- they call paid APIs (LLMs, Serper);
- several steps were driven by hand-written decisions recorded inside the scripts.

**The canonical benchmark is the JSON in `data/` of this repository.** Use these scripts to see how a question was
changed, not to regenerate the data.

Some scripts and docstrings refer to dates. All of them are in 2026 (MM-DD).

## Versions

| Version | Date | What changed |
|---|---|---|
| 00 | 08-25 | Raw generated benchmark: 235 A + 235 B QAs (construction pipeline, see `construction/`). |
| 01 | 09-29 | Invalid QAs were removed (A 233, B 212) and the data was split by video. Train: A 171 / B 157. Test: A 62 / B 55. Unrepaired baseline. |
| 02 | 09-30 | **L0**: the automatic v8 query / url / code text fixes were re-applied. Numbers and answers are unchanged. |
| 03 | 10-02 | **L1 / L2 repair** plus post layers (`l0_l2/`). Web values were re-sourced from real Serper snippets with a verbatim quote and url, code was re-executed and answers were recomputed. Train is final from here until fix10. |
| 04 | 10-04 | One A-test question was reworded ("amateur", q14). |
| 05 | 10-05 | **fix7**: 7 test QAs were fixed (`apply/fix7_apply.py`). |
| 06 | 10-05 | **fix7b**: 2 of those QAs got a second pass (`apply/fix7b_apply.py`). |
| 07 | 10-11 | **fix8**: A\|test\|4\|0 (FIBA arc, TYPE_B → TYPE_A) (`apply/fix8_apply.py`). 445 QAs. |
| 08 | 10-13 | **fix10** (`wellposed/`, `fix9/`, `fix10/`, `apply/fix10_apply.py`, `fix10/rebalance_split.py`). 445 → 410 QAs / 200 videos. **Released version.** |

## Layout and run order

### 1. `l0_l2/`: versions 01 → 03 (09-30 .. 10-02)

Shared helpers:

- `fixlib.py`: LLM calls, Serper calls with key rotation, data access, code execution and access to the construction stage outputs.
- `v8_common.py`: the helpers of the earlier v8 SFT pipeline.

Steps, in the order they ran:

1. **Inputs from the v8 pipeline.** `v8_ws_pipeline.py` and `v8_wsfix.py` test the web_search queries against Serper with strict LLM value validation. `v8_writeback.py` writes the train fixes and `v8_test_code_check.py` reports test code statistics. These scripts produced `server_reports/*`, which L0 reads.
2. **L0.** `problem_list.py` builds `problem_qas.json`. `l0_apply.py` writes `stage_L0/`. `l0_garble.py` and `l0_labels.py` clean up text.
3. **L1, in-place repair with real evidence.**
   - `l1_evidence.py` extracts the value from the collected Serper results.
   - `l1_plaus.py` checks that value against the video description.
   - `l1_rewrite.py` rewrites the QA under programmatic checks and an LLM review.
   - `l1_textfix.py`, `post_redo_fix.py` and `post_textfix.py` apply cosmetic fixes.
   - `screen.py` and `review_dump.py` support the one-by-one manual review (`review_log.json`).
4. **L2, regeneration** with the construction Stage-3 generator.
   - `_patch_gen.py` patches the repair copy of `agentic_physics_qa.py` once: unit-aware leak check, value-free re-search, TYPE_B range and a threaded `validate_grounding`.
   - `l2_regen.py` calls `generate_single_qa` (generator deepseek-v3.2, tools web_search / page_browser / code_executor). Every source must pass `validate_grounding`; the script makes up to 4 attempts and then runs an LLM review.
   - Afterwards come `l2_salvage.py` and `post_l2_fix.py`.
5. **Post layers.** They are composed by `current.py` in this order:
   1. L1E (`l1e.py`)
   2. CONS (`consistency_check.py`, `cons_fix.py`)
   3. L1T (`_l1t_search.py`, `_l1t_hints.py`, `l1t.py`, `l1t_patch.py`)
   4. FIX39 / FIX39W (`_fix39_search.py`, `_fix39w_hints.py`, `fix39.py`, `l1t.py` with `L1T_SET=fix39w`)
   5. FIX3 (`fix3.py`)
   6. CONS2 (`cons2_review.py`, `cons_fix.py` round 2, `cons2_check.py`, `cons2_patch.py`)
   7. FIX4 (`fix4.py`)
   8. FIX5 / FIX6 (`fix5.py`, with `FIXLEVEL=FIX6` for FIX6; `_mkreject5.py`)
   9. FIX7 (`fix7.py`, `_srch.py`)
   10. FIX8 (`fix8.py`)

   Note that `fix7.py` / `fix8.py` here are post layers of version 03. They are not the 10-05 / 10-11 rounds.
6. **Write-back and rechecks.** `final_writeback.py` writes version 03. The rechecks are `final_recheck.py`, `strict_check.py`, `deep_audit.py` and `video_audit.py` (Gemini watches the base64 clip and lists contradictions). Their findings fed the FIX3 .. FIX8 layers.

### 2. `apply/`: versions 05 → 07

- `fix7_apply.py` (05) and `fix7b_apply.py` (06) contain hand-verified replacements. They re-execute every new code block.
- `fix8_apply.py` (07) contains one QA.

`fix7_apply.FIX` is the data directory that all apply scripts use.

### 3. `wellposed/` and `fix9/`: diagnosis of version 07 (10-11, 445 QAs)

- `wellposed_check.py` checks whether the question text determines the reference answer. Every check ends in executed code scored with the official ACC function.
- `qual_check.py` labels every term of the base search query as stated / implied / generic / added.
- `ana_tool_v36.py` is the tool-use helper that `wellposed_check` imports.
- `fix9/fix9.py` was a one-pass repair pilot. **It was never applied to the data** (no fix9 log or backups exist). fix10 still imports it for static checks and Serper result links (`TOOL_RE`, `hidden_nums`, `leaked`, `links`).

### 4. `fix10/`: version 07 → 08 (10-11 .. 10-13)

The design is described in `fix10/DESIGN.md` (criteria v2, query-first repair, the `unique()` search test). Steps:

1. **Recount.**
   - `recount.py` writes `recount.json`. It checks:
     - W: the base query returns one score-relevant value and is derivable from the question;
     - M: modelling;
     - S: static leaks;
     - V: video claims against frames;
     - C: the code reproduces the answer.
   - `recheck_ns.py` re-judges "not shown" claims on dense frames and writes `recount_final.json`.
   - `rejudge.py` applies criteria v2 with video and writes `recount_v2.json`: **255** needs_fix2 QAs.
   - `units.py` handles SI normalisation of extracted values.
2. **Search repair.** `fix10.py` is the v7 base module (query-first constant repair, writer, reviewer and checks). `fix10_v13.py` .. `fix10_v17.py` are successive versions; each overrides parts of `fix10` (`import fix10 as X`) and the previous version, and each was generated from the previous one by a small patch script.
   - The full run over the 255 QAs gave `fix10_full_v13.json` (104 fixed / 151 unresolved).
   - `fix10_v15.py` ran on the 151 unresolved QAs. `given_convert.py` then turned web constants that no search states uniquely into stated givens (`given_v15.json`). *(The command lines of these two runs were not preserved.)*
3. **Type check and audit** (`fix10_chain4.sh`).
   - `type_check.py` and `make_seeds16.py` find wrong answer forms: 5 more QAs, for **260** needs_fix2 in total.
   - `fix10_v16.py` runs on the remaining 57, followed by `given_convert.py`.
   - `audit10.py` runs a strict whole-QA audit with video frames (reviewer gpt-5.6-terra).
   - Then come 3 × (`audit_fix.py` repair + re-audit), and `report10.py`.
4. **Redesign** (`fix10_chain5.sh`, `fix10_chain6.sh`).
   - `fix10_v17.py` runs on 11 QAs and `redesign17.py` on 37 QAs.
   - `redesign18.py` follows. It is redesign17 plus the v18 changes listed in its docstring, and runs in up to 3 passes.
   - `merge_raudit.py` and the audit / repair rounds follow, ending with `report10.py`.
   - The id lists `ids11.json` / `ids37.json` were written by hand-run one-liners that were not preserved.
5. **Selection.**
   - `select_final.py` keeps a needs_fix2 QA only if its last version passed the strict audit with a matching signature.
   - It keeps the 07 version for 4 videos whose QAs failed every redesign, and adds extra redesigned QAs until 410 QAs / 200 videos.
   - Its output is `final/final_ids.json` and `select_report.txt`.
   - Final counts (`select_report.txt`):
     - 230 needs_fix2 QAs passed: 219 repaired QAs plus 11 of 36 extra candidates.
     - 185 untouched + 219 repaired + 4 kept-original = 408 QAs.
     - 2 extras were then added, giving 410 QAs / 200 videos.
     - All other needs_fix2 QAs were deleted.
6. **Apply.** `apply/fix10_apply.py` writes the six dataset files. Two test videos with unrepaired QAs (a00588/00436, a00818/01984) moved to train.
7. **Re-balance.** `fix10/rebalance_split.py b` restores the KPR strata. The input is `*.before_rebalance.json`.
   - Moved from train to test: a00209/00291, a00252/01129, a00786/00063, a00818/01712.
   - Moved from test to train: a00198/00150, a00315/01016, a00759/01079.
   - The test set never contains a QA, or a hidden value, that a trained model saw.

The remaining scripts in `apply/` do not change any question:

- `fix10_assemble.py` and `final_assemble.py` re-assembled the zero-shot result files of the six evaluated models after fix8 / fix10.
- They re-scored only the QAs that changed, using the server scoring module; the ACC judge was gpt-4o-mini.

**Which "fix10 version" is final?** There is no single version. The data of version 08 comes from a cascade of record files. Later files override earlier ones:

1. `fix10_full_v13.json`
2. v15 / given_v15
3. v16 / given_v16 / seeds
4. v17
5. redesign17(b)
6. redesign18(b, c)
7. the audit_fix repairs

So the last search-repair script is `fix10_v17.py` and the last redesign script is `redesign18.py`; both build on the shared base `fix10.py`.

*Uncertain:* 40 records of `fix10_full_v13.json` carry a field that only `fix10_v14.py` writes. The full run was therefore probably resumed with v14. The record files listed above are not part of this release; the resulting QAs are in `data/` (plus `fix10_log.json` / `final_ids.json` in the authors' archive).

## Models

| Stage | Role | Model (env var, default) |
|---|---|---|
| L0 / L1 / L2 review, first L1E round | reviewer / rewriter | gpt-6-astra via an OpenAI-compatible relay (`REVIEWER_MODEL`; `CHAT_API_BASE`, `OPENROUTER_API_KEY`) |
| from the L1E rerun on, CONS .. FIX8, deep_audit | reviewer / rewriter | DeepSeek-V4-Flash, API id `deepseek-flash` (`DEEPSEEK_MODEL`; `DEEPSEEK_API_KEY`) |
| L2 regeneration | Stage-3 generator | deepseek-v3.2 (`DEEPSEEK_GEN_MODEL`; through `CHAT_API_BASE` / `OPENROUTER_API_KEY`) |
| video_audit | video grounding | Gemini-3.6-Flash (`GEMINI_MODEL`, default `gemini-3.6-flash`) |
| wellposed / qual_check / fix9 / fix10 | checker, writer, fixer (with video frames) | `deepseek-flash` (`WP_MODEL`, `FIX_MODEL`, `DEEPSEEK_MODEL`) |
| fix10 reviewer, audit10 | independent reviewer / strict audit | `fix10.py` default gpt-4o; v13+ and audit10 gpt-5.6-terra (`FIX10_REVIEWER`). Both go through the ACC-judge endpoint (`ACC_JUDGE_BASE`). On OpenRouter the id may need a provider prefix. |
| ACC scoring inside the checks | official ACC (TYPE_D / fallback judge) | gpt-4o-mini, through the server `metrics_final` module |

Web search uses Serper (`https://google.serper.dev/search`). Every Serper call in L0–L2 is logged to `search_log.jsonl`, and the logged url / snippet is the evidence that is cited.

## Checks

The checks applied along the way include:

- **Real evidence.** Every web_search source carries a verbatim quote from a real Serper result.
- **Code reproduces the answer.** Code is re-executed, and the answer, or both ends of an answer range, must be printed by the code.
- **No leaks.** Hidden or web values must not appear in the question (unit-aware). Tool names must not appear in the question.
- **Fields stay fixed.** question_type / difficulty / hidden-reading counts are preserved. For KPR, hidden_cv_chosen follows the web value.
- **Cross-QA consistency.** Properties of the same object within one video must agree.
- **Video consistency.** Claims are checked against the video (Gemini on the clip, or deepseek-flash on frames).
- **Well-posedness.** Alternative values or modelling choices scored with the official ACC must not change the score unless the question excludes them.
- **Unique-value search test (fix10).** The base query and a paraphrase return one value, from ≥ 2 sites or a standard source.
- **Strict audit.** For fix10, a whole-QA audit with frames.
- **Answer form matches type.** One value → TYPE_A, an interval → TYPE_B.
- **Split leak rule.** The test set holds no QA identical to a trained one and no trained hidden value.

## Environment variables

| Variable | Used by | Meaning |
|---|---|---|
| `DEEPSEEK_API_KEY`, `DEEPSEEK_API_BASE` (default `https://api.deepseek.com`), `DEEPSEEK_MODEL` (default `deepseek-flash`) | fixlib, wellposed_check, recount, recheck_ns, fix10 | DeepSeek chat endpoint |
| `DEEPSEEK_GEN_MODEL` (default `deepseek-v3.2`) | l2_regen | L2 generator |
| `CHAT_API_BASE` (default `https://openrouter.ai/api/v1`), `OPENROUTER_API_KEY` | v8_common, l2_regen, video_audit | OpenAI-compatible endpoint (originally a relay) |
| `REVIEWER_MODEL` (default `gpt-6-astra`) | v8_common | reviewer of the v8 pipeline |
| `GEMINI_MODEL` (default `gemini-3.6-flash`), `OPENROUTER_API_KEY` | video_audit | video grounding |
| `ACC_JUDGE_BASE` (default `https://openrouter.ai/api/v1`), `ACC_JUDGE_API_KEY` (falls back to `OPENROUTER_API_KEY`) | wellposed_check → fix10 reviewer / audit10 | the original "ACC-judge endpoint" |
| `FIX10_REVIEWER` | fix10*, audit10 | reviewer model id |
| `SERPER_API_KEYS` | fixlib, v8_common, wellposed_check (overrides the key list of the exec'd sampler), fix9 / fix10 via wellposed_check | comma-separated Serper keys, rotated |
| `TRIPHYS_WORK_DIR` | fixlib, problem_list, apply/*, rebalance_split | directory with `{a,b}class_qa_{train,test}.json` / `*_dataset_cleaned.json` (originally `fix_code/`) |
| `V8_WORK_DIR` | fixlib | repair state directory (originally `fix_code/_v8_work/`; default: the script directory) |
| `TRIPHYS_EXP_ROOT` | v8_common, wellposed_check, ana_tool_v36 (and fix9/fix10 through them) | server experiment directory (`data/v9`, `scripts/`, `results/`) |
| `TRIPHYS_CONSTRUCTION_ROOT` | fixlib | root holding `A_class_Benchmark/output/*/<video>` and `B_class_Benchmark/outputs/<video>` (construction stage-1/2/2b/2.5 outputs) |
| `CONSTRUCTION_DIR` | l2_regen | directory containing the repair copy of `agentic_physics_qa.py` |
| `EGOEXO_ROOT`, `BAAI_ROOT` | recount (and fix10 through it) | videos: `<EGOEXO_ROOT>/<take_uid>/best_exo.mp4`, `<BAAI_ROOT>/<folder_id>/<video_id>.mp4` |
| `FIX1001_DIR` | video_audit | directory with `run_api_subset.py` (`video_b64`) |
| `EXPERIMENT_RIGHT_DIR` | fix10_assemble, final_assemble | server scoring code (`metrics_final`) |
| `DATASET_TOOL_DIR` | rebalance_split | directory with `dataset_processing_tool.py` |
| `TRIPHYS_PYTHON` (default: current interpreter) | wellposed_check | interpreter used to execute solution code |
| `PYTHON` | fix10_chain*.sh | interpreter |

The original tuning variables are kept as they were:

- `V8_SPLIT`, `FIXLEVEL`, `L1T_SET`, `ROUND2`-style switches (see each script);
- `WP_MODEL`, `FIX_MODEL`;
- `FIX10_OUT`, `FIX10_PAR`, `FIX10_SERPER_MAX`, `FIX10_TARGET`;
- `RECOUNT_OUT`, `MODE`, `AUDIT_FIX_MAXR`, `REDESIGN_N`, `REDESIGN_PAR`.

## Missing dependencies

The following were not part of these directories and are **not released here**:

- **`metrics_final`** (server scoring module). It provides `calculate_acc_with_fallback` / `calculate_scs_with_fallback`. Its released counterpart is `triphys/metrics.py`; the function names differ, so the imports would need adapting.
- **`sample_dpo_v9_cached.py`** (server `scripts/`). `wellposed_check.search` execs only its Serper key list and `query_web_search`; the counterpart is the web_search tool in `triphys/tools.py`.
- **`run_api_subset.py`** (server evaluation directory). `video_audit` needs only its `video_b64` (re-encode the clip and base64-encode it).
- **`agentic_physics_qa.py`, repair copy** (`fix_code/benchmark_construction/`, patched by `_patch_gen.py`). Used by `l2_regen.py` / `l2_salvage.py` for `generate_single_qa`, `validate_grounding` and `extract_qa_pairs`. The construction code is released under `construction/`. Its module path and patch state may differ, so the imports and `CONSTRUCTION_DIR` may need adjusting; `_patch_gen.py` still uses the relative path `../benchmark_construction/agentic_physics_qa.py`.
- **`dataset_processing_tool.py`** (`merge_rare_strata`, `extract_physics_main_category`). Used by `rebalance_split.py`.
- **Intermediate state.** These files are not released:
  - v8 pipeline state (`data/v8_ws_state.json`, `server_reports/*`);
  - `search_log.jsonl`, `review_log.json`, `evidence*.json`, `*_results.json`;
  - construction stage outputs and LLM caches;
  - fix10 record files (`recount*.json`, `fix10_full_v13.json`, `fix10_v15/16/17.json`, `given_*`, `seed_v16.json`, `redesign17*/18*`, `repair_v16/v18*`, `audit_v16.json`, `ids_extra.json`, `keep_orig.json`).
- **Python packages:** `requests`, `httpx`, `openai`, `opencv-python` (frames), `numpy`.

The following scripts were deliberately left out:

- pilot versions `fix10_v8` .. `fix10_v12`;
- the patch scripts that generated each `fix10_v*` / `redesign18` from its predecessor (the generated files are included);
- one-off viewers and diff scripts;
- the evaluation re-run drivers;
- `fix9_apply.py` (never run).
