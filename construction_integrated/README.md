# Benchmark construction (integrated)

This directory holds one self-contained construction pipeline for the MPR (A-class, EgoExo4D) and KPR (B-class,
BAAI Physics-aware-videos) question pools. The checks that make a question well-posed are part of construction:
Stage 3 writes every question under grounding, leak and well-posedness checks, Stage 4b verifies every question and
repairs it within a bounded budget, and Stage 5 keeps only verified questions. API keys, endpoints, model ids and
paths come from the environment.

## Layout

| file | role |
|---|---|
| `aclass_pipeline_core.py` | per-video pipeline `run_pipeline_for_video()` (Stages 1-5) used by both classes; standalone per-folder runner |
| `run_aclass.py` | MPR runner over `manifest.json` (all stages, resumable, shared physics library) |
| `run_aclass_batch.py` | MPR batch run over a take list (4 in parallel) |
| `run_baai_160_batch.py` | KPR runner over the BAAI selection (4 in parallel) |
| `extract_video_info.py` | Stage 1: clip segmentation (ffmpeg) + Gemini clip descriptions / observable quantities |
| `classify_b1_b2.py` | Stage 1.5 (KPR): LLM picks the `searchable_objects` whose specs can be web-searched |
| `physics_extractor.py` | Stage 2.5 CV measurements (MPR) / scale-free VLM estimates (KPR); EgoExo4D GT loading |
| `build_event_graph.py` | Stage 2: event graph |
| `build_physical_event_library.py` | Stage 2.5a: shared physics event library |
| `expand_event_graph_with_physics.py` | Stage 2.5b: graph expansion with physics matches |
| `agentic_physics_qa.py` | Stage 3 author agent for **MPR** |
| `agentic_physics_qa_kpr.py` | Stage 3 author agent for **KPR** |
| `grounding.py` | Stage 3 checks: grounding, leaks, answer / range form, unique-value search test, requery / qualify, stated givens |
| `check_qa_quality.py` | Stage 4: LLM quality check (with tools) |
| `verify_qa.py` | Stage 4b: verification of every QA (code, Stage-3 checks, video, web, model, consistency, quality, audit) |
| `repair_qa.py` | Stage 4b: bounded repair (smallest fix, rewrite) |
| `verify_common.py` | shared helpers: model calls with on-disk cache, Serper search + search log, code execution, official ACC, frame sampling |
| `units.py` | unit conversion for comparing values across units |
| `filter_valid_qa.py` | Stage 5: rule filter |
| `split_videos.py` | video-level stratified train/test split with greedy exchange |
| `tools/` | agent tools: `web_search`, `page_browser`, `web_image_search` (Serper + Jina), `code_executor`, `visual_question_answering`, video/image/audio search |
| `select_videos/` | video selection (`select_egoexo200_physics.py`; `select_baai200_library.py` → `download_baai200.py` → `select_baai_160.py`) and EgoExo4D packing (`pack_egoexo_server.py`) |

The answer scoring of the checks reuses the evaluation package [`../triphys`](../triphys) (`metrics.answer_score`,
the official ACC, and `tools.SERPER_FAILED`). It is found next to this directory, or through `TRIPHYS_ROOT` (the
directory that contains `triphys/`).

## Pipeline order (`run_pipeline_for_video`)

Each stage caches its output in the per-video output directory and is skipped when that cache exists.

1. **Stage 1, extract** (`stage1_omni_info.json`). ffmpeg splits the video into clips. Gemini describes each clip, listing visual elements and observable physical quantities.
2. **Class-specific step**:
   - **MPR**: the EgoExo4D `ego_pose` body annotation `annotations/<take_uid>.json` is loaded and injected as `gt_physics`, saved to `stage1_gt_physics.json`.
   - **KPR**: Stage 1.5 calls the LLM to select `searchable_objects` and writes `stage2b_class.json`.
3. **Stage 2.5 CV** (`stage25_cv_physics.json`, `physics_extractor.run_stage25_cv`):
   - **MPR**: YOLO11n + ByteTrack object tracks, YOLO11-pose human pose (MediaPipe fallback), YOLO11n + YOLO-World reference objects for scale; EgoExo4D GT supplies the measurement frames and anchors the scale.
   - **KPR**: Gemini 5-frame scale-free estimation (angles, counts, ratios); no metric calibration.
4. **Stage 2, event graph** (`stage2_event_graph.json`). LLM, conditioned on the Stage 2.5 measurements.
5. **Stage 2.5a, physics event library** (`_shared_physics_library.json`). LLM; built once and shared by all videos.
6. **Stage 2.5b, expansion** (`stage2b_physics_expanded.json`). LLM.
7. **Stage 3, grounded QA generation** (`stage3_physics_qa.json`). The tool-using author agent writes one question per eligible tier (see below); every attempt must pass the Stage-3 checks.
8. **Stage 4, quality check** (`stage4_qa_checked.json`). A tool-using LLM judge checks understandability, answer correctness and uniqueness, the "answerable without video / without tools" flags, and sources.
9. **Stage 4b, verification and bounded repair** (`stage4b_verified.json`). Every question is verified; a failing question is repaired and re-verified, and dropped if it still fails.
10. **Stage 5, rule filter** (`filter_valid_qa.is_valid_item`). Keeps questions with `verification.status == "verified"` that pass the rule filter, and marks `valid` in `stage4b_verified.json`.

`split_videos.py` then splits the kept questions of each subtask into train and test videos.

With `skip_check=True` (`--no_check`), Stages 4, 4b and 5 are skipped and the Stage 3 questions are the output.

### Stage 3: grounded generation

`aclass_pipeline_core.stage3_qa` dispatches on `video_class`: B goes to `agentic_physics_qa_kpr.py`, everything else to
`agentic_physics_qa.py`. Tier eligibility is as follows:

- **MPR**: set by the number of reliable CV measurements; 0 / 1 / 2 of them are hidden from the question for easy / medium / hard.
- **KPR**: set by the number of `searchable_objects`; an easy / medium / hard question has 1 / 2 / 3 web constants (one per hidden spec).

For each tier the agent gets its exploration prompt plus a block of grounding requirements, and
`grounding.grounded_generate` runs at most **4 attempts**. The problems of a rejected attempt are fed back to the
agent.

Before the checks, `grounding.settle_answer` sets the type of a numeric QA from its inputs; the agent's label is not
kept. The type is TYPE_B when a range input is propagated (a web constant stated as a range, or a range the question
states), and TYPE_A otherwise. If the type changes, or the answer / `answer_range` is not printed by the code, Gemini
rewrites the answer and the solution from the code output.

An attempt is accepted only if it passes all of these checks:

| check | rule |
|---|---|
| grounded web values | every `web_search` source has `search_query`, a `url` that one of the agent's `web_search` calls returned, and a `verified_quote` copied verbatim from that result that states the value; a query that contains the value itself is re-run without it |
| search log | every `web_search` of the agent and every search of the checks is appended to `search_log.jsonl` in the video's output directory (query + results) |
| code re-execution | every `code_executor` block is re-run; its stdout replaces `key_information`; the answer, and both ends of an `answer_range`, must be printed by the last block |
| leaks | no hidden value and no web value in the question in any unit (unit-aware comparison), no tool / search wording (search, Google, look up, online, internet, code) |
| answer form | TYPE_A one number; TYPE_B an interval "low-high unit" with `answer_range` propagated from a range input; TYPE_C Yes / No with the deciding number; TYPE_D a comparison with its ratio / difference |
| inputs | every number the code uses is given by the question, the video readings or the web constants |
| range form | a query that asks for a range / tolerance / limits returns a range |
| sources | no forum / Q&A sources, no trivial constants, no spurious precision; KPR has exactly one web constant per hidden spec |
| unique-value search test | for every web constant, the Serper results of its query are extracted (two LLM extractions, one object-tagged); each value found is substituted into the code and scored with the official ACC; a value with ACC < 0.5 is a conflict unless question + video frames single out the reference value (Gemini); the value must be stated by two sites or by a standard / regulation / handbook quote; a paraphrased query must not return a conflicting value |

A web constant that fails the unique-value test is resolved within the attempt: Gemini (with the frames) proposes up
to **5** candidate queries for the same quantity of the same object, either a **requery** (only the query changes) or a
**qualify** (one number-free "Assume ..." sentence naming the material / grade / standard / condition is added to the
question). A candidate is accepted if its query returns the old value uniquely (only the query and assumption change),
or one different value uniquely (the value, code output, answer and solution are updated). If no candidate works, the
attempt is rejected.

If all 4 attempts fail and the last one failed only on the uniqueness of some (not all) of its web constants, those
constants may become **stated givens**: the question states their values, their searches are dropped, code and answer
stay unchanged. The difficulty slot rule applies: MPR keeps its difficulty and at least one web constant; a KPR
question's difficulty becomes the number of web constants left, and only if no other question of the video already
has that difficulty.

Accepted questions carry `verification.stage3` (attempts, per-constant test results, the search evidence behind every
web constant, and a signature of the question).

### Stage 4b: verification and bounded repair

`verify_qa.verify_video` verifies the questions of a video in tier order (easy, medium, hard). A question is verified
only if every check passes:

| check | what it does | fails on |
|---|---|---|
| C, code | runs the solution code | the code fails, or a TYPE_A / TYPE_B answer is not printed by it (and an answer re-derived from the output scores ACC < 0.9) |
| S, Stage-3 checks | the grounding, leak, answer-form, range-form and input checks above, on the search evidence recorded in Stage 3 | any problem |
| V, video consistency | Gemini labels every scene statement of the question on 10-16 frames; questioned statements are re-judged on up to 48 dense frames | a contradicted or absent statement, or an invented identity that changes which value applies |
| W, web well-posedness | the unique-value search test for every web constant, with the objects seen in V | any constant not unique |
| M, model well-posedness | the LLM lists up to 3 modelling choices the question leaves open, implemented as code edits and scored with the official ACC | a choice with ACC < 0.5 that a careful solver would make instead of the reference, judged from the question alone |
| CONS, cross-question consistency | the LLM compares the retrieved values with those of the already verified questions of the video | the same property of the same object has another value |
| Q, quality | the Stage-4 quality fields (re-run by the Stage-4 checker when the question changed) and the Stage-5 rule filter | rejected |
| AUDIT, strict audit | Gemini audits the whole question with the frames (leak, scene, assumption, source, physics, answer, ambiguous, unnatural), plus the deterministic answer-form check | any major defect, or no review |

C and S run first; if they fail, the rest is skipped. AUDIT runs only after all other checks pass.

A failing question gets at most **3 repair rounds** and is re-verified after each round:

- **Rounds 1-2, smallest fix** (`repair_qa.small_fix`). Gemini sees the frames, the whole question and the problems, and returns the smallest edit. Allowed edits are rewording the scene, a number-free "Assume ..." sentence, stating a missing input, a code edit with the answer following the new output, the answer type, or stating a web constant as a given (slot rule). Each fix gets up to 4 tries against deterministic checks. The code must run, and it must print the answer and range. No hidden or kept web value and no tool wording may enter the question, and "Assume" sentences carry no number. The video readings, physics area and question type are kept, and the answer form must match the type. If no fix passes, the next round is the rewrite.
- **Round 3, rewrite**. The Stage-3 author agent writes a new question of the same tier under the Stage-3 checks. It is told the problems of the failed question and the other questions of the video.

A question that still fails is kept in `stage4b_verified.json` with `verification.status = "dropped"` and its
problems. Every question carries `verification` = {status, checks, rounds, history}.

## Models

- **LLM**: DeepSeek-V3.2 (`deepseek/deepseek-v3.2` via OpenRouter). It is used for Stages 1.5, 2, 2.5a, 2.5b, 3 (author agent and text checks), 4 and the text checks of 4b (value extraction, substitution, validity, modelling choices, consistency, stated givens). The official ACC judge of `triphys/metrics.py` (TYPE_D magnitude and unreadable answers) is pointed at the same endpoint and model.
- **VLM**: `GEMINI_MODEL`, default `google/gemini-3.6-flash`. It is used for Stage 1, KPR Stage 2.5, the MPR VLM fallbacks, and every check that needs the frames: the video re-judging of web-value conflicts, requery / qualify candidates, the video consistency check, the strict audit and the smallest fix.
- All check calls run at temperature 0, at most 2 concurrently, and are cached per video in `verify_cache.jsonl`.
- **`visual_question_answering` tool**: an OpenAI-compatible endpoint set by `QWEN_BASE_URL`, `QWEN_MODEL` and `QWEN_API_KEY` (defaults `http://localhost:8089/v1`, `qwen3-omni-30b-a3b-thinking`); you must serve it yourself.
- **CV weights**: `yolo11n.pt`, `yolo11n-pose.pt` and `yolov8s-worldv2.pt` are downloaded by `ultralytics` on first use. The MediaPipe fallback needs `pose_landmarker_full.task` next to `physics_extractor.py`.
- **Video/image/audio search tools**: these need `sentence-transformers`. Without it they return empty results.

## Environment variables

| variable | used by | default |
|---|---|---|
| `OPENROUTER_API_KEY` | Gemini calls; LLM calls when `DEEPSEEK_BASE` is OpenRouter | (required) |
| `CHAT_API_BASE` | Gemini endpoint | `https://openrouter.ai/api/v1` |
| `VLM_API_KEY` | Gemini key when `CHAT_API_BASE` is not OpenRouter | `OPENROUTER_API_KEY` |
| `GEMINI_MODEL` | VLM model id | `google/gemini-3.6-flash` |
| `DEEPSEEK_BASE` | LLM endpoint | `https://openrouter.ai/api/v1` |
| `DEEPSEEK_API_KEY` | LLM key when `DEEPSEEK_BASE` is not OpenRouter | - |
| `DEEPSEEK_MODEL` | LLM model id | `deepseek/deepseek-v3.2` (use the provider's id, e.g. `deepseek-chat`, with `https://api.deepseek.com`) |
| `SERPER_API_KEYS` | `web_search` / `web_image_search` and the Stage 3 / 4b searches (comma-separated). The author agent uses `SERPER_API_KEY` if set, otherwise the first key; the checks rotate through all keys when one is exhausted | (required) |
| `SERPER_API_KEY` | single Serper key for the author agent (set by the runners) | first of `SERPER_API_KEYS` |
| `JINA_API_KEY` | `page_browser` (r.jina.ai) | empty |
| `QWEN_BASE_URL`, `QWEN_MODEL`, `QWEN_API_KEY` | `visual_question_answering` | see above |
| `TRIPHYS_ROOT` | directory that contains `triphys/` | parent of this directory |
| `EGOEXO_RAW_ROOT` | `select_videos/*egoexo*` (raw EgoExo4D: `takes.json`, `takes/`, `annotations/`) | (required there) |
| `EGOEXO_SELECTION` | selected take list written/read by the selection and packing scripts | `selected_egoexo200.json` |
| `EGOEXO_SELECTED_ROOT` | packed EgoExo4D dir: `manifest.json`, `takes/<uid>/best_exo.mp4`, `annotations/<uid>.json` | `A_class_Benchmark/egoexo4d_selected` |
| `MPR_OUTPUT_DIR` | MPR outputs `<dir>/<category>/<take_uid>/` and the shared library | `A_class_Benchmark/output` |
| `MPR_QA_DATASET` | aggregated MPR QA file written by `run_aclass.py` | `<parent of MPR_OUTPUT_DIR>/aclass_qa_dataset.json` |
| `BAAI_ROOT` | BAAI videos `<root>/<folder_id>/<video_id>.mp4` | `B_class_Benchmark/BAAI` |
| `BAAI_CANDIDATES` | KPR candidate pool written by `select_baai200_library.py`, read by `download_baai200.py` | `selected_baai200.json` |
| `BAAI_SELECTION` | KPR selection file | `baai_selected_160.json` |
| `KPR_OUTPUT_DIR` | KPR outputs `<dir>/<folder_id>/<video_id>/` | `B_class_Benchmark/outputs` |
| `SHARED_LIBRARY` | physics library reused by the KPR runner | `$MPR_OUTPUT_DIR/_shared_physics_library.json` |

## Requirements

You need `ffmpeg` on `PATH`; the scripts exit at import time without it. Install the Python packages:

```bash
pip install openai httpx tqdm numpy scipy opencv-python ultralytics mediapipe aiohttp beautifulsoup4 chardet \
            pandas sympy openpyxl pyarrow h5py pillow fpdf reportlab pyyaml
# optional (video/image/audio search tools): torch sentence-transformers
```

Run every command from this directory, because the tools write `./cache`.

## Commands

```bash
set -a; source ../.env; set +a        # keys and paths (see table above)

# 0. video selection
EGOEXO_RAW_ROOT=/data/EgoExo4D python select_videos/select_egoexo200_physics.py
EGOEXO_RAW_ROOT=/data/EgoExo4D EGOEXO_SELECTED_ROOT=/data/egoexo4d_selected \
    python select_videos/pack_egoexo_server.py                                    # writes manifest.json, copies best exo + ego_pose GT
python select_videos/select_baai200_library.py                                    # writes $BAAI_CANDIDATES (200)
BAAI_ROOT=/data/BAAI python select_videos/download_baai200.py
BAAI_ROOT=/data/BAAI python select_videos/select_baai_160.py                      # writes $BAAI_SELECTION (160)

# 1. MPR: all stages (builds the shared physics library on first run)
python run_aclass.py --parallel 4
#    Stage 3-5 over a take list (delete a take's stage3 / stage4 / stage4b json to regenerate it)
python run_aclass_batch.py --videos rerun_videos.json --results batch_results.json

# 2. KPR: all stages, reusing the MPR physics library
python run_baai_160_batch.py

# single video / single folder runner
python aclass_pipeline_core.py --videos_dir ./Videos --out_dir ./out

# 3. video-level split (per subtask)
python split_videos.py mpr "$MPR_QA_DATASET" --out_dir splits
python split_videos.py kpr bclass_qa_dataset.json --out_dir splits
```

From Python, one video:

```python
from aclass_pipeline_core import run_pipeline_for_video, load_json
lib = load_json("A_class_Benchmark/output/_shared_physics_library.json")
summary = await run_pipeline_for_video("BAAI/a00128/00907.mp4", "out/a00128/00907", shared_library=lib,
                                       video_class="B", dataset_class="B")
```

Outputs per video:

| file | content |
|---|---|
| `stage1_omni_info.json`, `stage1_gt_physics.json` (MPR), `stage2b_class.json` (KPR), `stage25_cv_physics.json`, `stage2_event_graph.json`, `stage2b_physics_expanded.json` | Stages 1-2.5 |
| `stage3_physics_qa.json` | questions that passed the Stage-3 checks, with `verification.stage3` and the per-tier attempt log (`stage3_log`) |
| `stage4_qa_checked.json` | Stage-4 quality fields |
| `stage4b_verified.json` | every question with `verification` (verified / dropped, checks, repair history) and the Stage-5 `valid` flag |
| `search_log.jsonl` | every web search of the author agent (`"source": "author"`) and of the checks (`"source": "check"`) with its results |
| `verify_cache.jsonl` | cache of the check model calls and searches (a re-run reuses it) |

The summary returned by `run_pipeline_for_video` counts `stage3_qa_pairs`, `stage3_stated_given`, `stage4_checked`,
`stage4b_verified`, `stage4b_repaired`, `stage4b_dropped` and `stage5_valid_qa`. The runners collect only the questions
marked `valid`.

## Video selection

- **MPR** (`select_egoexo200_physics.py`): EgoExo4D takes that have an `ego_pose` body annotation and last less than 60 s. Each take's atomic action descriptions are matched against keyword lists of 15 physical event types (projectile motion, collision, rotation, oscillation, friction, deformation, lifting, pouring, splashing, deceleration, balance, mechanisms, reflection, applied force, free fall). Takes are ranked by the number of event types detected, with longer takes first among ties, and the top 200 are kept. `pack_egoexo_server.py` copies each take's best exocentric view and its `ego_pose` annotation.
- **KPR**: `select_baai200_library.py` matches the caption of every Physics-aware-videos entry against keyword lists for the categories of the shared physical event library, excludes scene types without a measurable subject (ocean waves, rain, cooking, dancing, team sports, traffic), visits categories from the least to the most represented with up to 10 recordings each (ranked by precise-keyword matches), and fills the pool to 200. `select_baai_160.py` then takes one video per scene folder (the lowest video id) and one more video per folder in folder order until 160 videos are selected.

## Split

`split_videos.py` runs separately for MPR and KPR and keeps all questions of a video on one side:

1. Each video gets its modal question type, difficulty and parent physics category; the triple is its stratum.
2. A stratum with fewer than three videos is merged into one with the same type and difficulty, otherwise into one with the same type, and otherwise keeps its identity.
3. With seed 42, 30% of each stratum's videos (at least one, and at most all but one) go to the test side.
4. Single train/test video swaps are then applied greedily while they reduce the deviation of the test side's question counts per type, difficulty, parent category and type x difficulty from the test share. Every stratum with two or more videos keeps at least one video on each side.

## Notes

- The pipeline calls LLMs and web search, so a re-run produces different questions.
- Stage 3 and Stage 4b share the per-video model and search cache, so the unique-value test of an accepted constant is not re-searched in Stage 4b.
- `run_stage25_cv` runs one YOLO tracking pass per clip for both classes, to attach `track_id`/`bbox_region` hints to subjects.
