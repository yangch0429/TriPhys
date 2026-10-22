# Benchmark construction

This directory holds the code that generates the MPR (A-class, EgoExo4D) and KPR (B-class,
BAAI Physics-aware-videos) question pools: video selection, the per-video construction pipeline and the video-level
train/test split. API keys, endpoints, model ids and paths come from the environment.

The generated questions then go through the question-repair stage in [`../repair`](../repair).
[`../construction_integrated`](../construction_integrated) runs construction and the repair standards as one pipeline.

## Layout

| file | role |
|---|---|
| `aclass_pipeline_core.py` | per-video pipeline `run_pipeline_for_video()` (Stages 1-5) used by both classes; standalone per-folder runner |
| `run_aclass.py` | MPR runner over `manifest.json` (all stages, resumable, shared physics library) |
| `run_aclass_batch.py` | MPR batch re-run over a take list (4 in parallel); produced the final MPR Stage 3-5 outputs |
| `run_baai_160_batch.py` | KPR runner over the BAAI selection (4 in parallel) |
| `extract_video_info.py` | Stage 1: clip segmentation (ffmpeg) + Gemini clip descriptions / observable quantities |
| `classify_b1_b2.py` | Stage 1.5 (KPR): LLM picks the `searchable_objects` whose specs can be web-searched |
| `physics_extractor.py` | Stage 2.5 CV measurements (MPR) / scale-free VLM estimates (KPR); EgoExo4D GT loading |
| `build_event_graph.py` | Stage 2: event graph |
| `build_physical_event_library.py` | Stage 2.5a: shared physics event library |
| `expand_event_graph_with_physics.py` | Stage 2.5b: graph expansion with physics matches |
| `agentic_physics_qa.py` | Stage 3 author agent for **MPR** |
| `agentic_physics_qa_kpr.py` | Stage 3 author agent for **KPR** |
| `check_qa_quality.py` | Stage 4: LLM quality check (with tools) |
| `filter_valid_qa.py` | Stage 5: rule filter |
| `tools/` | agent tools: `web_search`, `page_browser`, `web_image_search` (Serper + Jina), `code_executor`, `visual_question_answering`, video/image/audio search |
| `select_videos/` | video selection (`select_egoexo200_physics.py`; `select_baai200_library.py` → `download_baai200.py` → `select_baai_160.py`) and EgoExo4D packing (`pack_egoexo_server.py`) |
| `split_videos.py` | video-level stratified train/test split with greedy exchange |

## Pipeline order (`run_pipeline_for_video`)

Each stage caches its output in the per-video output directory and is skipped when that cache exists.

1. **Stage 1, extract** (`stage1_omni_info.json`). ffmpeg splits the video into clips. Gemini (`GEMINI_MODEL`) describes each clip, listing visual elements and observable physical quantities.
2. **Class-specific step**:
   - **MPR**: the EgoExo4D `ego_pose` body annotation `annotations/<take_uid>.json` is loaded and injected as `gt_physics`, saved to `stage1_gt_physics.json`. It provides 3D joints, upright/measurement frames and derived GT quantities.
   - **KPR**: Stage 1.5 calls the LLM to select `searchable_objects` and writes `stage2b_class.json`.
3. **Stage 2.5 CV** (`stage25_cv_physics.json`, `physics_extractor.run_stage25_cv`):
   - **MPR**: YOLO11n + ByteTrack object tracks (`yolo11n.pt`). Human pose from YOLO11-pose (`yolo11n-pose.pt`), with MediaPipe Pose Landmarker as fallback. Reference-object detection with YOLO11n + YOLO-World (`yolov8s-worldv2.pt`) for scale calibration. EgoExo4D GT supplies the measurement frames and anchors the scale.
   - **KPR**: Gemini 5-frame scale-free estimation (`_vlm_extract_b_physics`). It uses 3 optical-flow event frames plus 2 uniform frames and requests only scale-free quantities such as angles, counts and ratios. No metric calibration is done.
4. **Stage 2, event graph** (`stage2_event_graph.json`). LLM, conditioned on the Stage 2.5 measurements.
5. **Stage 2.5a, physics event library** (`_shared_physics_library.json`). LLM. It is built once and shared by all videos; the KPR runner reuses the MPR library.
6. **Stage 2.5b, expansion** (`stage2b_physics_expanded.json`). LLM.
7. **Stage 3, QA generation** (`stage3_physics_qa.json`). A tool-using LLM author agent writes questions per difficulty tier (easy/medium/hard):
   - **MPR**: tiers are set by the number of reliable CV measurements and how many of them are hidden from the question text (0/1/2 hidden for easy/medium/hard).
   - **KPR**: tiers are set by the number of `searchable_objects` (1/2/3 or more). The agent must web-search the hidden object specs.
   - **Both classes**: if a question's text leaks a hidden value or spec, that tier is regenerated once. The retry replaces the original only if it does not leak.
8. **Stage 4, quality check** (`stage4_qa_checked.json`). A tool-using LLM judge checks understandability, answer correctness and uniqueness, the "answerable without video / without tools" flags, and sources.
9. **Stage 5, rule filter** (`filter_valid_qa.is_valid_item`). Keeps items that pass Stage 4. It rejects ambiguous object references, inconsistent answer ranges and unverifiable sources, and marks `valid` in `stage4_qa_checked.json`.

### Stage-3 author agents

Stage 3 uses one author agent per subtask. `aclass_pipeline_core.stage3_qa` dispatches on `video_class`: B goes to `agentic_physics_qa_kpr.py` and everything else to `agentic_physics_qa.py`.

- `agentic_physics_qa_kpr.py` (KPR):
  - tier eligibility comes from `searchable_objects`;
  - the prompts list the searchable objects and require `web_search` for their specs;
  - it checks the question text for leaked specs, caches web-search results and validates answer ranges.
- `agentic_physics_qa.py` (MPR):
  - tier eligibility comes from the reliable CV measurements;
  - length/distance measurements are not hidden in Medium/Hard questions.

KPR specs are retrieved by the Stage-3 agent itself, so `expanded["web_search_specs"]` stays empty.

## Models

- **LLM**: DeepSeek-V3.2 (`deepseek/deepseek-v3.2` via OpenRouter). It is used for Stages 1.5, 2, 2.5a, 2.5b, 3 and 4.
- **VLM**: `GEMINI_MODEL`, default `google/gemini-3.6-flash`. It is used for Stage 1, KPR Stage 2.5 and the MPR VLM fallbacks.
- **`visual_question_answering` tool**: an OpenAI-compatible endpoint set by `QWEN_BASE_URL`, `QWEN_MODEL` and `QWEN_API_KEY`. Defaults are `http://localhost:8089/v1` and `qwen3-omni-30b-a3b-thinking`, so you must serve this yourself.
- **CV weights**:
  - `yolo11n.pt`, `yolo11n-pose.pt` and `yolov8s-worldv2.pt` are downloaded automatically by `ultralytics` on first use.
  - The MediaPipe fallback needs `pose_landmarker_full.task` next to `physics_extractor.py`. Download it from the MediaPipe model page; it is not downloaded automatically.
- **Video/image/audio search tools**: these need `sentence-transformers`. Without it they return empty results.

## Environment variables

| variable | used by | default |
|---|---|---|
| `OPENROUTER_API_KEY` | Gemini calls; LLM calls when `DEEPSEEK_BASE` is OpenRouter | (required) |
| `CHAT_API_BASE` | Gemini endpoint | `https://openrouter.ai/api/v1` |
| `VLM_API_KEY` | Gemini key when `CHAT_API_BASE` is not OpenRouter (e.g. where OpenRouter does not serve Gemini) | `OPENROUTER_API_KEY` |
| `GEMINI_MODEL` | VLM model id | `google/gemini-3.6-flash` |
| `DEEPSEEK_BASE` | LLM endpoint | `https://openrouter.ai/api/v1` (what produced the data) |
| `DEEPSEEK_API_KEY` | LLM key when `DEEPSEEK_BASE` is not OpenRouter | - |
| `DEEPSEEK_MODEL` | LLM model id | `deepseek/deepseek-v3.2` (use the provider's id, e.g. `deepseek-chat`, with `https://api.deepseek.com`) |
| `SERPER_API_KEYS` | `web_search` / `web_image_search` (comma-separated). The first key is used; `run_aclass.py` rotates every 64 videos; `run_baai_160_batch.py` uses keys [0] and [1] for its two halves | (required) |
| `JINA_API_KEY` | `page_browser` (r.jina.ai) | empty |
| `QWEN_BASE_URL`, `QWEN_MODEL`, `QWEN_API_KEY` | `visual_question_answering` | see above |
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

The evaluation and training code reads `EGOEXO_ROOT`, which corresponds to `$EGOEXO_SELECTED_ROOT/takes`.

## Requirements

You need `ffmpeg` on `PATH`; the scripts exit at import time without it. Install the Python packages:

```bash
pip install openai tqdm numpy scipy opencv-python ultralytics mediapipe aiohttp beautifulsoup4 chardet \
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
#    final MPR Stage 3-5 re-run over a take list (delete a take's stage3/stage4 json to regenerate it)
python run_aclass_batch.py --videos rerun_videos.json --results batch_results.json

# 2. KPR: all stages, reusing the MPR physics library
python run_baai_160_batch.py

# single video / single folder runner
python aclass_pipeline_core.py --videos_dir ./Videos --out_dir ./out

# 3. video-level split (per subtask)
python split_videos.py mpr "$MPR_QA_DATASET" --out_dir splits
python split_videos.py kpr bclass_qa_dataset.json --out_dir splits
```

Outputs per video: `stage1_omni_info.json`, `stage1_gt_physics.json` (MPR), `stage2b_class.json` (KPR), `stage25_cv_physics.json`, `stage2_event_graph.json`, `stage2b_physics_expanded.json`, `stage3_physics_qa.json`, and `stage4_qa_checked.json` with `valid` flags from Stage 5. The released question files come from these outputs after the split and repair steps in `../repair`.

## Video selection

- **MPR** (`select_egoexo200_physics.py`): EgoExo4D takes that have an `ego_pose` body annotation and last less than 60 s. Each take's atomic action descriptions are matched against keyword lists of 15 physical event types (projectile motion, collision, rotation, oscillation, friction, deformation, lifting, pouring, splashing, deceleration, balance, mechanisms, reflection, applied force, free fall). Takes are ranked by the number of event types detected, with longer takes first among ties, and the top 200 are kept. `pack_egoexo_server.py` copies each take's best exocentric view and its `ego_pose` annotation.
- **KPR**: `select_baai200_library.py` matches the caption of every Physics-aware-videos entry against keyword lists for the categories of the shared physical event library, excludes scene types without a measurable subject (ocean waves, rain, cooking, dancing, team sports, traffic), visits categories from the least to the most represented with up to 10 recordings each (ranked by precise-keyword matches), and fills the pool to 200 (22 categories covered). `select_baai_160.py` then takes one video per scene folder (the lowest video id) and one more video per folder in folder order until 160 videos are selected.

## Split

`split_videos.py` runs separately for MPR and KPR and keeps all questions of a video on one side:

1. Each video gets its modal question type, difficulty and parent physics category; the triple is its stratum.
2. A stratum with fewer than three videos is merged into one with the same type and difficulty, otherwise into one with the same type, and otherwise keeps its identity.
3. With seed 42, 30% of each stratum's videos (at least one, and at most all but one) go to the test side.
4. Single train/test video swaps are then applied greedily while they reduce the deviation of the test side's question counts per type, difficulty, parent category and type x difficulty from the test share. Every stratum with two or more videos keeps at least one video on each side.

## Notes

- The pipeline calls LLMs and web search, so a re-run produces different questions. The benchmark files are in [`../data`](../data).
- `run_stage25_cv` runs one YOLO tracking pass per clip for both classes, to attach `track_id`/`bbox_region` hints to subjects.
- `run_aclass_batch.py` reads ego_pose GT from `annotations/<uid>.json`, the layout written by `pack_egoexo_server.py`.
