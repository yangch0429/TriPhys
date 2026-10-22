# Training (Stage I SFT, Stage II DPO) and evaluation of trained models

The scripts in this directory are the ones that trained and evaluated the TriPhys models. They are copies of the
training-server code. The release edits are limited to the following:

- Machine paths come from `scripts/srvpaths.py` (environment variables). Every other path is relative to the work directory.
- Keys, endpoints and model ids are read from environment variables (`.env.example`).

"A" / "A-class" in the code means MPR and "B" / "B-class" means KPR. The dates in docstrings and comments are in 2026 (MM-DD).

## Layout

| Here | On the training server | Content |
|---|---|---|
| `evallib/` | `experiment_right/` | Evaluation library used by every evaluation and sampling run: `prompts.py` (system / user prompt), `response_parser.py`, `metrics_final.py` (ACC: rules + gpt-4o-mini fallback; SCS: DeepSeek-V4-Flash), `llm_two_step_eval.py` |
| `n1/` | `experiment_paper/fc/training/n1/` | `eval_n1.py`: evaluation of a trained model (native Qwen tool calls `web_search` / `code_executor`, at most 6 rounds); `eval_n1_sample.py`: the same loop for sampling train QAs (`GEN_TEMP`); `train_sft_n1.py` / `train_sft_n1_mp.py`: Stage-I LoRA SFT (DDP / one process over several GPUs); `n1_tools.py` |
| `scripts/` | `experiment_paper/scripts/` | Stage-I data (`p1_*`), Stage-II data and DPO (`p2_*`, `train_dpo_p2*.py`), shell chains, `run_det.py` |
| `teacher_pilot/` | `experiment_paper/fc/training/teacher_pilot/` | Gemini teacher with real tools that never sees the GT (`teacher_full.py`, built on `teacher_pilot.py`) |
| `setup_work.sh`, `run_4b.sh`, `p1_afill_data.sh` | (none) | Work-directory setup and the commands in the order they ran (see below) |

`scripts/` also holds the older copy of the metrics library (`metrics_final.py`, `llm_two_step_eval.py`, `prompts.py`,
`response_parser.py`). Its SCS judge is gpt-4o-mini. Python resolves `import metrics_final` by `sys.path` order, as it did on the server:

- `n1/eval_n1*.py`, `p2_branch.py`, `p2_online.py`, `p1_commit.py` and `teacher_*.py` put `evallib/` first. They score with the Flash SCS judge.
- `p2_pairs.py` / `p2_pairs_full.py` import from `scripts/`. They re-score only rollouts whose judge call returned `None`.

## Setup

```bash
pip install -r requirements.txt            # Transformers 5.17.0, PEFT 0.20.0, fla 0.3.2 (Qwen3.5 Gated DeltaNet kernels)
set -a; source .env; set +a                # EGOEXO_ROOT, BAAI_ROOT, OPENROUTER_API_KEY, DEEPSEEK_API_KEY, SERPER_API_KEYS, ...
export TRIPHYS_WORK=/path/to/work          # default: <repo>/work
bash training/setup_work.sh
```

`setup_work.sh` creates the work directory. All scripts run from it (`cd $TRIPHYS_WORK`):

- `scripts` links to `training/scripts`.
- `fc/training` links to `training`.
- The released split is copied to `data/f10/{aclass,bclass}_qa_{train,test}.json`: MPR / KPR, 300 train / 110 test QAs.

Base models default to the Hugging Face ids (`BASE_MODEL=Qwen/Qwen3.5-4B`, `B9=Qwen/Qwen3.5-9B`, `B27=Qwen/Qwen3.6-27B`). A local directory can be set instead. `PY` / `TR` select the python / torchrun binaries.

## Qwen3.5-4B: Stage I and Stage II

`run_4b.sh` lists the commands in the order they ran. It needs 8 GPUs with 80 GB each.

**Stage I: GT-verified trajectories with rewritten answer blocks → SFT (`comb1_1e4`)**

1. `p1_selfsample.sh`: base-model sampling of the 300 train QAs (T 0.7, 4 repeats, `eval_n1_sample.py`).
2. `p1_rft_build.py`: GT verifier. It keeps trajectories with ACC ≥ .9, an `<answer>`, at least one tool call and no Serper failure, at most `--per_qa` per QA. `p1_rft_balance.py` balances classes and types.
3. `p1_rft_chain.sh` trains on the result (`train_sft_n1.py`: LoRA r 32, 3 epochs, lr 1e-4, 4 GPUs × accum 4) and evaluates it. The resulting model samples again (`p1_selfsample2.sh`).
4. `teacher_full.py` + `p1_gemini_conv.py`: Gemini teacher trajectories for QAs that no self-sample solved.
5. `p1_rescore_samples.py`: ACC re-score of sampled rows below .9 (judge-call failures during sampling).
6. `p1_rewrite.py`: Gemini rewrites the `<answer>` block into a step chain. The trajectory before it is unchanged. A rewrite is kept only if its Flash SCS is ≥ max(.75, the original SCS); otherwise the original stays.
7. `p1_commit.py` / `p1_commit_build.py`: commit continuations. The prefix is the first 2 tool rounds of a stuck self-sample (no `<answer>`). A one-off note asks the Stage-I model to commit to an estimate, and the model writes the rest. A continuation is kept only if ACC ≥ .9 and the note is not mentioned. The note is removed from the stored trajectory.
8. `rwf_comb1` = the three rewritten sets, then SFT → `comb1_1e4`.
9. `p1_afill_data.sh`: MPR refill pool `afill_rwf` (more samples for MPR train QAs not covered by `rwf_comb1`). This pool is used only as fallback *chosen* trajectories in Stage II.

**Stage II: on-policy rollouts → full-trajectory pairs → DPO (`p2_full1`)**

10. `p2_worker.sh` × 16 runs `p2_online.py` (comb1_1e4 rollouts, T 0.7, prefix branching with K 4 continuations, shared QA queue).
11. `p2_pairs_full.py` builds pairs, at most 2 per QA:
    - chosen: comb1's own correct rollout (ACC ≥ .9, uses the reference value, SCS ≥ .75); otherwise a Stage-I trajectory of the same QA (`rwf_comb1`, `afill_rwf`).
    - rejected: a wrong (ACC < .5) or unanswered rollout of the same QA.
12. `train_dpo_p2.py`: the SFT model is merged and used as both init and frozen reference. Training is length-normalised DPO, scored from the first token where chosen and rejected differ, with A/B class balance; β 0.1, lr 5e-6, 2 epochs.
13. `p2_reeval_n1.sh p2_full1`: evaluation on the 110 test QAs.

Qwen3.5-9B and Qwen3.6-27B are trained on the 4B data (`rwf_comb1`, `dpo_p2_full1`) with `scripts/p9_chain.sh` and
`scripts/p27_chain.sh`.

## Evaluating a trained model

```bash
python fc/training/n1/eval_n1.py --model_path outputs/phase1/comb1_1e4/final_model --base_model Qwen/Qwen3.5-4B \
    --output_dir results/comb1 --gpu 0 --n_shards 8 --shard_id 0      # one process per shard, then merge
```

`scripts/p1_rft_chain.sh` and `scripts/p2_reeval_n1.sh` show the sharded run and the merge (`run_det.merge`). For a DPO
model, `--base_model` is the merged SFT model (`outputs/phase1/comb1_1e4/merged`) and `--model_path` is the DPO adapter. The test data default to
`data/f10/*_qa_test.json`. Zero-shot baselines (no tools) are evaluated with `evaluation/`.

## Reproducibility

The training data are model samples (T 0.7) scored by API judges, and web search results change over time. A rerun therefore
produces data of the same kind, not the same rows. The paper's runs had 209 Stage-I trajectories (MPR 90 / KPR 119) and 143 DPO pairs.

On the server, each step read the outputs of the step before it. Some of those outputs were intermediate versions:

- An earlier rewrite pass (`p1_rewrite.py --prev`) was used as a cache.
- The commit continuations were sampled with an earlier Stage-I model.
- The MPR coverage for `afill_rwf` was computed against a superset of `rwf_comb1` with the same MPR QAs.

`run_4b.sh` runs every step from its inputs in this repository.
