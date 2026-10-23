#!/usr/bin/env python3
"""Tool-use evaluation of the TriPhys models (LoRA adapter on a base model) and of the untrained base model, with the
Qwen3.5 native tool format. Tools are passed to the chat template (n1_tools.TOOLS, schema in the system turn); tool
calls are parsed from the native XML format (JSON is also accepted). --model_path none evaluates the base model without
LoRA. Each output row stores the full turns ("turns": assistant / tool messages), which the data-building scripts reuse.
Machine paths: training/scripts/srvpaths.py.

  - No oracle: the model must use real Serper web search
  - No static fallback: Serper failures are logged to *_serper_failures.json
  - Several <tool_call> blocks can be executed in one round
  - question_id is passed to Serper for failure attribution
"""
import os
os.environ["FORCE_QWENVL_VIDEO_READER"] = "decord"

import sys, json, re, io, contextlib, math, torch, argparse, asyncio, logging
from pathlib import Path
from difflib import SequenceMatcher
from transformers import AutoProcessor, Qwen3_5ForConditionalGeneration
from peft import PeftModel
from qwen_vl_utils import process_vision_info
from tqdm import tqdm
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from n1_tools import TOOLS, parse_tool_calls

# Paths (training/scripts/srvpaths.py)
_SCRIPTS = os.path.join(os.path.dirname(os.path.dirname(os.path.realpath(__file__))), 'scripts')
sys.path.insert(0, _SCRIPTS)
from srvpaths import BASE_MODEL, EGOEXO_ROOT, BAAI_ROOT, EVAL_LIB, E
sys.path.remove(_SCRIPTS)

sys.path.insert(0, EVAL_LIB)
from prompts import SYSTEM_PROMPT, build_user_prompt
from response_parser import parse_response
from metrics_final import calculate_acc_with_fallback, calculate_scs_with_fallback, calculate_ogs

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)

GEN_PARAMS = dict(do_sample=True, temperature=0.1, top_p=0.9, top_k=20, repetition_penalty=1.05)
MAX_TOKENS = 8192   # per-round limit

# Serper keys: env SERPER_API_KEYS (comma-separated) or SERPER_API_KEY
SERPER_KEYS = [k for k in os.environ.get('SERPER_API_KEYS', os.environ.get('SERPER_API_KEY', '')).split(',') if k]
_serper_key_idx = [0]

# Track questions where Serper failed — saved to output_dir at the end
_serper_failures: list = []


def query_web_search(query: str, question_id: str = "") -> str:
    """Real Serper API search; rotates keys on 429."""
    import urllib.request as _urlreq, urllib.error as _urlerr
    for _attempt in range(len(SERPER_KEYS) * 2):
        key = SERPER_KEYS[_serper_key_idx[0] % len(SERPER_KEYS)]
        try:
            payload = json.dumps({"q": query, "num": 10}).encode()
            req = _urlreq.Request(
                "https://google.serper.dev/search", data=payload,
                headers={"X-API-KEY": key, "Content-Type": "application/json"})
            with _urlreq.urlopen(req, timeout=8) as resp:
                data = json.loads(resp.read())
            snippets = [r.get("snippet", "") for r in data.get("organic", []) if r.get("snippet")]
            if snippets:
                return json.dumps({"result": f"Search results for \"{query}\": " + " | ".join(snippets[:10])})
            return json.dumps({"result": f"No results found for: {query}"})
        except _urlerr.HTTPError as e:
            if e.code in (400, 401, 402, 403, 429):   # rate limit or key out of credits
                logger.warning(f"  Serper key[{_serper_key_idx[0] % len(SERPER_KEYS)}] rate-limited, rotating...")
                _serper_key_idx[0] += 1
                continue
            _serper_failures.append({"question_id": question_id, "query": query, "error": str(e)})
            logger.warning(f"  Serper HTTP {e.code} for q={question_id!r}: {e}")
            return json.dumps({"result": f"__SERPER_FAILED__: {query}"})
        except Exception as e:
            _serper_failures.append({"question_id": question_id, "query": query, "error": str(e)})
            logger.warning(f"  Serper failed for q={question_id!r} query={query!r}: {e}")
            return json.dumps({"result": f"__SERPER_FAILED__: {query}"})
    _serper_failures.append({"question_id": question_id, "query": query, "error": "all keys exhausted"})
    logger.error(f"  All Serper keys exhausted for q={question_id!r} query={query!r}")
    return json.dumps({"result": f"__SERPER_FAILED__: {query}"})


class SFTEvaluator:
    def __init__(self, model_path: str, gpu: int = 0):
        self.device = torch.device(f"cuda:{gpu}")
        logger.info(f"Loading model from {model_path} on cuda:{gpu}")

        import glob as _glob
        self._baai_lookup = {
            Path(p).stem: p
            for p in _glob.glob(f"{BAAI_ROOT}/*/*.mp4")
        }
        logger.info(f"BAAI lookup: {len(self._baai_lookup)} videos")

        self.processor = AutoProcessor.from_pretrained(BASE_MODEL, trust_remote_code=True)
        base = Qwen3_5ForConditionalGeneration.from_pretrained(
            BASE_MODEL, trust_remote_code=True,
            torch_dtype=torch.bfloat16, device_map=f"cuda:{gpu}"
        )
        self.model = base if model_path == "none" else PeftModel.from_pretrained(base, model_path)
        self.model.eval()

        self._im_end_id = self.processor.tokenizer.convert_tokens_to_ids("<|im_end|>")
        self._video_cache = {}

    def _get_video(self, video_path: str):
        if video_path not in self._video_cache:
            vm = [{"role": "user", "content": [{"type": "video", "video": video_path, "fps": 1.0}]}]
            _, video_inputs, video_kwargs = process_vision_info(vm, return_video_kwargs=True)
            if video_inputs:
                n = video_inputs[0].shape[0]
                video_kwargs["video_metadata"] = [{"total_num_frames": n, "fps": 1.0, "frames_indices": list(range(n))}]
            KEEP = {"video_metadata"}
            video_kwargs = {k: v if k in KEEP else (v[0] if isinstance(v, list) and len(v) == 1 else v)
                            for k, v in video_kwargs.items()}
            self._video_cache[video_path] = (video_inputs, video_kwargs)
        return self._video_cache[video_path]

    def _exec_code(self, code: str) -> str:
        out = io.StringIO()
        g = {"__builtins__": __builtins__, "math": math,
             **{f: getattr(math, f) for f in dir(math) if not f.startswith("_")}}
        try:
            import numpy as np
            g["np"] = g["numpy"] = np
        except ImportError:
            pass
        # 60 s wall-clock limit per call, so that model-written code that never returns cannot stall a shard
        import signal, threading
        use_alarm = threading.current_thread() is threading.main_thread()
        def _to(signum, frame):
            raise TimeoutError("code execution exceeded 60 s")
        try:
            if use_alarm:
                old = signal.signal(signal.SIGALRM, _to); signal.alarm(60)
            with contextlib.redirect_stdout(out):
                exec(code, g)
            return out.getvalue().strip() or "No output"
        except BaseException as e:
            if isinstance(e, KeyboardInterrupt):
                raise
            return f"Error: {e}"
        finally:
            if use_alarm:
                signal.alarm(0); signal.signal(signal.SIGALRM, old)

    def _run_round(self, messages, video_inputs, video_kwargs, max_new_tokens, prefix=""):
        text = self.processor.apply_chat_template(messages, tools=TOOLS, tokenize=False, add_generation_prompt=True)
        if prefix:
            text += prefix
        inputs = self.processor(
            text=[text], images=None, videos=video_inputs,
            padding=True, return_tensors="pt", **video_kwargs
        ).to(self.device)
        L = inputs["input_ids"].shape[1]
        with torch.no_grad():
            out = self.model.generate(
                **inputs, max_new_tokens=max_new_tokens,
                eos_token_id=self._im_end_id, **GEN_PARAMS)
        raw = self.processor.tokenizer.decode(out[0, L:], skip_special_tokens=False)
        return raw.replace("<|im_end|>", "").strip(), out.shape[1] - L

    def generate(self, question: str, question_type: str, video_path: str,
                 question_id: str = "") -> tuple[str, list]:
        """Multi-turn TIR generation with multi-tool-per-round support.
        Each round can produce multiple <tool_call> blocks; all are executed and
        each result is appended as a separate tool message before the next round.
        If model never produces </answer>, raw_response is passed to LLM fallback.
        Returns (full_raw_response, tool_log).
        """
        tool_log = []
        content = ""
        self.last_turns = []
        try:
            video_inputs, video_kwargs = self._get_video(video_path)
            messages = [
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": [
                    {"type": "video", "video": video_path, "fps": 1.0},
                    {"type": "text", "text": build_user_prompt(question, question_type)}
                ]}
            ]
            prev_results_sig = None

            for rnd in range(6):
                content, ntok = self._run_round(messages, video_inputs, video_kwargs, MAX_TOKENS)
                logger.info(f"  Round {rnd}: {ntok} tok | {content[:80]!r}")

                if re.search(r"<answer>.*?</answer>", content, re.DOTALL):
                    self.last_turns = messages[2:] + [{"role": "assistant", "content": content}]
                    return content, tool_log

                # Extract ALL tool_call blocks in this round
                tc_blocks = parse_tool_calls(content)
                if not tc_blocks:
                    logger.info("  No tool_call, no answer — passing to LLM fallback")
                    break

                messages.append({"role": "assistant", "content": content})

                round_results = []
                for tool_name, tool_args in tc_blocks:
                    if tool_name == "unknown":
                        logger.error("  TC parse error")

                    if tool_name == "code_executor":
                        result = json.dumps({"result": self._exec_code(tool_args.get("code", ""))})
                    elif tool_name == "web_search":
                        query = tool_args.get("query", "") or tool_args.get("q", "")
                        result = query_web_search(query, question_id)
                    else:
                        result = json.dumps({"result": f"Tool '{tool_name}' unavailable."})

                    tool_log.append({
                        "round": rnd, "tool": tool_name, "args": tool_args,
                        "result": json.loads(result).get("result", "")[:200]
                    })
                    logger.info(f"  Tool {tool_name} ({len(tc_blocks)} this round): {result[:80]}")
                    messages.append({"role": "tool", "content": result})
                    round_results.append(result)

                # Stop if identical results as previous round (loop detection)
                sig = str(round_results)
                if sig == prev_results_sig:
                    logger.warning("  Repeated tool results — stopping")
                    break
                prev_results_sig = sig

            self.last_turns = messages[2:] + [{"role": "assistant", "content": content}]
            return content, tool_log

        except Exception as e:
            logger.error(f"generate error: {e}")
            import traceback; traceback.print_exc()
            return "", tool_log


async def evaluate_aclass(evaluator: SFTEvaluator, data_path: str, output_path: str,
                          shard_id: int = 0, n_shards: int = 1):
    """Evaluate A-class data (list of takes with qa_pairs)."""
    with open(data_path) as f:
        data = json.load(f)

    # Shard: each GPU handles a disjoint subset of top-level takes
    if n_shards > 1:
        data = [item for i, item in enumerate(data) if i % n_shards == shard_id]
        logger.info(f"Shard {shard_id}/{n_shards}: {len(data)} takes")

    results = []

    for take in tqdm(data, desc="A-class"):
        take_uid = take.get("take_uid", "")
        video_id = take.get("video_id", "")

        if take_uid:
            video_path = f"{EGOEXO_ROOT}/{take_uid}/best_exo.mp4"
        else:
            video_path = evaluator._baai_lookup.get(video_id)
            if not video_path:
                logger.warning(f"BAAI video not found: {video_id}")
                continue

        for qa in take.get("qa_pairs", []):
            if not qa.get("valid", True):
                continue
            qid = f"{take_uid or video_id}_{qa['question_type']}"
            response, tool_log = evaluator.generate(
                qa["question"], qa["question_type"], video_path, qid)
            parsed = parse_response(response)

            rd = {
                "prediction": parsed.prediction,
                "reasoning_chain": parsed.reasoning_chain,
                "ground_truth": qa["answer"],
                "answer_range": qa.get("answer_range"),
                "annotated_solution": qa.get("annotated_solution", []),
                "question": qa["question"],
                "question_type": qa["question_type"],
                # LLM fallback fields (same as the zero-shot evaluation)
                "needs_llm_eval": parsed.needs_llm_eval,
                "raw_response": response,
            }
            acc = await calculate_acc_with_fallback(rd)
            scs_res = await calculate_scs_with_fallback(rd)
            scs = scs_res.get("overall", scs_res.get("SCS", 0.0)) if isinstance(scs_res, dict) else float(scs_res or 0)
            scs_details = _scs_dims(scs_res, scs)
            ogs = calculate_ogs(acc, scs)

            results.append({
                "question_id": f"{take_uid or video_id}_{qa['question_type']}",
                "question": qa["question"],
                "question_type": qa["question_type"],
                "video_id": video_id, "take_uid": take_uid,
                "response": response,
                "parsed_answer": parsed.prediction,
                "reasoning_chain": parsed.reasoning_chain,
                "ground_truth": qa["answer"],
                "answer_range": qa.get("answer_range"),
                "annotated_solution": qa.get("annotated_solution", []),
                "sources": qa.get("sources", []),
                "ACC": acc, "SCS": scs, "OGS": ogs, "scs_details": scs_details,
                "tool_log": tool_log,
                "turns": evaluator.last_turns,
            })

    return _save_results(results, output_path, "A-class")


async def evaluate_bclass(evaluator: SFTEvaluator, data_path: str, output_path: str,
                          shard_id: int = 0, n_shards: int = 1):
    """Evaluate B-class data (list of videos, each with nested qa_pairs)."""
    with open(data_path) as f:
        raw = json.load(f)
    video_list = raw if isinstance(raw, list) else [raw]

    # Shard: each GPU handles a disjoint subset of top-level videos
    if n_shards > 1:
        video_list = [item for i, item in enumerate(video_list) if i % n_shards == shard_id]
        logger.info(f"Shard {shard_id}/{n_shards}: {len(video_list)} videos")

    results = []

    for video in tqdm(video_list, desc="B-class"):
        video_id = video.get("video_id", "")
        folder_id = video.get("folder_id", "")
        # folder path first: some BAAI file names repeat across folders, so a lookup by file stem alone can pick the
        # wrong video
        _fp = f"{BAAI_ROOT}/{folder_id}/{video_id}.mp4"
        key = video_id if video_id in evaluator._baai_lookup else folder_id
        video_path = _fp if os.path.exists(_fp) else evaluator._baai_lookup.get(key)
        if not video_path:
            import glob as _glob
            matches = _glob.glob(f"{BAAI_ROOT}/{folder_id}/{video_id}.mp4")
            video_path = matches[0] if matches else None
        if not video_path:
            logger.warning(f"BAAI video not found: {folder_id}/{video_id}")
            continue

        for qa in video.get("qa_pairs", []):
            if not qa.get("valid", True):
                continue
            qid = f"{folder_id}_{video_id}_{qa['question_type']}"
            response, tool_log = evaluator.generate(
                qa["question"], qa["question_type"], video_path, qid)
            parsed = parse_response(response)

            rd = {
                "prediction": parsed.prediction,
                "reasoning_chain": parsed.reasoning_chain,
                "ground_truth": qa["answer"],
                "answer_range": qa.get("answer_range"),
                "annotated_solution": qa.get("annotated_solution", []),
                "question": qa["question"],
                "question_type": qa["question_type"],
                # LLM fallback fields (same as the zero-shot evaluation)
                "needs_llm_eval": parsed.needs_llm_eval,
                "raw_response": response,
            }
            acc = await calculate_acc_with_fallback(rd)
            scs_res = await calculate_scs_with_fallback(rd)
            scs = scs_res.get("overall", scs_res.get("SCS", 0.0)) if isinstance(scs_res, dict) else float(scs_res or 0)
            scs_details = _scs_dims(scs_res, scs)
            ogs = calculate_ogs(acc, scs)

            results.append({
                "question_id": f"{folder_id}_{video_id}_{qa['question_type']}",
                "question": qa["question"],
                "question_type": qa["question_type"],
                "folder_id": folder_id, "video_id": video_id,
                "response": response,
                "parsed_answer": parsed.prediction,
                "reasoning_chain": parsed.reasoning_chain,
                "ground_truth": qa["answer"],
                "answer_range": qa.get("answer_range"),
                "annotated_solution": qa.get("annotated_solution", []),
                "sources": qa.get("sources", []),
                "ACC": acc, "SCS": scs, "OGS": ogs, "scs_details": scs_details,
                "tool_log": tool_log,
                "turns": evaluator.last_turns,
            })

    return _save_results(results, output_path, "B-class")


SCS_DIMS = ("S_obs", "S_formula", "S_derive", "S_conclude")


def _scs_dims(scs_res, scs):
    """keep the four SCS judge dimensions per QA (same shape as scs_details of the zero-shot evaluation);
    None when the judge returned no details (e.g. no reasoning chain and no LLM fallback)."""
    d = scs_res.get("details") if isinstance(scs_res, dict) else None
    if not isinstance(d, dict) or not all(k in d for k in SCS_DIMS):
        return None
    return {"overall": scs, **{k: float(d[k]) for k in SCS_DIMS}}


def _dim_means(results):
    rows = [r["scs_details"] for r in results if r.get("scs_details")]
    return {k: (sum(r[k] for r in rows) / len(rows) if rows else None) for k in SCS_DIMS} | {"n_with_details": len(rows)}


def _save_results(results, output_path, label):
    if not results:
        logger.error(f"No results for {label}")
        return {}
    avg_acc = sum(r["ACC"] for r in results) / len(results)
    avg_scs = sum(r["SCS"] for r in results) / len(results)
    avg_ogs = sum(r["OGS"] for r in results) / len(results)

    # Mark questions that had Serper failures for later re-eval
    failed_qids = {f["question_id"] for f in _serper_failures}
    for r in results:
        r["serper_failed"] = r.get("question_id", "") in failed_qids

    out = {"label": label, "n": len(results),
           "summary": {"ACC": avg_acc, "SCS": avg_scs, "OGS": avg_ogs, "SCS_dims": _dim_means(results)},
           "serper_failures": len(failed_qids),
           "results": results}
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(out, f, indent=2, ensure_ascii=False)
    logger.info(f"{label}: ACC={avg_acc:.4f} SCS={avg_scs:.4f} OGS={avg_ogs:.4f} → {output_path}")

    # Save Serper failures as separate file for re-eval
    if _serper_failures:
        fail_path = output_path.replace(".json", "_serper_failures.json")
        with open(fail_path, "w", encoding="utf-8") as f:
            json.dump(_serper_failures, f, indent=2, ensure_ascii=False)
        logger.warning(f"  Serper failures saved → {fail_path} ({len(_serper_failures)} queries)")

    # Failure analysis summary
    serper_fail_qs = [r for r in results if r.get("serper_failed")]
    code_fail = [r for r in results if any(
        t["tool"] == "code_executor" and "Error" in t["result"] for t in r.get("tool_log", []))]
    no_tool = [r for r in results if not r.get("tool_log")]
    logger.info(f"  Serper-failed questions: {len(serper_fail_qs)}/{len(results)}")
    logger.info(f"  code_executor errors: {len(code_fail)}/{len(results)}")
    logger.info(f"  no tool calls: {len(no_tool)}/{len(results)}")
    return out


async def main():
    global BASE_MODEL
    p = argparse.ArgumentParser()
    p.add_argument("--model_path", required=True, help="LoRA checkpoint path, or none for the base model")
    p.add_argument("--class", dest="cls", choices=["A", "B", "both"], default="both")
    p.add_argument("--aclass_data", default=E + "/data/f10/aclass_qa_test.json")
    p.add_argument("--bclass_data", default=E + "/data/f10/bclass_qa_test.json")
    p.add_argument("--output_dir", default=E + "/results/a800_eval")
    p.add_argument("--gpu", type=int, default=0)
    p.add_argument("--base_model", default=BASE_MODEL)
    p.add_argument("--shard_id",  type=int, default=0,
                   help="Shard index for parallel eval (0-indexed)")
    p.add_argument("--n_shards",  type=int, default=1,
                   help="Total shards for parallel eval (1 = no sharding)")
    args = p.parse_args()

    BASE_MODEL = args.base_model

    evaluator = SFTEvaluator(args.model_path, args.gpu)
    tag = "base_none" if args.model_path == "none" else Path(args.model_path).parent.name + "_" + Path(args.model_path).name
    shard_suffix = f"_shard{args.shard_id}" if args.n_shards > 1 else ""

    tasks = []
    if args.cls in ("A", "both"):
        out_a = f"{args.output_dir}/{tag}_aclass{shard_suffix}.json"
        tasks.append(evaluate_aclass(evaluator, args.aclass_data, out_a,
                                     args.shard_id, args.n_shards))
    if args.cls in ("B", "both"):
        out_b = f"{args.output_dir}/{tag}_bclass{shard_suffix}.json"
        tasks.append(evaluate_bclass(evaluator, args.bclass_data, out_b,
                                     args.shard_id, args.n_shards))

    for t in tasks:
        await t

asyncio.run(main())
