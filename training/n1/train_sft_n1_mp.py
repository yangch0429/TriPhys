"""MP variant (10/20, 27B): = train_sft_n1.py, but one process with the model split over all visible GPUs
(device_map auto, 60 GiB per GPU) because a full 27B replica + 16k activations does not fit one A800 under DDP.
Run with python, not torchrun; --grad_accum keeps the effective batch. Loss / data / LoRA unchanged.

n1 SFT (10/14): train_sft_v29.py with the Qwen3.5 native tool format and a loss weight on reasoning tokens.
Changes vs v29: apply_chat_template(..., tools=n1_tools.TOOLS) (assistant tool calls given as `tool_calls`, rendered as
native XML by the template); --think_w: loss weight of tokens inside <think>...</think> of assistant turns (1.0 = v29);
--grad_accum. Everything else (LoRA targets, masking, video handling, LabelOnlyTrainer) unchanged.
"""
#!/usr/bin/env python3
"""
Fixed SFT training for Qwen3.5-4B VL with tool use.

Key fixes over previous version:
1. enable_thinking=False  → model outputs directly in Step/tool_call/answer format
2. Proper label masking   → only assistant turns contribute to loss
3. Real video loading     → visual tokens fed to model
4. Multi-turn support     → system/user/assistant/tool/assistant format

Usage:
  python train_sft_video_fixed.py --data DATA_JSON --output OUTPUT_DIR [--max_samples N] [--epochs E]
"""

import os, sys, json, re, argparse, logging
from pathlib import Path
from dataclasses import dataclass
from typing import List, Dict, Optional, Any

import torch
from torch.utils.data import Dataset, DataLoader
from transformers import Trainer, TrainingArguments, AutoProcessor, TrainerCallback, TrainerState, TrainerControl

os.environ['FORCE_QWENVL_VIDEO_READER'] = 'decord'
_SCRIPTS = os.path.join(os.path.dirname(os.path.dirname(os.path.realpath(__file__))), 'scripts')
sys.path.insert(0, _SCRIPTS)
from srvpaths import BASE_MODEL, EGOEXO_ROOT, BAAI_ROOT, EVAL_LIB, E  # noqa: E402
sys.path.insert(1, BASE_MODEL)

logging.basicConfig(level=logging.INFO, format='%(asctime)s %(levelname)s %(message)s')
logger = logging.getLogger(__name__)

MODEL_PATH = BASE_MODEL
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from n1_tools import TOOLS  # noqa: E402

# 10/20 MP: with the model split over several GPUs, autograd runs backward in one thread per device, and the fla Triton
# kernels can autotune concurrently. Triton 3.1's Autotuner keeps the call args in self.nargs and sets it to None at
# the end of run(), so a second thread crashes ('NoneType' object is not a mapping in _bench). Make nargs thread-local.
# (A global lock around run() deadlocks: the benchmark synchronizes a device whose stream waits on another device.)
import threading as _thr
import triton.runtime.autotuner as _at
_tl = _thr.local()


def _nargs_get(self):
    return getattr(_tl, 'd', {}).get(id(self))


def _nargs_set(self, v):
    if not hasattr(_tl, 'd'):
        _tl.d = {}
    _tl.d[id(self)] = v


_at.Autotuner.nargs = property(_nargs_get, _nargs_set)

THINK_W = 1.0

# ── Video path resolver ────────────────────────────────────────────────────────

def resolve_video_path(path_str: str) -> Optional[str]:
    """Convert training data video paths to real server paths."""
    if 'best_exo' in path_str or 'video_best_exo' in path_str:
        uid = re.search(r'([0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12})', path_str)
        if uid:
            return f'{EGOEXO_ROOT}/{uid.group(1)}/best_exo.mp4'
    if 'videos/a' in path_str:
        m = re.search(r'videos/(a\d+)[_\s]+(\d+)', path_str)
        if m:
            return f'{BAAI_ROOT}/{m.group(1)}/{m.group(2).zfill(5)}.mp4'
    return None


def fix_messages_video(messages: List[Dict]) -> List[Dict]:
    """Replace <video>path</video> text with proper video dict for processor.
    Uses fps=1.0 exactly as experiment_right does.
    """
    fixed = []
    for msg in messages:
        role = msg['role']
        content = msg.get('content', '')

        if role == 'user' and '<video>' in content:
            match = re.search(r'<video>([^<]+)</video>', content)
            if match:
                raw_path = match.group(1).strip()
                real_path = resolve_video_path(raw_path)
                if real_path and Path(real_path).exists():
                    text_part = re.sub(r'<video>[^<]+</video>', '', content).strip()
                    fixed.append({
                        'role': role,
                        'content': [
                            {'type': 'video', 'video': real_path, 'fps': 1.0},
                            {'type': 'text', 'text': text_part},
                        ]
                    })
                    continue
                else:
                    logger.warning(f'Video not found: {raw_path} -> {real_path}')
        fixed.append(msg)
    return fixed


# ── Label masking ──────────────────────────────────────────────────────────────

def mask_non_assistant_labels(input_ids: torch.Tensor, tokenizer) -> torch.Tensor:
    """
    Mask all tokens except assistant response content.
    Only tokens between <|im_start|>assistant\n ... <|im_end|> are unmasked.
    """
    labels = torch.full_like(input_ids, -100)
    ids = input_ids.tolist()

    im_start = tokenizer.convert_tokens_to_ids('<|im_start|>')
    im_end   = tokenizer.convert_tokens_to_ids('<|im_end|>')

    # "assistant\n" tokenized without special tokens
    asst_ids = tokenizer.encode('assistant\n', add_special_tokens=False)

    i = 0
    n = len(ids)
    while i < n:
        if ids[i] == im_start and i + len(asst_ids) < n:
            # Check if followed by "assistant\n"
            if ids[i+1 : i+1+len(asst_ids)] == asst_ids:
                # Unmask from after "assistant\n" until <|im_end|>
                start = i + 1 + len(asst_ids)
                j = start
                while j < n and ids[j] != im_end:
                    j += 1
                # Include <|im_end|> in labels (model must learn to stop)
                end = j + 1 if j < n else j
                labels[start:end] = input_ids[start:end]
                i = end
                continue
        i += 1

    return labels


def think_weights(input_ids: torch.Tensor, labels: torch.Tensor, tokenizer) -> torch.Tensor:
    """THINK_W for label tokens inside <think> ... </think> (the closing tag itself keeps weight 1), else 1"""
    w = torch.ones(len(input_ids), dtype=torch.float32)
    if THINK_W == 1.0:
        return w
    t0, t1 = tokenizer.convert_tokens_to_ids('<think>'), tokenizer.convert_tokens_to_ids('</think>')
    inside = False
    for i, t in enumerate(input_ids.tolist()):
        if t == t0:
            inside = True
        elif t == t1:
            inside = False
        elif inside and labels[i] != -100:
            w[i] = THINK_W
    return w


# ── Dataset ────────────────────────────────────────────────────────────────────

class VideoSFTDataset(Dataset):
    def __init__(self, data_path: str, processor, max_length: int = 16384,
                 max_samples: Optional[int] = None):
        with open(data_path) as f:
            raw = json.load(f)

        if max_samples:
            raw = raw[:max_samples]

        self.processor = processor
        self.max_length = max_length

        # Lazy loading: store message lists only, validate video paths at init.
        # Video frames are decoded on demand in __getitem__ to avoid RAM OOM.
        self.samples = []
        skipped = 0
        for i, item in enumerate(raw):
            msgs = fix_messages_video(item['messages'])
            ok = True
            for msg in msgs:
                if msg.get('role') == 'user':
                    content = msg.get('content', '')
                    if isinstance(content, list):
                        for part in content:
                            if part.get('type') == 'video' and not Path(part.get('video', '')).exists():
                                logger.warning(f'Sample {i}: video not found: {part.get("video")}')
                                ok = False
                                break
            if ok:
                self.samples.append(msgs)
            else:
                skipped += 1

        logger.info(f'Dataset: {len(self.samples)} valid samples ({skipped} path-invalid) from {len(raw)}')

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        msgs = self.samples[idx]
        try:
            result = self._process_sample(msgs, idx)
            if result is None:
                # Too long or no learnable tokens — try next sample
                for offset in range(1, min(10, len(self.samples))):
                    alt = (idx + offset) % len(self.samples)
                    r = self._process_sample(self.samples[alt], alt)
                    if r is not None:
                        return r
                raise RuntimeError(f'All nearby samples invalid near idx={idx}')
            return result
        except Exception as e:
            logger.error(f'Sample {idx} failed in __getitem__: {e}')
            raise

    def _process_sample(self, messages: List[Dict], idx: int):
        from qwen_vl_utils import process_vision_info

        # enable_thinking=True: model learns to think before acting.
        # With add_generation_prompt=False, no think tokens are injected — the
        # training labels are the original assistant content (tool calls + answer).
        # At inference the base model generates <think>...</think> naturally,
        # which is what we want: reasoning about tool use and logic chain.
        text = self.processor.apply_chat_template(
            messages,
            tools=TOOLS,
            tokenize=False,
            add_generation_prompt=False,
            enable_thinking=True,
        )

        # Extract visual inputs
        img_inputs, vid_inputs, vid_kwargs = process_vision_info(
            messages, return_video_kwargs=True
        )

        has_video = vid_inputs is not None and len(vid_inputs) > 0

        if has_video:
            # Token-budget-aware frame subsampling (reviewer recommendation).
            # NO fixed MAX_FRAMES cap — consistent with experiment_right evaluation.
            # Reserve text_budget tokens for prompt + target; visual tokens get the rest.
            # At ~256 tokens/frame: max_visual_frames = (max_length - text_budget) // 256
            TEXT_BUDGET = 6144    # tokens reserved for text (system + user + assistant)
            TOKENS_PER_FRAME = 256
            max_visual_frames = max(1, (self.max_length - TEXT_BUDGET) // TOKENS_PER_FRAME)
            n_raw = vid_inputs[0].shape[0]

            if n_raw > max_visual_frames:
                import numpy as np
                indices = np.linspace(0, n_raw - 1, max_visual_frames, dtype=int)
                vid_inputs = [vid_inputs[0][indices]]
                logger.debug(f'Sample {idx}: subsampled {n_raw} -> {max_visual_frames} frames')

            n_frames = vid_inputs[0].shape[0]
            KEEP = {'video_metadata'}
            vid_kwargs = {
                k: v if k in KEEP else (v[0] if isinstance(v, list) and len(v) == 1 else v)
                for k, v in vid_kwargs.items()
            }
            vid_kwargs['video_metadata'] = [{
                'total_num_frames': n_frames, 'fps': 1.0,
                'frames_indices': list(range(n_frames)),
            }]

            inputs = self.processor(
                text=[text], images=img_inputs, videos=vid_inputs,
                padding=False, return_tensors='pt', **vid_kwargs
            )
        else:
            inputs = self.processor(
                text=[text], padding=False, return_tensors='pt'
            )

        input_ids = inputs['input_ids'][0]
        if len(input_ids) > self.max_length:
            logger.warning(f'Sample {idx}: {len(input_ids)} tokens > {self.max_length}, skipping')
            return

        labels = mask_non_assistant_labels(input_ids, self.processor.tokenizer)

        # Reject sample if no assistant tokens to learn from
        if (labels != -100).sum() == 0:
            logger.warning(f'Sample {idx}: zero learnable tokens after masking, skipping')
            return

        entry = {
            'input_ids': input_ids,
            'attention_mask': inputs['attention_mask'][0],
            'labels': labels,
            'loss_w': think_weights(input_ids, labels, self.processor.tokenizer),
        }
        if has_video and 'pixel_values_videos' in inputs:
            entry['pixel_values_videos'] = inputs['pixel_values_videos']
            entry['video_grid_thw'] = inputs['video_grid_thw']
        if 'mm_token_type_ids' in inputs:
            entry['mm_token_type_ids'] = inputs['mm_token_type_ids'][0]

        return entry


# ── Format monitoring callback ─────────────────────────────────────────────────

class FormatCheckCallback(TrainerCallback):
    """
    After each epoch, runs inference on 1 probe sample and logs 4 format metrics:
      tool_call_ok       – output contains valid <tool_call>{...}</tool_call> JSON
      answer_tag_ok      – output contains <answer>...</answer>
      parse_ok           – parse_response() extracts a prediction (not None)
      reasoning_chain_ok – parse_response() extracts a reasoning_chain

    These are the 4 things the model MUST learn:
    1. When to call tools and in which format
    2. When to wrap the final answer in <answer>
    3. That the answer is parseable (prediction found)
    4. That the reasoning chain inside <answer> is in Reasoning chain: / Final answer: format

    Log lines look like:
      [FormatCheck epoch 1.0] tool_call=1 answer_tag=1 parse_ok=0 reasoning_chain=0 | ...
    """
    def __init__(self, dataset, processor, eos_ids, check_every_n_epochs: int = 1):
        self.dataset = dataset
        self.processor = processor
        self.eos_ids = eos_ids
        self.check_every = check_every_n_epochs

    def on_epoch_end(self, args, state: TrainerState, control: TrainerControl,
                     model=None, **kwargs):
        # In DDP, only rank 0 runs this check to avoid redundant generation and log spam
        if args.local_rank not in (-1, 0):
            return
        epoch = state.epoch or 0
        if round(epoch) % self.check_every != 0:
            return
        if len(self.dataset) == 0:
            return

        try:
            sample = self.dataset[0]  # lazy: decodes 1 video on demand
        except Exception as e:
            logger.warning(f'[FormatCheck] failed to get sample 0: {e}')
            return
        model.eval()
        try:
            self._run_check(model, sample, epoch)
        except Exception as e:
            logger.warning(f'[FormatCheck] failed: {e}')
        finally:
            model.train()

    def _run_check(self, model, sample: dict, epoch: float):
        import re as _re
        sys.path.insert(0, EVAL_LIB)
        from response_parser import parse_response

        # The prompt is everything up to (but not including) the first learnable token.
        labels = sample['labels']
        first_label_positions = (labels != -100).nonzero(as_tuple=True)[0]
        if len(first_label_positions) == 0:
            logger.warning('[FormatCheck] sample has no learnable tokens')
            return
        cut = first_label_positions[0].item()  # first assistant token position

        device = next(model.parameters()).device
        prompt_ids = sample['input_ids'][:cut].unsqueeze(0).to(device)
        prompt_mask = sample['attention_mask'][:cut].unsqueeze(0).to(device)

        extra = {}
        if 'mm_token_type_ids' in sample:
            extra['mm_token_type_ids'] = sample['mm_token_type_ids'][:cut].unsqueeze(0).to(device)
        if 'pixel_values_videos' in sample:
            extra['pixel_values_videos'] = sample['pixel_values_videos'].to(device)
            extra['video_grid_thw'] = sample['video_grid_thw'].to(device)

        with torch.no_grad():
            gen = model.generate(
                input_ids=prompt_ids,
                attention_mask=prompt_mask,
                max_new_tokens=512,      # short — just need to see the format
                eos_token_id=self.eos_ids,
                do_sample=False,         # greedy — deterministic monitoring
                **extra,
            )

        trimmed = gen[0][cut:]
        raw = self.processor.tokenizer.decode(trimmed, skip_special_tokens=False)
        raw = raw.rstrip().removesuffix('<|im_end|>').rstrip()
        # enable_thinking=True: model generates <think>...</think> naturally;
        # no manual prepend needed — parse_response finds </think> in the output.

        # --- Measure 4 metrics ---
        tool_call_ok = bool(_re.search(
            r'<tool_call>\s*\{.*?"name"\s*:', raw, _re.DOTALL))
        answer_tag_ok = '<answer>' in raw
        parsed = parse_response(raw)
        parse_ok = parsed.prediction is not None
        reasoning_chain_ok = parsed.reasoning_chain is not None

        logger.info(
            f'[FormatCheck epoch {epoch:.1f}] '
            f'tool_call={int(tool_call_ok)} '
            f'answer_tag={int(answer_tag_ok)} '
            f'parse_ok={int(parse_ok)} '
            f'reasoning_chain={int(reasoning_chain_ok)} '
            f'| parse_status={parsed.status} '
            f'| out[:80]={raw[20:100]!r}'
        )



class LabelOnlyTrainer(Trainer):
    """lm_head only at positions whose next token is a label (logits_to_keep = index tensor)"""
    def compute_loss(self, model, inputs, return_outputs=False, num_items_in_batch=None):
        labels = inputs.pop('labels')
        loss_w = inputs.pop('loss_w')
        shift = labels[:, 1:]
        pos = (shift[0] != -100).nonzero(as_tuple=True)[0]
        out = model(**inputs, logits_to_keep=pos.tolist())  # list: accelerate hooks move tensor kwargs to cuda:0
        tgt = shift[0, pos].to(out.logits.device)
        ce = torch.nn.functional.cross_entropy(out.logits[0].float(), tgt, reduction='none')
        loss = (ce * loss_w[0, 1:][pos].to(ce.device)).sum()
        # same normalisation as Trainer.compute_loss: / num_items_in_batch if the model takes loss kwargs, else token mean
        # (training_step then divides by the accumulation steps)
        if self.model_accepts_loss_kwargs and num_items_in_batch is not None:
            loss = loss / num_items_in_batch
        else:
            loss = loss / max(1, tgt.numel())
        return (loss, out) if return_outputs else loss


def collate_fn(batch):
    """Pad batch to same length."""
    max_len = max(x['input_ids'].shape[0] for x in batch)
    has_video = 'pixel_values_videos' in batch[0]
    has_mm_ids = 'mm_token_type_ids' in batch[0]

    out = {
        'input_ids':      torch.zeros(len(batch), max_len, dtype=torch.long),
        'attention_mask': torch.zeros(len(batch), max_len, dtype=torch.long),
        'labels':         torch.full((len(batch), max_len), -100, dtype=torch.long),
        'loss_w':         torch.ones(len(batch), max_len, dtype=torch.float32),
    }
    if has_mm_ids:
        out['mm_token_type_ids'] = torch.zeros(len(batch), max_len, dtype=torch.long)

    for i, x in enumerate(batch):
        l = x['input_ids'].shape[0]
        out['input_ids'][i, :l]      = x['input_ids']
        out['attention_mask'][i, :l] = x['attention_mask']
        out['labels'][i, :l]         = x['labels']
        out['loss_w'][i, :l]         = x['loss_w']
        if has_mm_ids:
            out['mm_token_type_ids'][i, :l] = x['mm_token_type_ids']

    if has_video:
        out['pixel_values_videos'] = batch[0]['pixel_values_videos']
        out['video_grid_thw']      = batch[0]['video_grid_thw']

    return out


# ── Main ───────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data',        default=E + '/data/sft_training_data_fixed.json')
    parser.add_argument('--output',      default=E + '/outputs/sft_v2')
    parser.add_argument('--base_model',  default=MODEL_PATH)
    parser.add_argument('--max_samples', type=int, default=None)
    parser.add_argument('--max_length',  type=int, default=16384,  # reviewer: 16K for long videos
                        help='Max token length. Visual tokens ~256/frame; budget-aware subsampling applied.')
    parser.add_argument('--epochs',      type=int, default=5)
    parser.add_argument('--lr',          type=float, default=2e-4)
    parser.add_argument('--lora_r',      type=int, default=32)
    parser.add_argument('--save_steps',  type=int, default=10,
                        help='Save checkpoint every N steps (enables crash recovery).')
    parser.add_argument('--resume',      action='store_true',
                        help='Resume from latest checkpoint in output dir.')
    parser.add_argument('--think_w',     type=float, default=1.0, help='loss weight of reasoning tokens')
    parser.add_argument('--grad_accum',  type=int, default=4)
    args = parser.parse_args()
    global THINK_W
    THINK_W = args.think_w

    logger.info(f'Loading processor from {args.base_model}')
    from transformers import Qwen3VLProcessor
    processor = Qwen3VLProcessor.from_pretrained(args.base_model, trust_remote_code=True)

    # Load model FIRST to claim GPU memory immediately (before video loading).
    # This prevents VLLM/other processes from reclaiming the GPU during the
    # CPU-bound video decoding phase (~3-5 min for 100+ samples).
    logger.info(f'Loading model from {args.base_model}')
    from transformers import Qwen3_5ForConditionalGeneration
    from peft import LoraConfig, get_peft_model

    # No device_map='auto' — incompatible with DDP (torchrun).
    # Trainer handles device placement (single-GPU: cuda:0; DDP: cuda:LOCAL_RANK).
    model = Qwen3_5ForConditionalGeneration.from_pretrained(
        args.base_model,
        torch_dtype=torch.bfloat16,
        trust_remote_code=True,
        device_map='auto',
        max_memory={i: '60GiB' for i in range(torch.cuda.device_count())},
    )

    # Apply LoRA to language model layers only
    lora_config = LoraConfig(
        r=args.lora_r,
        lora_alpha=args.lora_r * 2,
        lora_dropout=0.05,
        target_modules=['q_proj', 'k_proj', 'v_proj', 'o_proj',
                        'gate_proj', 'up_proj', 'down_proj'],
        bias='none',
        task_type='CAUSAL_LM',
    )
    model = get_peft_model(model, lora_config)
    model.print_trainable_parameters()
    # use_reentrant=False required for PEFT + DDP gradient checkpointing
    model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})

    logger.info('Loading dataset')
    dataset = VideoSFTDataset(args.data, processor, args.max_length, args.max_samples)

    if len(dataset) == 0:
        logger.error('No valid samples! Check video paths and data format.')
        sys.exit(1)

    logger.info(f'Dataset size: {len(dataset)}')

    # Sample label coverage from first 5 items (lazy, to avoid preloading all)
    n_check = min(5, len(dataset))
    total_tokens = learnable_tokens = 0
    for i in range(n_check):
        s = dataset[i]
        total_tokens += len(s['input_ids'])
        learnable_tokens += (s['labels'] != -100).sum().item()
        del s
    if total_tokens > 0:
        logger.info(f'Label coverage (first {n_check} samples): {100*learnable_tokens/total_tokens:.1f}%')

    # Training args
    # ddp_find_unused_parameters=False: LoRA freezes base params but they still
    # participate in forward pass, so all declared params appear in autograd graph.
    training_args = TrainingArguments(
        output_dir=args.output,
        num_train_epochs=args.epochs,
        per_device_train_batch_size=1,
        gradient_accumulation_steps=args.grad_accum,
        learning_rate=args.lr,
        warmup_steps=4,
        lr_scheduler_type='cosine',
        bf16=True,
        logging_steps=5,
        save_strategy='steps',
        save_steps=args.save_steps,
        save_total_limit=3,
        gradient_checkpointing=True,
        remove_unused_columns=False,
        dataloader_num_workers=0,
        report_to='none',
        ddp_find_unused_parameters=False,
    )

    trainer = LabelOnlyTrainer(
        model=model,
        args=training_args,
        train_dataset=dataset,
        data_collator=collate_fn,
        processing_class=processor.tokenizer,
    )

    logger.info('Starting training')
    trainer.train(resume_from_checkpoint=True if args.resume else None)

    output_path = Path(args.output) / 'final_model'
    trainer.save_model(str(output_path))
    processor.save_pretrained(str(output_path))
    logger.info(f'Saved to {output_path}')


if __name__ == '__main__':
    main()
