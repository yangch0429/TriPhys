#!/usr/bin/env python3
"""DPO (Stage II) on top of the Stage-I model (base + Stage-I LoRA), video + multi-turn tool trajectories in the Qwen3.5
native tool format (VideoSFTDataset of train_sft_n1.py: chat template with tools=, frame subsampling, label mask).
  policy    = (base merged with the Stage-I LoRA) + new LoRA
  reference = the same model with the new LoRA disabled (= the Stage-I model)
  sequence log-prob = sum over assistant tokens (all rounds; tool/system/user masked; logits only at label
                      positions, chunked log-softmax)
  loss = -log sigmoid(beta * ((pi_c - ref_c) - (pi_r - ref_r)))  [+ sft_alpha * NLL(chosen) / n_tokens]
--length_norm: per-token log-ratios, beta_eff = beta x mean pair length ('tokens' of the pair file).
--class_balance: loss weights so that the A and B pairs have equal total weight.
--save_half: also save an adapter at half of each epoch.
--mask_common_prefix: chosen and rejected share a message prefix; labels before the first differing token are masked in
both, so only the decision after the shared prefix is scored (for whole-trajectory pairs the masked tokens are
identical in both and cancel in the DPO margin anyway).
Reference log-probs are computed once (first epoch) and cached. No DDP wrapper: LoRA gradients are
all-reduced by hand after each accumulation window (two policy forwards per pair before backward).
Input pipeline:
  - decoded videos (qwen_vl_utils.process_vision_info) are cached per video (LRU 32 per rank): chosen and rejected of
    a pair, and the pairs of one QA, share the video, so each is decoded once instead of 2-4x per epoch
  - one background thread prepares (_process_sample) the next --prefetch pairs while the GPU runs the current one;
    all tokenizer / processor calls stay in that single thread (fast tokenizers are not thread-safe)
The merged Stage-I model (+ processor) is saved to --merged_dir; it is the --base_model of the DPO adapter in eval_n1.py.
Usage (from the work dir $TRIPHYS_WORK; see training/run.sh):
  torchrun --nproc_per_node=8 scripts/train_dpo_p2.py --sft_ckpt outputs/phase1/comb1_1e4/final_model
      --merged_dir outputs/phase1/comb1_1e4/merged --data data/p2/dpo_p2_full1.json --output outputs/phase2/p2_full1
      --max_length 16384 --epochs 2 --lr 5e-6 --beta 0.1 --sft_alpha 0 --grad_accum 2 --mask_common_prefix
      --length_norm --class_balance
"""
from srvpaths import BASE_MODEL, E, N1, SCRIPTS  # machine paths, see srvpaths.py
import os, sys, json, math, time, argparse, logging, random
import torch
import torch.distributed as dist
import torch.nn.functional as F
from torch.utils.checkpoint import checkpoint

os.environ['FORCE_QWENVL_VIDEO_READER'] = 'decord'
os.environ.setdefault('PYTORCH_CUDA_ALLOC_CONF', 'expandable_segments:True')
sys.path.insert(0, SCRIPTS)
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, N1)
from train_sft_n1 import VideoSFTDataset  # noqa: E402  (native tools= rendering, as in the SFT)

import collections
from concurrent.futures import ThreadPoolExecutor
import qwen_vl_utils  # noqa: E402
_pvi, _vcache = qwen_vl_utils.process_vision_info, collections.OrderedDict()


def _cached_pvi(messages, *a, **kw):
    key = repr([c for m in messages if isinstance(m.get('content'), list) for c in m['content']
                if isinstance(c, dict) and c.get('type') in ('video', 'image')]) + repr((a, sorted(kw.items())))
    if key not in _vcache:
        _vcache[key] = _pvi(messages, *a, **kw)
        if len(_vcache) > 32:
            _vcache.popitem(last=False)
    _vcache.move_to_end(key)
    return _vcache[key]


qwen_vl_utils.process_vision_info = _cached_pvi   # _process_sample imports it at call time

logging.basicConfig(level=logging.INFO, format='%(asctime)s %(levelname)s %(message)s')
logger = logging.getLogger(__name__)
MODEL_PATH = BASE_MODEL


def _chunk_lp(logits, tgt):
    x = logits.float()
    return (x.gather(-1, tgt[:, None]).squeeze(-1) - x.logsumexp(-1)).sum()


def seq_logp(model, entry, device, chunk=1024):
    """sum of log-probs of the label tokens (assistant turns). Logits are computed only at the positions that
    predict a label token (logits_to_keep = index tensor), and log-softmax runs in checkpointed chunks, so the
    full-sequence fp32 logits (seq x 248k vocab) are never materialized."""
    kw = {k: v.unsqueeze(0).to(device) for k, v in entry.items()
          if k in ('input_ids', 'attention_mask', 'mm_token_type_ids')}
    if 'pixel_values_videos' in entry:
        kw['pixel_values_videos'] = entry['pixel_values_videos'].to(device)
        kw['video_grid_thw'] = entry['video_grid_thw'].to(device)
    labels = entry['labels'].to(device)
    pos = (labels[1:] != -100).nonzero(as_tuple=True)[0]      # position i predicts token i+1
    tgt = labels[pos + 1]
    logits = model(**kw, logits_to_keep=pos).logits[0]
    total = 0.0
    for i in range(0, len(pos), chunk):
        lg, tg = logits[i:i + chunk], tgt[i:i + chunk]
        total = total + (checkpoint(_chunk_lp, lg, tg, use_reentrant=False) if torch.is_grad_enabled()
                         else _chunk_lp(lg, tg))
    return total, len(pos)


def mask_prefix(ec, er):
    a, b = ec['input_ids'], er['input_ids']
    n = min(len(a), len(b))
    diff = (a[:n] != b[:n]).nonzero(as_tuple=True)[0]
    d = diff[0].item() if len(diff) else n
    ec, er = dict(ec, labels=ec['labels'].clone()), dict(er, labels=er['labels'].clone())
    ec['labels'][:d] = -100
    er['labels'][:d] = -100
    return ec, er


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--sft_ckpt', required=True)
    ap.add_argument('--data', required=True)
    ap.add_argument('--output', required=True)
    ap.add_argument('--merged_dir', required=True)
    ap.add_argument('--base_model', default=MODEL_PATH)
    ap.add_argument('--max_length', type=int, default=16384)
    ap.add_argument('--epochs', type=int, default=2)
    ap.add_argument('--lr', type=float, default=5e-6)
    ap.add_argument('--beta', type=float, default=0.1)
    ap.add_argument('--sft_alpha', type=float, default=0.0)
    ap.add_argument('--lora_r', type=int, default=32)
    ap.add_argument('--grad_accum', type=int, default=4, help='pairs per rank per optimizer step')
    ap.add_argument('--seed', type=int, default=42)
    ap.add_argument('--mask_common_prefix', action='store_true')
    ap.add_argument('--length_norm', action='store_true')
    ap.add_argument('--class_balance', action='store_true')
    ap.add_argument('--save_half', action='store_true')
    ap.add_argument('--prefetch', type=int, default=3)
    args = ap.parse_args()

    dist.init_process_group('nccl')
    rank, world = dist.get_rank(), dist.get_world_size()
    local = int(os.environ.get('LOCAL_RANK', 0))
    torch.cuda.set_device(local)
    device = torch.device('cuda', local)

    from transformers import Qwen3VLProcessor, Qwen3_5ForConditionalGeneration
    from peft import PeftModel, LoraConfig, get_peft_model
    processor = Qwen3VLProcessor.from_pretrained(args.base_model, trust_remote_code=True)
    base = Qwen3_5ForConditionalGeneration.from_pretrained(args.base_model, torch_dtype=torch.bfloat16,
                                                           trust_remote_code=True)
    model = PeftModel.from_pretrained(base, args.sft_ckpt).merge_and_unload()
    if rank == 0 and not os.path.exists(os.path.join(args.merged_dir, 'config.json')):
        model.save_pretrained(args.merged_dir)
        processor.save_pretrained(args.merged_dir)
        logger.info('merged SFT model saved to %s' % args.merged_dir)
    dist.barrier()
    if args.seed != 42:   # a non-default seed also seeds the LoRA init (seed 42 leaves the init unseeded)
        torch.manual_seed(args.seed)
    model = get_peft_model(model, LoraConfig(
        r=args.lora_r, lora_alpha=args.lora_r * 2, lora_dropout=0.05, bias='none', task_type='CAUSAL_LM',
        target_modules=['q_proj', 'k_proj', 'v_proj', 'o_proj', 'gate_proj', 'up_proj', 'down_proj']))
    model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={'use_reentrant': False})
    model.enable_input_require_grads()
    model.config.use_cache = False
    model.to(device)
    if rank == 0:
        model.print_trainable_parameters()
    params = [p for p in model.parameters() if p.requires_grad]

    ds = VideoSFTDataset.__new__(VideoSFTDataset)
    ds.processor, ds.max_length = processor, args.max_length
    pairs = json.load(open(args.data))
    ncls = {c: sum(1 for p in pairs if p.get('cls') == c) for c in {p.get('cls') for p in pairs}}
    wcls = {c: len(pairs) / (len(ncls) * n) for c, n in ncls.items()} if args.class_balance else {c: 1.0 for c in ncls}
    mean_tok = sum(sum(p['tokens']) / 2 for p in pairs) / len(pairs) if args.length_norm else 1.0
    beta_eff = args.beta * mean_tok
    n_per_rank = len(pairs) // world            # equal step count on every rank
    steps_per_epoch = math.ceil(n_per_rank / args.grad_accum)
    total = steps_per_epoch * args.epochs
    opt = torch.optim.AdamW(params, lr=args.lr, weight_decay=0.0)
    warm = max(1, total // 10)
    sched = torch.optim.lr_scheduler.LambdaLR(
        opt, lambda s: (s + 1) / warm if s < warm else 0.5 * (1 + math.cos(math.pi * (s - warm) / max(1, total - warm))))
    if rank == 0:
        logger.info('class counts %s weights %s, length_norm %s mean tokens %.0f beta_eff %g' % (
            ncls, wcls, args.length_norm, mean_tok, beta_eff))
        logger.info('pairs %d, per rank %d, steps/epoch %d, total %d, lr %g, beta %g, sft_alpha %g, mask_common_prefix %s'
                    % (len(pairs), n_per_rank, steps_per_epoch, total, args.lr, args.beta, args.sft_alpha, args.mask_common_prefix))
    ref_cache = {}
    step = 0
    pool = ThreadPoolExecutor(1)

    def prep(i):
        return ds._process_sample(pairs[i]['chosen'], i), ds._process_sample(pairs[i]['rejected'], i)
    model.train()
    for ep in range(args.epochs):
        order = list(range(len(pairs)))
        random.Random(args.seed + ep).shuffle(order)
        mine = order[rank::world][:n_per_rank]
        stats = torch.zeros(4, device=device)      # loss, reward acc, margin, count
        futs = {}
        for bi in range(0, n_per_rank, args.grad_accum):
            t0 = time.time()
            batch = mine[bi:bi + args.grad_accum]
            for k, idx in enumerate(batch):
                for j in mine[bi + k:bi + k + 1 + args.prefetch]:
                    if j not in futs:
                        futs[j] = pool.submit(prep, j)
                p = pairs[idx]
                ec, er = futs.pop(idx).result()
                if ec is None or er is None:      # pairs that do not fit --max_length are skipped
                    logger.warning('pair %d does not fit, skipped' % idx)
                    continue
                if args.mask_common_prefix:
                    ec, er = mask_prefix(ec, er)
                if idx not in ref_cache:
                    with torch.no_grad(), model.disable_adapter():
                        ref_cache[idx] = (seq_logp(model, ec, device)[0].item(), seq_logp(model, er, device)[0].item())
                rc, rr = ref_cache[idx]
                pc, nc = seq_logp(model, ec, device)
                pr, nr = seq_logp(model, er, device)
                if args.length_norm:
                    margin = beta_eff * ((pc - rc) / max(1, nc) - (pr - rr) / max(1, nr))
                else:
                    margin = args.beta * ((pc - rc) - (pr - rr))
                loss = -F.logsigmoid(margin) * wcls.get(p.get('cls'), 1.0)
                if args.sft_alpha > 0:
                    loss = loss + args.sft_alpha * (-pc / max(1, nc))
                (loss / len(batch)).backward()
                stats += torch.tensor([loss.item(), float(margin.item() > 0), margin.item(), 1.0], device=device)
            for q in params:
                if q.grad is None:
                    q.grad = torch.zeros_like(q)
                dist.all_reduce(q.grad, op=dist.ReduceOp.SUM)
                q.grad /= world
            torch.nn.utils.clip_grad_norm_(params, 1.0)
            opt.step()
            sched.step()
            opt.zero_grad(set_to_none=True)
            step += 1
            s = stats.clone()
            dist.all_reduce(s)
            if rank == 0:
                logger.info('ep %d step %d/%d loss %.4f reward_acc %.3f margin %.4f lr %.2e (%.0fs)' % (
                    ep + 1, step, total, s[0] / s[3], s[1] / s[3], s[2] / s[3], sched.get_last_lr()[0], time.time() - t0))
            stats.zero_()
            if args.save_half and rank == 0 and bi // args.grad_accum + 1 == steps_per_epoch // 2:
                d = os.path.join(args.output, 'checkpoint-ep%d.5' % ep)
                model.save_pretrained(d)
                processor.save_pretrained(d)
                logger.info('saved %s' % d)
        if rank == 0:
            d = os.path.join(args.output, 'checkpoint-ep%d' % (ep + 1))
            model.save_pretrained(d)
            processor.save_pretrained(d)
            logger.info('saved %s' % d)
        dist.barrier()
    if rank == 0:
        d = os.path.join(args.output, 'final_model')
        model.save_pretrained(d)
        processor.save_pretrained(d)
        logger.info('saved %s' % d)
    dist.destroy_process_group()


if __name__ == '__main__':
    main()
