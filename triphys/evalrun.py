"""Shared evaluation driver: iterate over questions, solve, score, append one JSON line per question (resumable).
`solve(item) -> (last_turn, turns, tool_log)`."""
import asyncio
import json
import logging
import os
import threading
from concurrent.futures import ThreadPoolExecutor

from . import metrics
from .tools import SERPER_FAILED

logger = logging.getLogger(__name__)


def done_ids(path):
    if not os.path.exists(path):
        return set()
    return {json.loads(l)['qid'] for l in open(path, encoding='utf-8') if l.strip()}


def shard(items, shard_id, n_shards):
    """split by video so that one video is decoded once per worker"""
    vids = sorted({x['video'] for x in items})
    mine = set(vids[shard_id::n_shards])
    return [x for x in items if x['video'] in mine]


def run(items, solve, out_path, thinking_opened=False, score=True, model_name='', workers=1):
    """workers > 1: questions are solved in parallel threads (API models); local models use workers=1"""
    os.makedirs(os.path.dirname(os.path.abspath(out_path)), exist_ok=True)
    done = done_ids(out_path)
    todo = [x for x in items if x['qid'] not in done]
    logger.info('%d questions, %d cached, %d to run -> %s', len(items), len(done), len(todo), out_path)
    lock = threading.Lock()
    counter = [0]

    def one(it):
        qa = it['qa']
        try:
            last, turns, tool_log = solve(it)
            err = None
        except Exception as e:                 # recorded; such rows can be re-run by deleting them
            logger.exception('solve failed for %s', it['qid'])
            last, turns, tool_log, err = '', [], [], '%s: %s' % (type(e).__name__, e)
        row = {'qid': it['qid'], 'subtask': it['subtask'], 'model': model_name, 'question': qa['question'],
               'question_type': qa['question_type'], 'difficulty': qa.get('difficulty'),
               'ground_truth': qa['answer'], 'answer_range': qa.get('answer_range'), 'response': last,
               'turns': turns, 'tool_log': tool_log, 'n_tool_calls': len(tool_log), 'solve_error': err,
               'serper_failed': any(SERPER_FAILED in str(t.get('content', '')) for t in turns if t['role'] == 'tool')}
        if score:
            row.update(asyncio.run(metrics.score_response(qa, last, thinking_opened)))
        with lock:
            with open(out_path, 'a', encoding='utf-8') as f:
                f.write(json.dumps(row, ensure_ascii=False) + '\n')
            counter[0] += 1
            logger.info('[%d/%d] %s %s ACC %s SCS %s tools %d', counter[0], len(todo), it['subtask'], it['qid'],
                        row.get('ACC'), row.get('SCS'), len(tool_log))

    if workers <= 1:
        for it in todo:
            one(it)
    else:
        with ThreadPoolExecutor(workers) as ex:
            list(ex.map(one, todo))
