"""The two solver tools (web_search, code_executor) and the Qwen3.5 native tool-call format.

Qwen models receive TOOLS through the chat template (`apply_chat_template(..., tools=TOOLS)`), which writes the schema
into the system turn; calls are XML
  <tool_call>\n<function=web_search>\n<parameter=query>\n...\n</parameter>\n</function>\n</tool_call>
and tool results are role "tool" messages. API models receive the same TOOLS through function calling. Every tool
result is the JSON string {"result": ...}."""
import contextlib
import io
import json
import logging
import math
import re
import signal
import threading
import urllib.error
import urllib.request

from . import config

logger = logging.getLogger(__name__)

TOOLS = [
    {"type": "function", "function": {
        "name": "web_search",
        "description": "Search the web (Google) and return the top-10 result snippets. Use it to look up external "
                       "reference values the question does not give (typical masses or sizes of objects, material "
                       "properties, standards, typical human or sports data).",
        "parameters": {"type": "object", "properties": {"query": {"type": "string", "description": "search query"}},
                       "required": ["query"]}}},
    {"type": "function", "function": {
        "name": "code_executor",
        "description": "Execute Python code (math and numpy available) and return what it prints. Use it for "
                       "numerical calculation.",
        "parameters": {"type": "object", "properties": {"code": {"type": "string",
                                                                 "description": "Python code that prints the results"}},
                       "required": ["code"]}}},
]

SERPER_FAILED = '__SERPER_FAILED__'
CODE_TIMEOUT_S = 60

_TC = re.compile(r'<tool_call>(.*?)</tool_call>', re.S)
_FN = re.compile(r'<function=([^>\s]+)>(.*?)</function>', re.S)
_PA = re.compile(r'<parameter=([^>\s]+)>\n?(.*?)\n?</parameter>', re.S)


def parse_tool_calls(content):
    """[(name, args)] of all tool calls in an assistant turn (after </think>); native XML, JSON kept as fallback"""
    out = []
    for blk in _TC.findall(content.split('</think>')[-1]):
        m = _FN.search(blk)
        if m:
            out.append((m.group(1).strip(), {k.strip(): v for k, v in _PA.findall(m.group(2))}))
            continue
        try:
            d = json.loads(blk)
            out.append((d.get('name', ''), d.get('arguments', {}) or {}))
        except Exception:
            out.append(('unknown', {}))
    return out


def assistant_msg(think, calls=(), content=''):
    """training message: reasoning in <think>, tool calls rendered by the chat template"""
    m = {'role': 'assistant', 'content': '<think>\n%s\n</think>\n\n%s' % (think.strip(), content)}
    if calls:
        m['tool_calls'] = [{'type': 'function', 'function': {'name': n, 'arguments': a}} for n, a in calls]
    return m


# ---------------------------------------------------------------- web_search
class WebSearch:
    """Serper Google search, top-10 snippets. Keys (env SERPER_API_KEYS) rotate on 400/401/402/403/429. A failed
    search returns __SERPER_FAILED__ and is logged in .failures (training data drops such trajectories)."""

    def __init__(self, keys=None, timeout=8):
        self.keys = keys if keys is not None else config.serper_keys()
        self.i = 0
        self.timeout = timeout
        self.failures = []

    def __call__(self, query, question_id=''):
        if not self.keys:
            raise RuntimeError('SERPER_API_KEYS is not set')
        for _ in range(len(self.keys) * 2):
            k = self.keys[self.i % len(self.keys)]
            try:
                req = urllib.request.Request('https://google.serper.dev/search',
                                             data=json.dumps({'q': query, 'num': 10}).encode(),
                                             headers={'X-API-KEY': k, 'Content-Type': 'application/json'})
                with urllib.request.urlopen(req, timeout=self.timeout) as r:
                    data = json.loads(r.read())
                snip = [x.get('snippet', '') for x in data.get('organic', []) if x.get('snippet')]
                if snip:
                    return json.dumps({'result': 'Search results for "%s": ' % query + ' | '.join(snip[:10])})
                return json.dumps({'result': 'No results found for: %s' % query})
            except urllib.error.HTTPError as e:
                if e.code in (400, 401, 402, 403, 429):
                    logger.warning('Serper key #%d rejected (HTTP %d), rotating', self.i % len(self.keys), e.code)
                    self.i += 1
                    continue
                err = 'HTTP %d' % e.code
            except Exception as e:
                err = str(e)
            self.failures.append({'question_id': question_id, 'query': query, 'error': err})
            return json.dumps({'result': '%s: %s' % (SERPER_FAILED, query)})
        self.failures.append({'question_id': question_id, 'query': query, 'error': 'all keys exhausted'})
        return json.dumps({'result': '%s: %s' % (SERPER_FAILED, query)})


# ---------------------------------------------------------------- code_executor
def exec_code(code: str) -> str:
    """run code with math / numpy in scope, return stdout; 60 s wall-clock limit (SIGALRM, main thread on POSIX)"""
    out = io.StringIO()
    g = {'__builtins__': __builtins__, 'math': math,
         **{f: getattr(math, f) for f in dir(math) if not f.startswith('_')}}
    try:
        import numpy as np
        g['np'] = g['numpy'] = np
    except ImportError:
        pass
    use_alarm = hasattr(signal, 'SIGALRM') and threading.current_thread() is threading.main_thread()

    def _to(signum, frame):
        raise TimeoutError('code execution exceeded %d s' % CODE_TIMEOUT_S)
    old = None
    try:
        if use_alarm:
            old = signal.signal(signal.SIGALRM, _to)
            signal.alarm(CODE_TIMEOUT_S)
        with contextlib.redirect_stdout(out):
            exec(code, g)
        return out.getvalue().strip() or 'No output'
    except BaseException as e:
        if isinstance(e, KeyboardInterrupt):
            raise
        return 'Error: %s' % e
    finally:
        if use_alarm:
            signal.alarm(0)
            signal.signal(signal.SIGALRM, old)


def run_tool(name, args, search, question_id=''):
    """execute one call -> JSON result string"""
    if name == 'code_executor':
        return json.dumps({'result': exec_code(args.get('code', ''))})
    if name == 'web_search':
        return search(args.get('query', '') or args.get('q', ''), question_id)
    return json.dumps({'result': "Tool '%s' unavailable." % name})
