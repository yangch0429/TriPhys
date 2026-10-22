"""n1 (10/14 SFT/DPO redesign): Qwen3.5 native tool format shared by data build, training and eval.
Tools are passed to the chat template (`tools=TOOLS`), which writes the tool schema into the system turn; calls are XML
  <tool_call>\n<function=web_search>\n<parameter=query>\n...\n</parameter>\n</function>\n</tool_call>
and tool results are role "tool" messages (rendered as <tool_response> in a user turn). Result payload unchanged from
eval_sft_v9: json.dumps({"result": ...})."""
import json, re

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

_TC = re.compile(r'<tool_call>(.*?)</tool_call>', re.S)
_FN = re.compile(r'<function=([^>\s]+)>(.*?)</function>', re.S)
_PA = re.compile(r'<parameter=([^>\s]+)>\n?(.*?)\n?</parameter>', re.S)


def parse_tool_calls(content):
    """[(name, args)] of all tool calls in an assistant turn; native XML, JSON kept as fallback"""
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
    """training message: reasoning in <think>, tool calls rendered by the template"""
    m = {'role': 'assistant', 'content': '<think>\n%s\n</think>\n\n%s' % (think.strip(), content)}
    if calls:
        m['tool_calls'] = [{'type': 'function', 'function': {'name': n, 'arguments': a}} for n, a in calls]
    return m


def parse_lenient(content):
    """eval_n1b: parse_tool_calls, plus bare <function=...>...</function> blocks (no <tool_call> wrapper) after </think>"""
    out = parse_tool_calls(content)
    if out:
        return out
    tail = content.split('</think>')[-1]
    return [(m.group(1).strip(), {k.strip(): v for k, v in _PA.findall(m.group(2))}) for m in _FN.finditer(tail)]
