"""Minimal async client for OpenAI-compatible /chat/completions endpoints, with retry on rate limits and server errors."""
import asyncio
import json
import logging

import httpx

logger = logging.getLogger(__name__)
RETRY_STATUS = (408, 409, 429, 500, 502, 503, 504)


async def chat(base, key, model, messages, temperature=0.0, max_tokens=800, timeout=60.0, retries=4, **extra):
    """returns the full JSON response; raises after `retries` failed attempts"""
    url = base.rstrip('/') + '/chat/completions'
    body = {'model': model, 'messages': messages, 'temperature': temperature, 'max_tokens': max_tokens, **extra}
    last = None
    for i in range(retries):
        try:
            async with httpx.AsyncClient(timeout=timeout) as c:
                r = await c.post(url, headers={'Authorization': 'Bearer ' + key}, json=body)
            if r.status_code == 200:
                d = r.json()
                if d.get('choices'):
                    return d
                last = 'no choices: %s' % json.dumps(d)[:200]
            else:
                last = 'HTTP %d: %s' % (r.status_code, r.text[:200])
                if r.status_code not in RETRY_STATUS:
                    break
        except Exception as e:  # timeout / network
            last = '%s: %s' % (type(e).__name__, str(e)[:200])
        await asyncio.sleep(min(60, 3 * 2 ** i))
    raise RuntimeError('chat(%s) failed: %s' % (model, last))


def content(resp):
    return (resp['choices'][0]['message'].get('content') or '').strip()


def json_block(text):
    """parse a JSON object from a reply that may be wrapped in a markdown fence"""
    t = text.strip()
    if '```' in t:
        parts = t.split('```')
        if len(parts) >= 2:
            t = parts[1][4:] if parts[1].startswith('json') else parts[1]
    if '{' in t:
        t = t[t.index('{'):t.rindex('}') + 1]
    return json.loads(t)
