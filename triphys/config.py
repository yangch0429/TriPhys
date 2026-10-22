"""Runtime configuration. Every API key and machine-specific path is read from the environment; nothing is stored in
the repository. See .env.example for the full list."""
import os
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
DATA_DIR = Path(os.environ.get('TRIPHYS_DATA_DIR', REPO / 'data'))


def env(name, default=None, required=False):
    v = os.environ.get(name, default)
    if required and not v:
        raise RuntimeError('environment variable %s is not set (see .env.example)' % name)
    return v


# ---- videos (not redistributed; see README "Videos") ----
# MPR: <EGOEXO_ROOT>/<take_uid>/best_exo.mp4      KPR: <BAAI_ROOT>/<folder_id>/<video_id>.mp4
EGOEXO_ROOT = env('EGOEXO_ROOT', '')
BAAI_ROOT = env('BAAI_ROOT', '')

# ---- OpenAI-compatible chat endpoint for GPT-5.5 / Gemini-3.6-Flash (evaluation, Stage-I rewriting and teacher) ----
CHAT_API_BASE = env('CHAT_API_BASE', 'https://openrouter.ai/api/v1')
CHAT_API_KEY_ENV = 'OPENROUTER_API_KEY'
GPT_MODEL = env('GPT_MODEL', 'openai/gpt-5.5')
GEMINI_MODEL = env('GEMINI_MODEL', 'google/gemini-3.6-flash')

# ---- ACC judge (model-based evaluation of nonconforming responses and TYPE_D) ----
ACC_JUDGE_BASE = env('ACC_JUDGE_BASE', 'https://openrouter.ai/api/v1')
ACC_JUDGE_MODEL = env('ACC_JUDGE_MODEL', 'openai/gpt-4o-mini')
ACC_JUDGE_KEY_ENV = 'ACC_JUDGE_API_KEY'          # falls back to OPENROUTER_API_KEY

# ---- SCS judge: DeepSeek-V4-Flash (API model id "deepseek-flash", thinking disabled), temperature 0.3, 60 s ----
SCS_JUDGE_BASE = env('SCS_JUDGE_BASE', 'https://api.deepseek.com')
SCS_JUDGE_MODEL = env('SCS_JUDGE_MODEL', 'deepseek-flash')
SCS_JUDGE_KEY_ENV = 'DEEPSEEK_API_KEY'

# ---- web_search tool: Serper (Google), comma-separated keys rotated on rate limit / exhausted credit ----
SERPER_KEYS_ENV = 'SERPER_API_KEYS'


def key(name, fallback=None):
    v = os.environ.get(name) or (os.environ.get(fallback) if fallback else None)
    if not v:
        raise RuntimeError('API key %s is not set (see .env.example)' % name)
    return v


def chat_key(model=None):
    """key of the chat endpoint (GPT-5.5 / Gemini-3.6-Flash): OPENROUTER_API_KEY"""
    return key(CHAT_API_KEY_ENV)


def serper_keys():
    return [k.strip() for k in os.environ.get(SERPER_KEYS_ENV, '').split(',') if k.strip()]
