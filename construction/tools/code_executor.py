import os
import io
import re
import asyncio
import random
import string
import time
import uuid
from typing import Any, Dict
from contextlib import redirect_stdout
import logging

logger = logging.getLogger(__name__)

FILES_OUTPUT_DIR = os.getenv("FILES_OUTPUT_DIR", "./files")

UNSAFE_PATTERNS = [
    r'(?<!\w)(subprocess|multiprocessing|threading|ctypes|_thread)\s*[.\(]',
    r'os\.(system|popen|fork|kill|remove|rmdir|unlink)\s*\(',
    r'shutil\.(rmtree|move)\s*\(',
    r'(?<!\w)(eval|exec|__import__|compile)\s*\(',
    r'socket\.socket\s*\(',
]

ALLOWED_IMPORTS = """
import numpy as np
import pandas as pd
import sympy
import math
import json
import csv
import datetime
import random
import string
import re
import collections
import itertools
import openpyxl
import pyarrow
import h5py
from PIL import Image
import fpdf
from reportlab.pdfgen import canvas
from reportlab.lib.pagesizes import letter
import yaml
import sqlite3
import pickle
import logging
import tarfile
import gzip
import zipfile
import wave
import xml.etree.ElementTree as ET
from sympy import symbols, Eq, solve, simplify, expand, factor
from collections import Counter, defaultdict
from itertools import combinations, permutations
from functools import reduce
"""


class CodeExecutionError(Exception):
    pass


class SafeRuntime:
    def __init__(self):
        self._global_vars = {}
        self._setup_environment()

    def _setup_environment(self):
        exec(ALLOWED_IMPORTS, self._global_vars)
        self._global_vars["FILES_OUTPUT_DIR"] = FILES_OUTPUT_DIR
        self._global_vars["generate_unique_filename"] = generate_unique_filename

    def _check_safety(self, code: str) -> None:
        for pattern in UNSAFE_PATTERNS:
            if re.search(pattern, code):
                raise CodeExecutionError("Unsafe code pattern detected. Execution blocked for security reasons.")

    def exec_code(self, code: str) -> None:
        self._check_safety(code)
        exec(code, self._global_vars)

    def eval_code(self, expr: str) -> Any:
        return eval(expr, self._global_vars)


def generate_unique_filename(extension: str, prefix: str = "") -> str:
    timestamp = time.strftime("%m%d%H%M%S")
    unique_id = uuid.uuid4().hex[:8]
    ext = extension.lstrip(".")
    if prefix:
        return f"{timestamp}_{prefix}_{unique_id}.{ext}"
    return f"{timestamp}_{unique_id}.{ext}"


async def code_executor(code: str) -> Dict[str, Any]:
    os.makedirs(FILES_OUTPUT_DIR, exist_ok=True)

    result = {
        "stdout": "",
        "result": None,
        "created_files": [],
        "error": None,
    }

    try:
        runtime = SafeRuntime()
        code = code.strip()
        full_code = f'FILES_OUTPUT_DIR = "{FILES_OUTPUT_DIR}"\n' + code

        stdout_capture = io.StringIO()

        def _execute():
            with redirect_stdout(stdout_capture):
                runtime.exec_code(full_code)

        try:
            await asyncio.wait_for(asyncio.to_thread(_execute), timeout=30.0)
        except asyncio.TimeoutError:
            result["error"] = "Code execution timed out after 30 seconds"
            return result

        stdout_capture.seek(0)
        result["stdout"] = stdout_capture.read().strip()

        if os.path.exists(FILES_OUTPUT_DIR):
            current_time = time.time()
            for filename in os.listdir(FILES_OUTPUT_DIR):
                filepath = os.path.join(FILES_OUTPUT_DIR, filename)
                if os.path.isfile(filepath):
                    mtime = os.path.getmtime(filepath)
                    if current_time - mtime < 60:
                        result["created_files"].append({
                            "path": filepath,
                            "filename": filename,
                            "size": os.path.getsize(filepath),
                        })

        try:
            code_lines = code.strip().split("\n")
            if code_lines:
                last_line = code_lines[-1].strip()
                if (
                    last_line
                    and not last_line.startswith(("import", "from", "def", "class", "if", "for", "while", "try", "with", "#"))
                    and "=" not in last_line.split("(")[0]
                    and not last_line.endswith(":")
                ):
                    try:
                        expr_result = runtime.eval_code(last_line)
                        if expr_result is not None:
                            result["result"] = str(expr_result)
                    except Exception:
                        pass
        except Exception:
            pass

    except CodeExecutionError as e:
        result["error"] = str(e)
    except Exception as e:
        result["error"] = f"Execution error: {str(e)}"

    return result


def get_openai_function_code_executor(create_files: bool = False) -> dict:
    if create_files:
        description = (
            "Execute Python code in a secure sandbox environment. Use this tool for:\n"
            "1. Complex mathematical computations (algebra, calculus, statistics, etc.)\n"
            "2. Data processing and analysis\n"
            "3. Creating files for constructing file-based questions\n\n"
            "Available modules: numpy, pandas, sympy, math, json, csv, datetime, random, collections, itertools, "
            "openpyxl, pyarrow, h5py, Pillow, fpdf, reportlab, pyyaml, sqlite3, pickle, tarfile, gzip, zipfile, wave, xml.etree.ElementTree\n"
            f"Use print() to output results. FILES_OUTPUT_DIR = '{FILES_OUTPUT_DIR}' variable is available for saving files.\n"
            "Saved filenames must be randomly generated with 10 letters plus extension to ensure uniqueness."
        )
    else:
        description = (
            "Execute Python code in a secure sandbox environment. Use this tool for:\n"
            "1. Complex mathematical computations (algebra, calculus, statistics, etc.)\n"
            "2. Data processing and analysis\n"
            "Available modules: numpy, pandas, sympy, math, json, csv, datetime, random, collections, itertools\n"
            "Use print() to output results."
        )
    return {
        "type": "function",
        "function": {
            "name": "code_executor",
            "description": description,
            "parameters": {
                "type": "object",
                "properties": {
                    "code": {"type": "string", "description": "Python code to execute."}
                },
                "required": ["code"],
            },
        },
    }
