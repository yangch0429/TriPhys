"""
Step 2.5a: Build Physical Event Library

Uses an LLM to automatically generate a structured JSON knowledge base
of physical event entries. Each entry captures:
  - Event_Category: hierarchical path (e.g. "Mechanics/Collision")
  - Visual_Triggers: surface-level action keywords observable in video
  - Latent_Properties: hidden physical variables that cannot be seen directly
  - Actionable_Rules: conditional tool-call logic to retrieve latent variables
  - Graph_Injection: how to extend the event graph with new nodes

Usage:
    python build_physical_event_library.py \
        --output physical_event_library.json \
        --model deepseek-v3.2 \
        --api_url http://localhost:8089/v1
"""

import json
import os
import asyncio
import argparse
import logging
import sys
import re
from typing import List, Dict, Any
from openai import AsyncOpenAI

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    datefmt="%H:%M:%S",
    handlers=[logging.StreamHandler(sys.stdout)],
)
logging.getLogger("httpx").setLevel(logging.WARNING)
logging.getLogger("openai").setLevel(logging.WARNING)
logger = logging.getLogger(__name__)

DEFAULT_MODEL = "deepseek-v3.2"
DEFAULT_API_URL = "http://localhost:8089/v1"
DEFAULT_API_KEY = "empty"

# Top-level physical ontology — the LLM will expand each leaf into full entries.
# Structured after NVIDIA Cosmos-Reason1 and PHYSICS dataset taxonomies.
PHYSICS_ONTOLOGY = {
    "Mechanics": [
        "Collision",
        "Sliding_Friction",
        "Projectile_Motion",
        "Circular_Motion",
        "Spring_Oscillation",
        "Pulley_System",
        "Inclined_Plane",
        "Free_Fall",
    ],
    "Thermodynamics": [
        "Heat_Transfer",
        "Melting_Solidification",
        "Evaporation_Condensation",
        "Thermal_Expansion",
        "Combustion",
    ],
    "Fluid_Dynamics": [
        "Pouring_Flow",
        "Splashing_Impact",
        "Buoyancy_Floating",
        "Pressure_Difference",
        "Viscous_Flow",
    ],
    "Electromagnetism": [
        "Electric_Circuit",
        "Magnetic_Attraction_Repulsion",
        "Electromagnetic_Induction",
        "Electrostatic_Interaction",
    ],
    "Optics": [
        "Reflection",
        "Refraction",
        "Shadow_Formation",
        "Lens_Focusing",
    ],
    "Wave_Acoustics": [
        "Sound_Propagation",
        "Resonance",
        "Doppler_Effect",
    ],
}

# Extended observable mechanics subcategories (v2)
# These are more fine-grained and directly triggerable from video CV extraction.
# status: "active" = used in generation; "sensor_enhanced" = requires Class-A GT;
#         "dormant" = not yet triggerable with current pipeline
PHYSICS_ONTOLOGY_V2_MECHANICS = {
    "Mechanics": {
        # Original 8 categories (kept)
        "Collision":                  {"status": "active",           "requires_scale": True},
        "Sliding_Friction":           {"status": "active",           "requires_scale": True},
        "Projectile_Motion":          {"status": "active",           "requires_scale": True},
        "Circular_Motion":            {"status": "active",           "requires_scale": True},
        "Spring_Oscillation":         {"status": "active",           "requires_scale": True},
        "Pulley_System":              {"status": "active",           "requires_scale": True},
        "Inclined_Plane":             {"status": "active",           "requires_scale": True},
        "Free_Fall":                  {"status": "active",           "requires_scale": True},
        # New observable subcategories
        "Translational_Kinematics":   {"status": "active",           "requires_scale": True},
        "Projectile_Ascent":          {"status": "active",           "requires_scale": True},
        "Projectile_Descent":         {"status": "active",           "requires_scale": True},
        "Launch_Parameter_Estimation":{"status": "active",           "requires_scale": False},  # angle only
        "Rolling_Motion":             {"status": "active",           "requires_scale": True},
        "Bounce_Restitution":         {"status": "active",           "requires_scale": True},
        "Work_Energy_Conversion":     {"status": "active",           "requires_scale": True},
        "Rotational_Kinematics":      {"status": "active",           "requires_scale": True},
        "Balance_Stability_Torque":   {"status": "active",           "requires_scale": False},
        "Relative_Motion_Pursuit":    {"status": "active",           "requires_scale": True},
        "Trajectory_Comparison":      {"status": "active",           "requires_scale": False},
        "Momentum_Comparison":        {"status": "active",           "requires_scale": True},
        "Impulse_Momentum":           {"status": "active",           "requires_scale": True},
        # Sensor-enhanced (Class-A only)
        "Time_To_Collision":          {"status": "sensor_enhanced",  "requires_scale": True},
        "Braking_Dynamics":           {"status": "sensor_enhanced",  "requires_scale": True},
        "Lane_Change_Kinematics":     {"status": "sensor_enhanced",  "requires_scale": True},
        "3D_Manipulation_Trajectory": {"status": "sensor_enhanced",  "requires_scale": True},
        "Contact_Sequence_Reasoning": {"status": "sensor_enhanced",  "requires_scale": False},
    },
    "Thermodynamics": {
        "Heat_Transfer":              {"status": "dormant",          "requires_scale": False},
        "Melting_Solidification":     {"status": "dormant",          "requires_scale": False},
        "Evaporation_Condensation":   {"status": "dormant",          "requires_scale": False},
        "Thermal_Expansion":          {"status": "dormant",          "requires_scale": False},
        "Combustion":                 {"status": "dormant",          "requires_scale": False},
    },
    "Fluid_Dynamics": {
        "Pouring_Flow":               {"status": "active",           "requires_scale": False},
        "Splashing_Impact":           {"status": "active",           "requires_scale": False},
        "Buoyancy_Floating":          {"status": "active",           "requires_scale": True},
        "Pressure_Difference":        {"status": "dormant",          "requires_scale": True},
        "Viscous_Flow":               {"status": "dormant",          "requires_scale": False},
        "Drag_Deceleration":          {"status": "active",           "requires_scale": True},
    },
    "Electromagnetism": {
        "Electric_Circuit":           {"status": "dormant",          "requires_scale": False},
        "Magnetic_Attraction_Repulsion": {"status": "dormant",       "requires_scale": False},
        "Electromagnetic_Induction":  {"status": "dormant",          "requires_scale": False},
        "Electrostatic_Interaction":  {"status": "dormant",          "requires_scale": False},
    },
    "Optics": {
        "Reflection":                 {"status": "active",           "requires_scale": False},
        "Refraction":                 {"status": "active",           "requires_scale": False},
        "Shadow_Formation":           {"status": "active",           "requires_scale": False},
        "Lens_Focusing":              {"status": "dormant",          "requires_scale": False},
    },
    "Wave_Acoustics": {
        "Sound_Propagation":          {"status": "dormant",          "requires_scale": False},
        "Resonance":                  {"status": "dormant",          "requires_scale": False},
        "Doppler_Effect":             {"status": "dormant",          "requires_scale": False},
    },
}


SYSTEM_PROMPT = """You are a physics expert and knowledge engineer. Your task is to construct
structured JSON entries for a Physical Event Library used in a multimodal benchmark pipeline.

Each entry maps a physical phenomenon observable in videos to the hidden physical variables,
tool-call logic, and event graph injection rules needed to construct challenging QA questions.

Always output ONLY valid JSON wrapped in ```json ... ```. No extra text outside the code block."""


def make_generation_prompt(domain: str, event_type: str) -> str:
    return f"""Generate a Physical Event Library entry for the phenomenon: **{domain}/{event_type}**

The entry must conform EXACTLY to this JSON schema:

```json
{{
  "Event_Category": "{domain}/{event_type}",
  "Description": "One-sentence physical description of this phenomenon.",
  "Visual_Triggers": [
    "verb or phrase observable in video that indicates this phenomenon is occurring",
    "..."
  ],
  "Latent_Properties": [
    "physical_variable_name (symbol, unit): brief explanation of why it cannot be directly observed",
    "..."
  ],
  "Actionable_Rules": {{
    "Condition_1": "If [condition about what IS known from video], trigger: tool_name(args). Describe clearly.",
    "Condition_2": "...",
    "Condition_3": "..."
  }},
  "Graph_Injection": "Describe precisely: which Latent_Properties become new leaf nodes attached to which node types in the event graph, and what edges connect them.",
  "Example_Question_Sketch": "A one-sentence sketch of the kind of multi-hop question this entry enables (must require code_executor for calculation).",
  "Relevant_Formulas": [
    "formula string with variable definitions",
    "..."
  ]
}}
```

Requirements:
1. `Visual_Triggers` must be 3-6 SHORT phrases/verbs (lowercase) that a VLM/action-recognition model would output when detecting this event.
2. `Latent_Properties` must list 2-5 physically meaningful hidden variables with symbol, unit, and a brief reason why they are not directly observable.
3. `Actionable_Rules` must have 2-4 conditions covering different scenarios; each must specify a concrete tool call:
   - `web_search(query='...')` — to look up material properties, standard values, or real-world data
   - `code_executor(formula='...')` — to compute a physical quantity using a formula
   - `page_browser(url='...')` — to retrieve a specific reference
   - `web_image_search(query='...')` — to identify an object or material visually
4. `Graph_Injection` must clearly state: "Add [variable] as a leaf node to [node_type] nodes via [relation_type] edge."
5. `Example_Question_Sketch` must require at least 2 hops and a code_executor call.
6. `Relevant_Formulas` must be 1-4 key physics formulas in LaTeX or plain text with all variables defined.

Generate the entry now:
"""


def extract_json_from_response(text: str) -> Dict[str, Any]:
    text = re.sub(r"<think>.*?</think>", "", text, flags=re.DOTALL).strip()

    if "```json" in text:
        text = text.split("```json")[1].split("```")[0]
    elif "```" in text:
        text = text.split("```")[1].split("```")[0]

    text = text.strip()
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        start = text.find("{")
        end = text.rfind("}")
        if start != -1 and end != -1:
            try:
                return json.loads(text[start : end + 1])
            except Exception:
                pass
    return {"error": "json_decode_failed", "raw": text[:300]}


async def generate_entry(
    client: AsyncOpenAI,
    sem: asyncio.Semaphore,
    domain: str,
    event_type: str,
    model: str,
) -> Dict[str, Any]:
    category = f"{domain}/{event_type}"
    prompt = make_generation_prompt(domain, event_type)

    async with sem:
        for attempt in range(3):
            try:
                response = await client.chat.completions.create(
                    model=model,
                    messages=[
                        {"role": "system", "content": SYSTEM_PROMPT},
                        {"role": "user", "content": prompt},
                    ],
                    max_tokens=4096,
                    temperature=0.3,
                    timeout=120,
                )
                content = response.choices[0].message.content
                entry = extract_json_from_response(content)

                if "error" in entry:
                    logger.warning(
                        f"[{category}] Parse failed (attempt {attempt+1}): {entry.get('error')}"
                    )
                    if attempt < 2:
                        await asyncio.sleep(2 ** attempt)
                        continue
                else:
                    logger.info(f"[{category}] Generated successfully.")
                    return entry

            except Exception as e:
                logger.error(f"[{category}] Attempt {attempt+1} error: {e}")
                if attempt < 2:
                    await asyncio.sleep(2 ** attempt)

    logger.error(f"[{category}] All attempts failed, returning skeleton.")
    return {
        "Event_Category": category,
        "Description": "",
        "Visual_Triggers": [],
        "Latent_Properties": [],
        "Actionable_Rules": {},
        "Graph_Injection": "",
        "Example_Question_Sketch": "",
        "Relevant_Formulas": [],
        "_error": "generation_failed",
    }


async def main(args):
    client = AsyncOpenAI(base_url=args.api_url, api_key=args.api_key)
    sem = asyncio.Semaphore(args.concurrent)

    tasks = [
        (domain, event_type)
        for domain, events in PHYSICS_ONTOLOGY.items()
        for event_type in events
    ]

    logger.info(f"Generating {len(tasks)} Physical Event Library entries...")

    coros = [
        generate_entry(client, sem, domain, event_type, args.model)
        for domain, event_type in tasks
    ]

    results = await asyncio.gather(*coros)
    await client.close()

    library = [r for r in results if not (args.skip_errors and "_error" in r)]

    output = {
        "version": "1.0",
        "description": (
            "Physical Event Library for OmniGAIA benchmark construction. "
            "Each entry maps observable video events to hidden physical variables and tool-call rules."
        ),
        "ontology_source": "NVIDIA Cosmos-Reason1 + PHYSICS dataset taxonomy",
        "total_entries": len(library),
        "entries": library,
    }

    with open(args.output, "w", encoding="utf-8") as f:
        json.dump(output, f, indent=2, ensure_ascii=False)

    success = sum(1 for r in library if "_error" not in r)
    logger.info(
        f"Saved {len(library)} entries to {args.output} "
        f"({success} successful, {len(library)-success} failed)."
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Step 2.5a: Build Physical Event Library via LLM"
    )
    parser.add_argument(
        "--output",
        type=str,
        default="physical_event_library.json",
        help="Output JSON path for the knowledge base",
    )
    parser.add_argument("--model", type=str, default=DEFAULT_MODEL)
    parser.add_argument("--api_url", type=str, default=DEFAULT_API_URL)
    parser.add_argument("--api_key", type=str, default=DEFAULT_API_KEY)
    parser.add_argument(
        "--concurrent", type=int, default=8, help="Parallel LLM requests"
    )
    parser.add_argument(
        "--skip_errors",
        action="store_true",
        help="Exclude failed entries from output",
    )

    args = parser.parse_args()

    if os.name == "nt":
        asyncio.set_event_loop_policy(asyncio.WindowsSelectorEventLoopPolicy())

    asyncio.run(main(args))
