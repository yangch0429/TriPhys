"""
Select 200 EgoExo4D takes from <60s ego_pose annotated ones,
prioritizing physical event diversity based on atomic_descriptions text.
"""
import json, os, re
from collections import defaultdict

ROOT = os.environ["EGOEXO_RAW_ROOT"]   # raw EgoExo4D download (takes.json, annotations/)
ANNO_DIR = os.path.join(ROOT, "annotations", "ego_pose", "train", "body", "annotation")
TAKES_JSON = os.path.join(ROOT, "takes.json")
ATOMIC_TRAIN = os.path.join(ROOT, "annotations", "atomic_descriptions_train.json")
ATOMIC_VAL   = os.path.join(ROOT, "annotations", "atomic_descriptions_val.json")
OUT_FILE = os.environ.get("EGOEXO_SELECTION", "selected_egoexo200.json")

TARGET = 200
MAX_DURATION = 60.0

# Physics event keyword mapping
PHYSICS_EVENTS = {
    "Projectile_Motion":    ["shoot", "shoots", "throw", "throws", "toss", "tosses", "launch", "jump shot", "layup", "free throw", "pass", "passes", "kick", "kicks", "hit", "hits"],
    "Collision_Impact":     ["bounce", "bounces", "collide", "block", "blocks", "catch", "catches", "rebound", "rebounds", "deflect", "land", "lands", "tackle"],
    "Rotation_Spin":        ["spin", "spins", "rotate", "rotates", "turn", "turns", "pivot", "pivots", "dribble", "dribbles", "roll", "rolls"],
    "Pendulum_Oscillation": ["swing", "swings", "oscillate", "rock", "rocks"],
    "Friction_Sliding":     ["slide", "slides", "slip", "slips", "skid", "drag", "drags", "crawl"],
    "Elastic_Deformation":  ["stretch", "bend", "bends", "compress", "flex", "flexes"],
    "Lifting_Lowering":     ["lift", "lifts", "raise", "raises", "pick up", "put down", "lower", "lowers", "carry", "carries"],
    "Pouring_Flow":         ["pour", "pours", "spill", "spills", "fill", "fills", "liquid", "water", "drink", "drinks"],
    "Splashing_Impact":     ["splash", "splashes", "drop", "drops into water"],
    "Drag_Deceleration":    ["slow", "decelerate", "brake", "brakes", "stop", "stops"],
    "Balance_Stability":    ["balance", "balances", "stabilize", "lean", "leans", "stand", "stands"],
    "Gear_Mechanism":       ["tighten", "tightens", "loosen", "loosens", "screw", "screws", "wrench", "bolt", "bolts", "pedal", "pedals", "wheel", "chain"],
    "Reflection":           ["reflect", "mirror", "shadow"],
    "Impact_Force":         ["push", "pushes", "pull", "pulls", "press", "presses", "force", "apply"],
    "Free_Fall":            ["fall", "falls", "drop", "drops", "gravity"],
}

def get_descriptions(uid, atomic_data):
    """Return list of description texts for a take_uid."""
    ann_list = atomic_data.get(uid, [])
    texts = []
    for ann in ann_list:
        if ann.get("rejected"):
            continue
        for d in ann.get("descriptions", []):
            t = d.get("text", "")
            if t:
                texts.append(t.lower())
    return texts

def detect_physics_events(texts):
    """Return set of physics event types present in the text."""
    events = set()
    combined = " ".join(texts)
    for event, keywords in PHYSICS_EVENTS.items():
        for kw in keywords:
            if kw in combined:
                events.add(event)
                break
    return events

# Load data
with open(TAKES_JSON) as f:
    takes = json.load(f)

annotated = {fn.replace(".json", "") for fn in os.listdir(ANNO_DIR) if fn.endswith(".json")}
print(f"Ego_pose annotated takes: {len(annotated)}")

# Load atomic descriptions
atomic_data = {}
for fpath in [ATOMIC_TRAIN, ATOMIC_VAL]:
    if os.path.exists(fpath):
        raw = json.load(open(fpath))
        atomic_data.update(raw.get("annotations", {}))
print(f"Atomic descriptions covers: {len(atomic_data)} takes")

# Filter: annotated + <60s
candidates = [
    t for t in takes
    if t["take_uid"] in annotated and 0 < t.get("duration_sec", 0) < MAX_DURATION
]
print(f"Candidates (<{MAX_DURATION}s, ego_pose): {len(candidates)}")

# Detect physics events for each candidate
print("Detecting physics events from descriptions...")
for t in candidates:
    texts = get_descriptions(t["take_uid"], atomic_data)
    t["_physics_events"] = detect_physics_events(texts)
    t["_n_events"] = len(t["_physics_events"])
    t["_n_descriptions"] = len(texts)

# Print event coverage
event_counts = defaultdict(int)
for t in candidates:
    for ev in t["_physics_events"]:
        event_counts[ev] += 1

print("\nPhysics event coverage (candidates):")
for ev, cnt in sorted(event_counts.items(), key=lambda x: -x[1]):
    print(f"  {ev}: {cnt} takes")

# Select 200 with physics event diversity
# Strategy: greedy - always pick the take that adds the most new event coverage
# Sort by: (1) num physics events desc, (2) duration desc as tiebreak
candidates.sort(key=lambda x: (-x["_n_events"], -x.get("duration_sec", 0)))

selected = []
covered_events = defaultdict(int)

# First pass: prefer takes with rarest events
for t in sorted(candidates, key=lambda x: -x["_n_events"]):
    if len(selected) >= TARGET:
        break
    # Score: sum of (1 / (covered_events[ev] + 1)) for each event in take
    score = sum(1.0 / (covered_events[ev] + 1) for ev in t["_physics_events"]) if t["_physics_events"] else 0.01
    t["_diversity_score"] = score

candidates.sort(key=lambda x: -x["_diversity_score"])

for t in candidates:
    if len(selected) >= TARGET:
        break
    selected.append(t)
    for ev in t["_physics_events"]:
        covered_events[ev] += 1

# If not enough, fill with remaining sorted by n_events
remaining = [t for t in candidates if t not in selected]
remaining.sort(key=lambda x: -x["_n_events"])
for t in remaining:
    if len(selected) >= TARGET:
        break
    selected.append(t)

print(f"\nSelected: {len(selected)} takes")

# Stats
final_events = defaultdict(int)
for t in selected:
    for ev in t["_physics_events"]:
        final_events[ev] += 1

print("\nPhysics event distribution in selected:")
for ev, cnt in sorted(final_events.items(), key=lambda x: -x[1]):
    print(f"  {ev}: {cnt} takes")

task_counts = defaultdict(int)
for t in selected:
    task_counts[t.get("parent_task_name", "Unknown")] += 1
print("\nTask distribution in selected:")
for task, cnt in sorted(task_counts.items(), key=lambda x: -x[1]):
    print(f"  {task}: {cnt}")

dur = [t["duration_sec"] for t in selected]
print(f"\nDuration stats: mean={sum(dur)/len(dur):.1f}s min={min(dur):.1f}s max={max(dur):.1f}s")

no_events = sum(1 for t in selected if not t["_physics_events"])
print(f"Takes with no detected physics events: {no_events}")

# Build output
output = []
for t in selected:
    output.append({
        "take_uid": t["take_uid"],
        "take_name": t["take_name"],
        "task_name": t["task_name"],
        "parent_task_name": t.get("parent_task_name", ""),
        "duration_sec": round(t["duration_sec"], 2),
        "best_exo": t.get("best_exo", ""),
        "university_name": t.get("university_name", ""),
        "physics_events": sorted(t["_physics_events"]),
        "n_descriptions": t["_n_descriptions"],
        "has_trimmed_trajectory": t.get("has_trimmed_trajectory", False),
    })

with open(OUT_FILE, "w") as f:
    json.dump(output, f, indent=2, ensure_ascii=False)
print(f"\nSaved to {OUT_FILE}")
