"""
select_baai200_library.py - KPR candidate pool from the Physics-aware-videos metadata.

1. The caption of every entry in metadata/dataset.jsonl is matched against keyword lists for the categories of the
   shared physical event library (plus Bounce_Restitution and Rolling_Motion). A category matches when one of its
   positive keywords appears and none of its negative keywords does.
2. Scene types without a measurable subject (ocean waves, rain, cooking, dancing, team sports, traffic, ...) are excluded.
3. Categories are visited from the least to the most represented; each contributes up to 10 recordings, ranked by the
   number of precise ("bonus") keyword matches and then by shorter caption.
4. The pool is filled to 200 recordings by bonus score.

Output: $BAAI_CANDIDATES (default selected_baai200.json); download with download_baai200.py, then select_baai_160.py.
"""
from huggingface_hub import hf_hub_download
import json, collections, random, os

random.seed(42)

# Load metadata
p = hf_hub_download('BAAI-DataCube/Physics-aware-videos', 'metadata/dataset.jsonl', repo_type='dataset')
all_entries = [json.loads(l) for l in open(p, encoding='utf-8')]
print(f"Total BAAI entries: {len(all_entries)}")

# Physics library 29 categories -> keywords mapping
# pos: at least one must appear; neg: if any appear, exclude; bonus: extra precision keywords
LIBRARY_CATEGORIES = {
    # Mechanics
    "Mechanics/Collision": {
        "pos": ["collide", "collision", "bounces off", "ball hits", "strikes another",
                "billiard", "snooker", "pool ball", "newton's cradle", "rebound",
                "cue ball", "impact with", "crash into"],
        "neg": ["car crash", "explosion", "tsunami", "traffic accident"],
        "bonus": ["billiard", "snooker", "newton's cradle", "pool ball", "cue ball", "rebound"],
    },
    "Mechanics/Sliding_Friction": {
        "pos": ["slides down", "sliding down", "glides down", "friction",
                "block slides", "object slides", "slides along", "skids to a stop",
                "dragged across", "pulled across a surface"],
        "neg": ["ice skating competition", "skiing trick", "snowboard trick"],
        "bonus": ["friction", "block slides", "object slides", "skids to a stop"],
    },
    "Mechanics/Projectile_Motion": {
        "pos": ["thrown into the air", "launches into", "catapult", "javelin",
                "arrow flies", "ball is thrown", "tosses into", "hurls",
                "kicked into the air", "projectile", "launched into the air",
                "basketball shot", "ball is shot"],
        "neg": ["bowling ball", "dart game", "baseball pitching"],
        "bonus": ["javelin", "arrow flies", "catapult", "projectile", "launched into the air"],
    },
    "Mechanics/Circular_Motion": {
        "pos": ["spinning", "rotates", "circular motion", "centrifugal",
                "spins around", "rotating disk", "gyroscope", "top spins",
                "carousel", "merry-go-round", "spinning wheel"],
        "neg": [],
        "bonus": ["gyroscope", "centrifugal", "rotating disk", "carousel"],
    },
    "Mechanics/Spring_Oscillation": {
        "pos": ["spring", "oscillat", "pendulum", "newton's cradle",
                "swings back and forth", "back and forth", "bungee",
                "elastic restoring", "vibrating spring"],
        "neg": [],
        "bonus": ["pendulum", "newton's cradle", "spring oscillat", "bungee", "vibrating spring"],
    },
    "Mechanics/Pulley_System": {
        "pos": ["pulley", "counterweight", "winch", "hoisted by rope",
                "block and tackle", "lifted by a rope and pulley",
                "mechanical advantage"],
        "neg": ["crane demolish", "wrecking ball"],
        "bonus": ["pulley", "counterweight", "block and tackle", "mechanical advantage"],
    },
    "Mechanics/Inclined_Plane": {
        "pos": ["rolls down a ramp", "rolls down the ramp", "rolls down a slope",
                "rolls down the slope", "ball rolls down", "marble rolls down",
                "rolls down a track", "rolls down the track", "down a wooden track",
                "down an incline", "rolls along a ramp", "down a curved track"],
        "neg": ["skateboard", "skiing", "snowboard", "bicycle down"],
        "bonus": ["marble rolls", "ball rolls down", "wooden track", "curved track", "incline"],
    },
    "Mechanics/Free_Fall": {
        "pos": ["free fall", "falls from height", "drops from height", "object falls vertically",
                "ball drops vertically", "gravity pulls down", "falls vertically",
                "dropped from a height", "free-fall"],
        "neg": ["parachute", "skydiving", "bungee jumping"],
        "bonus": ["free fall", "free-fall", "gravity pulls", "dropped from a height"],
    },
    "Mechanics/Rolling_Motion": {
        "pos": ["rolls along", "rolling along", "bowling ball rolls", "ball rolls across",
                "marble rolls across", "rolls smoothly", "rolling motion",
                "rolls across the surface", "rolls across the table",
                "wooden track with balls", "balls rolling"],
        "neg": ["rolls down"],
        "bonus": ["bowling ball rolls", "rolling motion", "rolls smoothly", "wooden track with balls"],
    },
    "Mechanics/Bounce_Restitution": {
        "pos": ["bounces", "bouncing ball", "rebounds from", "ball bounces", "drops and bounces",
                "elastic collision", "ball repeatedly bounces",
                "dropped onto surface and bounces"],
        "neg": [],
        "bonus": ["bouncing ball", "elastic collision", "drops and bounces", "repeatedly bounces"],
    },
    # Fluid Dynamics
    "Fluid_Dynamics/Buoyancy_Floating": {
        "pos": ["floats on water", "floating on", "buoyancy", "sinks into water",
                "submerges", "object floats", "boat floats",
                "density experiment", "archimedes", "denser than water"],
        "neg": ["swimming", "surfing", "diver", "swimming pool"],
        "bonus": ["buoyancy", "archimedes", "density experiment", "denser than water"],
    },
    "Fluid_Dynamics/Pouring_Flow": {
        "pos": ["liquid pours", "pours liquid", "water pours into", "honey pours",
                "oil pours", "viscous liquid pours", "liquid flows into",
                "fluid pours", "water is poured into a container"],
        "neg": ["rain", "waterfall", "river current", "ocean", "flood", "wave"],
        "bonus": ["honey pours", "viscous liquid", "oil pours"],
    },
    "Fluid_Dynamics/Splashing_Impact": {
        "pos": ["stone drops into water", "rock thrown into water",
                "ball drops into water", "object falls into water",
                "dropped into a pool", "stone thrown into a pond",
                "rock into a pond", "pebble into water"],
        "neg": ["ocean wave", "waves crashing", "rain", "river", "waterfall", "flood", "surf"],
        "bonus": ["stone drops into water", "rock thrown into water", "pebble into water"],
    },
    "Fluid_Dynamics/Pressure_Difference": {
        "pos": ["pressure difference", "vacuum pump", "compressed air demo",
                "air pressure experiment", "balloon is inflated by pressure",
                "atmospheric pressure", "water pressure experiment",
                "syringe demonstrates", "magdeburg hemisphere", "vacuum chamber"],
        "neg": ["vacuum cleaner", "hoover", "dust", "carpet", "cleaning",
                "pressure washer", "spray", "inflate balloon party",
                "suction cup", "suction cups", "mechanical toy", "toy features",
                "assembl", "wooden frame", "construction"],
        "bonus": ["vacuum pump", "pressure difference", "atmospheric pressure",
                  "magdeburg", "vacuum chamber", "syringe demonstrates"],
    },
    "Fluid_Dynamics/Viscous_Flow": {
        "pos": ["honey flows", "syrup flows", "molasses", "viscous liquid",
                "thick liquid flows", "lava flows slowly", "viscosity",
                "non-newtonian", "slime flows"],
        "neg": [],
        "bonus": ["honey flows", "viscous", "non-newtonian", "lava flows"],
    },
    # Electromagnetism
    "Electromagnetism/Electric_Circuit": {
        "pos": ["electric circuit", "electrical circuit", "current flows through wire",
                "led lights up when circuit", "battery powers a circuit",
                "circuit is complete", "circuit board lights",
                "connect the circuit", "close the circuit"],
        "neg": ["circuit board crush", "circuit board shred", "circuit board recycl",
                "crush", "shred", "recycle", "traffic light", "neon sign",
                "solar panel installation"],
        "bonus": ["circuit is complete", "current flows through wire", "led lights up"],
    },
    "Electromagnetism/Magnetic_Attraction_Repulsion": {
        "pos": ["magnet attracts", "magnetic field", "iron filings", "compass needle",
                "magnets repel", "magnet picks up", "attracts metal objects",
                "bar magnet", "horseshoe magnet"],
        "neg": [],
        "bonus": ["iron filings", "compass needle", "magnets repel", "bar magnet"],
    },
    "Electromagnetism/Electromagnetic_Induction": {
        "pos": ["electromagnetic induction", "coil and magnet", "faraday",
                "induced current", "moving magnet through coil", "generator coil",
                "solenoid"],
        "neg": [],
        "bonus": ["electromagnetic induction", "faraday", "induced current", "solenoid"],
    },
    "Electromagnetism/Electrostatic_Interaction": {
        "pos": ["static electricity", "electrostatic", "charged balloon",
                "static charge", "van de graaff",
                "rubbed balloon", "attracted by static", "electrostatic charge",
                "triboelectric", "charged rod"],
        "neg": ["firework", "sparkler", "welding", "grinding", "fire", "flame",
                "explosion", "rocket", "combustion", "lightning", "thunder"],
        "bonus": ["van de graaff", "static electricity", "electrostatic", "charged balloon"],
    },
    # Optics
    "Optics/Reflection": {
        "pos": ["mirror reflects", "light reflects off a mirror", "reflection of light",
                "mirror image", "light bounces off mirror", "reflected light beam",
                "specular reflection", "reflects off a mirror", "laser reflects",
                "light reflected by"],
        "neg": ["selfie mirror", "bathroom mirror routine",
                "sunlight reflects off water", "reflects off the water",
                "reflects off the sea", "reflects off the ocean", "coastal",
                "walking through water", "pond reflection"],
        "bonus": ["specular reflection", "light bounces off", "reflection of light",
                  "laser reflects"],
    },
    "Optics/Refraction": {
        "pos": ["refract", "bends light", "prism splits light", "pencil in water",
                "straw in glass", "light bends when", "rainbow from prism",
                "spoon in water appears bent", "light through glass"],
        "neg": [],
        "bonus": ["refraction", "prism splits", "bends light", "appears bent"],
    },
    "Optics/Shadow_Formation": {
        "pos": ["casts a shadow", "shadow forms", "shadow moves with",
                "eclipse", "light source creates shadow", "object blocks light",
                "shadow of the", "umbra", "sundial"],
        "neg": ["shadow boxing", "shadow puppet show"],
        "bonus": ["eclipse", "umbra", "sundial", "casts a shadow"],
    },
    "Optics/Lens_Focusing": {
        "pos": ["magnifying glass focuses", "lens focuses light",
                "converging lens", "diverging lens", "convex lens",
                "magnifying glass burns", "light converges through lens",
                "focus light through lens", "lens demonstration",
                "focal length", "light focused by lens"],
        "neg": ["crush", "shred", "sword", "cut", "break", "smash",
                "fish", "fishing", "pond", "aquarium", "swimming", "underwater",
                "gopro", "security camera", "camera records", "camera films",
                "camera captures", "dash cam", "waves", "ocean", "pier",
                "jump", "gym", "sport", "workout"],
        "bonus": ["converging lens", "magnifying glass burns",
                  "light converges through lens", "focal length",
                  "lens demonstration"],
    },
    # Thermodynamics
    "Thermodynamics/Combustion": {
        "pos": ["candle flame", "candle burns", "wood burns", "burning wood",
                "matches light", "fire spreads", "combustion", "ignites",
                "flame burns through", "paper catches fire",
                "fire is lit", "torch burns", "flame flickers"],
        "neg": ["firework", "fireworks", "bonfire party", "explosion", "pyrotechnic",
                "rocket", "sparkler", "welding sparks", "grinding sparks"],
        "bonus": ["combustion", "ignites", "candle burns", "flame burns through",
                  "fire is lit"],
    },
    "Thermodynamics/Heat_Transfer": {
        "pos": ["heat transfer", "conducts heat", "thermal conduction", "metal heats up",
                "temperature rises", "heat flows from", "warm to cool",
                "infrared camera shows", "heat spreads through"],
        "neg": [],
        "bonus": ["heat transfer", "thermal conduction", "heat flows", "infrared camera"],
    },
    "Thermodynamics/Melting_Solidification": {
        "pos": ["ice melts", "wax melts", "metal melts", "chocolate melts",
                "solidifies", "solidification", "freezes solid",
                "melting point", "ice cube melts"],
        "neg": [],
        "bonus": ["ice melts", "wax melts", "solidification", "melting point"],
    },
    "Thermodynamics/Evaporation_Condensation": {
        "pos": ["water evaporates", "evaporation", "condensation forms",
                "steam rises from", "vapor rises", "dew forms",
                "droplets condense", "evaporates quickly"],
        "neg": [],
        "bonus": ["evaporation", "condensation forms", "vapor rises", "dew forms"],
    },
    "Thermodynamics/Thermal_Expansion": {
        "pos": ["thermal expansion", "expands when heated", "bimetallic strip",
                "balloon expands when heated", "heated gas expands",
                "metal expands", "contracts when cooled"],
        "neg": [],
        "bonus": ["thermal expansion", "bimetallic", "expands when heated"],
    },
    # Wave_Acoustics
    "Wave_Acoustics/Sound_Propagation": {
        "pos": ["sound wave", "vibration creates sound", "tuning fork",
                "speaker vibrates", "acoustic wave", "sound travels through",
                "resonance chamber", "sound propagation"],
        "neg": [],
        "bonus": ["tuning fork", "sound wave", "acoustic wave", "sound propagation"],
    },
    "Wave_Acoustics/Resonance": {
        "pos": ["resonance", "resonates", "standing wave", "harmonic frequency",
                "natural frequency", "wine glass vibrates", "chladni",
                "objects resonate", "resonant frequency"],
        "neg": [],
        "bonus": ["resonance", "chladni", "standing wave", "wine glass vibrates"],
    },
    "Wave_Acoustics/Doppler_Effect": {
        "pos": ["doppler effect", "frequency changes as", "pitch changes as",
                "siren approaches", "ambulance passes", "doppler shift",
                "moving sound source"],
        "neg": [],
        "bonus": ["doppler effect", "doppler shift", "siren approaches"],
    },
}

# Global exclusions — scene types that are fundamentally unsuitable
GLOBAL_EXCLUDE = [
    "ocean waves", "waves crashing on", "rain pours", "rainfall", "heavy rain",
    "waterfall", "river flows",
    "cooking recipe", "chopping vegetables", "food preparation",
    "dancing", "ballet", "workout routine", "gym exercise",
    "soccer match", "football game", "basketball game", "crowd cheering",
    "swimming competition", "surfing competition", "skateboard trick",
    "traffic", "cars driving",
]

def score_entry(cap_lower, cat_info):
    for neg in cat_info.get("neg", []):
        if neg in cap_lower:
            return False, 0
    matched = any(kw in cap_lower for kw in cat_info["pos"])
    if not matched:
        return False, 0
    bonus = sum(1 for kw in cat_info.get("bonus", []) if kw in cap_lower)
    return True, bonus

print("Scanning all entries...")
for e in all_entries:
    cap = e["caption"].lower()
    e["_cats"] = []
    e["_bonus"] = 0
    if any(ex in cap for ex in GLOBAL_EXCLUDE):
        e["_excluded"] = True
        continue
    e["_excluded"] = False
    for cat, info in LIBRARY_CATEGORIES.items():
        matched, bonus = score_entry(cap, info)
        if matched:
            e["_cats"].append(cat)
            e["_bonus"] += bonus

candidates = [e for e in all_entries if not e.get("_excluded") and e["_cats"]]
print(f"Candidates with library match (not excluded): {len(candidates)}")

per_cat = collections.defaultdict(list)
for e in candidates:
    for cat in e["_cats"]:
        per_cat[cat].append(e)

print("\nCandidates per category:")
for cat in sorted(LIBRARY_CATEGORIES.keys()):
    print(f"  {cat}: {len(per_cat[cat])}")

# Greedy selection: rarest categories first, max 10 per category
MAX_PER_CAT = 10
selected_set = set()
selected = []
cat_counts = collections.defaultdict(int)

sorted_cats = sorted(LIBRARY_CATEGORIES.keys(), key=lambda c: len(per_cat[c]))

for cat in sorted_cats:
    cat_cands = per_cat[cat]
    # Sort: bonus desc, then caption length asc (shorter = more focused description)
    cat_cands.sort(key=lambda e: (-e["_bonus"], len(e["caption"])))
    added = 0
    for e in cat_cands:
        if added >= MAX_PER_CAT:
            break
        vid = e["video"]
        if vid in selected_set:
            cat_counts[cat] += 1
            added += 1
            continue
        selected_set.add(vid)
        selected.append({**e, "_primary_cat": cat})
        cat_counts[cat] += 1
        added += 1

print(f"\nInitial selected: {len(selected)}")

# Fill to 200 if needed
if len(selected) < 200:
    remaining = [e for e in candidates if e["video"] not in selected_set]
    remaining.sort(key=lambda e: -e["_bonus"])
    for e in remaining:
        if len(selected) >= 200:
            break
        selected_set.add(e["video"])
        selected.append({**e, "_primary_cat": e["_cats"][0] if e["_cats"] else "misc"})

print(f"Final selected: {len(selected)}")

out = []
for i, e in enumerate(selected):
    out.append({
        "id": f"BAAI_v3_{i+1:03d}",
        "video": e["video"],
        "caption": e["caption"],
        "physics_library_cats": e["_cats"],
        "primary_cat": e.get("_primary_cat", ""),
        "bonus_score": e["_bonus"],
    })

OUT = os.environ.get("BAAI_CANDIDATES", "selected_baai200.json")
with open(OUT, "w", encoding="utf-8") as f:
    json.dump(out, f, indent=2, ensure_ascii=False)
print(f"\nSaved {len(out)} entries -> {OUT}")

all_cats = collections.Counter(cat for e in out for cat in e["physics_library_cats"])
print("\nFinal event distribution:")
for cat, cnt in all_cats.most_common():
    print(f"  {cat}: {cnt}")

print("\nSample selections per category (top 2 each):")
shown = collections.defaultdict(int)
for e in out:
    cat = e["primary_cat"]
    if shown[cat] < 2:
        print(f"  [{cat}] {e['video']} | {e['caption'][:100]}")
        shown[cat] += 1
