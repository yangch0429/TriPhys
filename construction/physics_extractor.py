"""
physics_extractor.py  —  CV-based physics extraction (v2)

Architecture:
  Calibration  ->  Route A (MediaPipe Pose)  ->  Route B (YOLO + ByteTrack trajectory)
                                                ->  Route C (merge + confidence)

Calibration layer:
  Derives pixel/metre scale from known reference objects in the scene:
  1. Sport-specific known dimensions (hurdle height, basketball rim, javelin length, etc.)
  2. Athlete height from pose (fallback)
  Outputs: scale_px_per_m, calibration_source, calibration_confidence

Route A  MediaPipe Pose Tasks API  ->  athlete height, joint angles, velocity
Route B  YOLOv8 + ByteTrack        ->  object tracks, projectile trajectory, launch angle/speed
Route C  Merge + confidence        ->  uncertain_quantities list

Usage (standalone):
    python physics_extractor.py --clip path/to/clip.mp4 --sport hurdling

As a library:
    from physics_extractor import augment_opq
    augmented = augment_opq(clip_path, fps, vlm_opq_dict, sport_category="hurdling")
"""

import os
# Remove SOCKS proxy env var so httpx/openai clients work without socksio
for _k in ("ALL_PROXY", "all_proxy"):
    os.environ.pop(_k, None)
import cv2
import numpy as np
import logging
import math
import json
import argparse
from typing import Dict, Any, List, Optional, Tuple

logger = logging.getLogger("physics_extractor")

# ---------------------------------------------------------------------------
# Optional dependencies
# ---------------------------------------------------------------------------

try:
    import mediapipe as mp
    from mediapipe.tasks import python as _mp_tasks
    from mediapipe.tasks.python import vision as _mp_vision
    _MEDIAPIPE_OK = True
except ImportError:
    _MEDIAPIPE_OK = False
    logger.info("mediapipe not installed — Route A disabled")

_MODEL_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "pose_landmarker_full.task")

try:
    from scipy.optimize import curve_fit as _scipy_curve_fit
    _SCIPY_OK = True
except ImportError:
    _SCIPY_OK = False

try:
    from ultralytics import YOLO as _YOLO
    _YOLO_OK = True
except ImportError:
    _YOLO_OK = False
    logger.info("ultralytics not installed — Route B YOLO disabled")

# YOLOv11-pose (loaded lazily)
_YOLO_POSE_OK = False
_yolo_pose_model = None

# COCO keypoint index → MediaPipe landmark index (for _get_lm compatibility)
_COCO_TO_MP: Dict[int, int] = {
    0: 0,   # nose
    5: 11,  # left_shoulder
    6: 12,  # right_shoulder
    7: 13,  # left_elbow
    8: 14,  # right_elbow
    9: 15,  # left_wrist
    10: 16, # right_wrist
    11: 23, # left_hip
    12: 24, # right_hip
    13: 25, # left_knee
    14: 26, # right_knee
    15: 27, # left_ankle
    16: 28, # right_ankle
}


class _KP:
    """Minimal landmark wrapper so YOLOv11-pose keypoints work with _get_lm()."""
    __slots__ = ("x", "y", "visibility")

    def __init__(self, x: float, y: float, vis: float) -> None:
        self.x = float(x)
        self.y = float(y)
        self.visibility = float(vis)

# ---------------------------------------------------------------------------
# Sport-specific calibration priors (metres)
# ---------------------------------------------------------------------------
# Maps sport category -> list of (object_label_keywords, known_height_m, known_width_m)
# Used to derive pixel/metre scale from detected objects.
SPORT_CALIBRATION = {
    "hurdling": [
        (["hurdle"], 1.067, 1.18),   # men's 110m hurdle: 1.067 m high, 1.18 m wide
    ],
    "highjump": [
        (["bar", "crossbar"], None, 4.0),  # bar width 4 m
    ],
    "longjump": [
        (["board", "takeoff"], 0.2, 1.22),  # takeoff board 20 cm high, 1.22 m wide
    ],
    "basketball": [
        (["rim", "basket", "hoop"], 3.05, 0.46),  # rim height 3.05 m, diameter 0.46 m
        (["backboard"], 1.07, 1.83),
    ],
    "volleyball": [
        (["net"], 2.43, None),   # men's net height
    ],
    "tennis": [
        (["net"], 0.914, None),
    ],
    "badminton": [
        (["net"], 1.524, None),
    ],
    "javelin": [
        (["javelin"], None, 2.6),   # men's javelin length 2.6-2.7 m
    ],
    "shotput": [
        (["circle", "ring"], None, 2.135),  # throwing circle diameter 2.135 m
    ],
    "swimming": [
        (["lane", "lane_line"], None, 2.5),  # lane width 2.5 m
    ],
    "gymnastics": [
        (["beam", "balance_beam"], 1.25, 5.0),  # beam height 1.25 m, length 5 m
        (["mat", "floor"], None, 12.0),
    ],
    "soccer": [
        (["goal", "goalpost"], 2.44, 7.32),
    ],
    "football": [
        (["goal", "goalpost"], 3.05, 5.64),
    ],
    "baseball": [
        (["base", "plate"], 0.038, 0.38),
    ],
    "archery": [
        (["target", "bullseye"], None, 1.22),  # 70 m target diameter 1.22 m
    ],
    "polevault": [
        (["bar", "crossbar"], None, 4.5),
    ],
    "discus": [
        (["circle", "ring"], None, 2.5),
    ],
    "hammerthrow": [
        (["circle", "ring"], None, 2.135),
    ],
    "weightlifting": [
        (["bar", "barbell"], None, 2.2),  # standard barbell 2.2 m
    ],
    "rowing": [
        (["lane"], None, 13.5),  # lane width 13.5 m
    ],
    "cycling": [
        (["wheel", "bike"], None, 0.7),  # wheel diameter ~0.7 m
    ],
    "skiing": [
        (["gate", "pole"], None, 0.75),  # gate width ~0.75 m
    ],
    "iceskating": [],
    "figureskating": [],
    "boxing": [
        (["ring", "ropes"], None, 6.1),  # ring 6.1 m
    ],
    "wrestling": [
        (["mat", "circle"], None, 9.0),  # wrestling mat circle 9 m
    ],
    "judo": [
        (["mat", "tatami"], None, 10.0),
    ],
    "taekwondo": [
        (["mat"], None, 10.0),
    ],
    "fencing": [
        (["piste", "strip"], None, 14.0),  # piste length 14 m
    ],
    "cheerleading": [],
    "default": [],
}

# YOLO COCO class names used for bbox detection / calibration lookup
YOLO_SPORTS_CLASSES = {
    0:  "person",
    2:  "car",
    3:  "motorcycle",
    5:  "bus",
    7:  "truck",
    9:  "traffic light",
    10: "fire hydrant",
    11: "stop sign",
    15: "cat",
    16: "dog",
    24: "backpack",
    25: "umbrella",
    32: "sports ball",
    38: "tennis racket",
    39: "bottle",
    60: "dining table",
    62: "tv",
    63: "laptop",
    67: "cell phone",
    77: "clock",
    87: "bicycle",
}

# Non-sport common objects with well-known real-world dimensions
# Format: label -> (keyword_list, height_m_or_None, width_m_or_None, variance_pct)
GENERAL_REF_OBJECTS = {
    "car":           (["car", "sedan", "vehicle", "automobile"],  1.50, 4.50, 10),
    "door":          (["door", "doorway"],                        2.05, 0.90, 8),
    "dining_table":  (["dining table", "table"],                  0.75, None, 15),
    "stop_sign":     (["stop sign"],                              0.75, 0.75, 3),
    "traffic_light": (["traffic light", "signal"],                1.00, None, 10),
    "fire_hydrant":  (["fire hydrant"],                           0.60, None, 10),
    "bicycle":       (["bicycle", "bike"],                        1.10, None, 12),
    "bus":           (["bus"],                                    3.00, 12.00, 8),
    "truck":         (["truck"],                                  2.50, None, 15),
    "bottle":        (["bottle", "water bottle"],                 0.25, None, 20),
}

# Vehicle / traffic scene reference objects
VEHICLE_REF_OBJECTS = {
    "sedan":         (["car", "sedan", "automobile"],             1.50, 4.50, 10),
    "suv":           (["suv", "crossover"],                       1.70, 4.70, 10),
    "bus":           (["bus", "coach"],                           3.00, 12.00, 8),
    "truck":         (["truck", "lorry"],                         2.50, 8.00, 15),
    "motorcycle":    (["motorcycle", "motorbike"],                1.10, 2.10, 12),
    "bicycle":       (["bicycle", "bike"],                        1.10, 1.80, 12),
    "lane_width":    (["lane", "road_lane"],                      None, 3.75, 5),
    "stop_sign":     (["stop sign"],                              0.75, 0.75, 3),
    "traffic_light": (["traffic light"],                          1.00, None, 10),
    "traffic_cone":  (["cone", "traffic cone"],                   0.70, None, 10),
    "license_plate": (["license plate", "number plate"],          0.12, 0.52, 5),
}

# Indoor / daily life reference objects
INDOOR_REF_OBJECTS = {
    "door":          (["door", "doorway"],                        2.05, 0.90, 8),
    "dining_table":  (["dining table", "table"],                  0.75, 1.20, 15),
    "chair":         (["chair"],                                  0.85, 0.50, 15),
    "sofa":          (["sofa", "couch"],                          0.85, 2.00, 15),
    "bed":           (["bed"],                                    0.55, 1.40, 10),
    "refrigerator":  (["refrigerator", "fridge"],                 1.70, 0.70, 10),
    "bottle":        (["bottle", "water bottle"],                 0.25, None, 20),
    "mug":           (["mug", "cup"],                             0.10, None, 20),
    "laptop":        (["laptop"],                                 0.02, 0.35, 10),
    "keyboard":      (["keyboard"],                               0.03, 0.45, 8),
    "book":          (["book"],                                   0.02, 0.21, 15),
    "tv":            (["tv", "television", "monitor"],            0.60, 1.07, 20),
    "microwave":     (["microwave"],                              0.30, 0.50, 10),
    "sink":          (["sink"],                                   0.20, 0.60, 15),
    "toilet":        (["toilet"],                                 0.40, 0.37, 10),
    "bathtub":       (["bathtub", "bath"],                        0.55, 1.70, 10),
}

# Human body anthropometric priors (mean ± variance)
HUMAN_BODY_PRIORS = {
    "adult_height_m":       (1.75, 0.10),   # mean, std
    "adult_shoulder_w_m":   (0.45, 0.05),
    "adult_hand_length_m":  (0.19, 0.02),
    "adult_stride_m":       (0.75, 0.10),
    "child_height_m":       (1.20, 0.20),   # rough estimate for children
}

# Calibration source types and their reliability tiers
# source_type -> (tier, confidence_range)
CALIBRATION_SOURCE_TIERS = {
    "sensor_gt":        ("GT_A", (0.97, 0.99)),
    "computed_from_gt": ("GT_B", (0.92, 0.96)),
    "fused_gt":         ("GT_B", (0.88, 0.93)),
    "depth_map":        ("A",    (0.88, 0.95)),
    "camera_calib":     ("A",    (0.85, 0.92)),
    "known_object":     ("A",    (0.75, 0.88)),
    "human_prior":      ("B",    (0.55, 0.75)),
    "geometry":         ("B",    (0.50, 0.70)),
    "web_search_exact": ("B",    (0.80, 0.90)),
    "web_search_range": ("B",    (0.70, 0.85)),
    "vlm_estimate":     ("C",    (0.35, 0.55)),
    "scale_free":       ("C",    (0.40, 0.65)),  # relative-only, no absolute scale
    "frame_height":     ("D",    (0.20, 0.30)),
}

# ---------------------------------------------------------------------------
# MediaPipe landmark indices
# ---------------------------------------------------------------------------
_LM = {
    "nose": 0,
    "left_shoulder": 11, "right_shoulder": 12,
    "left_elbow": 13,    "right_elbow": 14,
    "left_wrist": 15,    "right_wrist": 16,
    "left_hip": 23,      "right_hip": 24,
    "left_knee": 25,     "right_knee": 26,
    "left_ankle": 27,    "right_ankle": 28,
    "left_heel": 29,     "right_heel": 30,
    "left_foot_index": 31, "right_foot_index": 32,
}

_SHOULDER_ANKLE_FRAC = 0.82
_HEIGHT_PRIOR_M = 1.75


# ===========================================================================
# Tier design — measurement reliability classification
# ===========================================================================
#
# GT_A  ±1%   — direct sensor GT (CAN bus, LiDAR, MoCap, IMU)
# GT_B  ±3%   — computed from GT (derived velocity, relative position)
# Tier A  ±5%   — depth map / camera calibration / known ref object (stable view)
# Tier B  ±12%  — vehicle/person bbox + prior, or ref_obj with mediocre view
# Tier C  ±25%  — pose-only / weak / scale-free (relative values only)
# Tier D  —     — rejected (too few frames, severe occlusion, etc.)
#
# scale_provenance values (new unified set):
#   "sensor_gt"         — direct sensor measurement (A-class datasets)
#   "computed_from_gt"  — derived from sensor GT
#   "depth_map"         — RGB-D depth provides metric scale
#   "camera_calib"      — known camera intrinsics/extrinsics
#   "independent_ref"   — ref object from SPORT/VEHICLE/INDOOR calibration tables
#   "human_prior"       — person bbox + population prior
#   "geometry"          — vanishing point / ground plane estimation
#   "scale_free"        — relative/angular values only (B2 class)
#   "frame_heuristic"   — frame-height fallback (very weak)
#   "none"              — no calibration


def build_quantity_metadata(
    value: float,
    unit: str,
    source_type: str,
    gt_source: str = "",
    derivation_type: str = "direct",
    measurement_range: Optional[Tuple[float, float]] = None,
) -> Dict[str, Any]:
    """
    Build a standardized quantity metadata dict with provenance fields.
    Used for all extracted physical quantities.

    source_type: one of CALIBRATION_SOURCE_TIERS keys
    gt_source: dataset name / sensor name (e.g. "nuscenes_canbus", "hoi4d_depth")
    derivation_type: "direct" | "computed_from_gt" | "cv_inferred"
    """
    tier_info = CALIBRATION_SOURCE_TIERS.get(source_type, ("C", (0.40, 0.60)))
    tier, conf_range = tier_info
    confidence = (conf_range[0] + conf_range[1]) / 2

    meta: Dict[str, Any] = {
        "value": value,
        "unit": unit,
        "source_type": source_type,
        "derivation_type": derivation_type,
        "reliability_tier": tier,
        "confidence": round(confidence, 3),
    }
    if gt_source:
        meta["gt_source"] = gt_source
    if measurement_range is not None:
        meta["measurement_range"] = list(measurement_range)
    return meta
#
# motion_observability_class values:
#   "planar"            — motion mostly in image plane, metric speed reliable
#   "depth_dominant"    — subject approaching / receding (bbox area change > 30%)
#   "ego_motion"        — camera is moving fast (optical flow dominant)
#   "unknown"           — not enough info

_TIER_A_TOLERANCE = 0.05
_TIER_B_TOLERANCE = 0.12
_TIER_C_TOLERANCE = 0.25

_SCALE_PROVENANCE_MAP = {
    "ref_object":          "independent_ref",
    "ref_object_width":    "independent_ref",
    "yolo_person_bbox_vlm_height":        "vlm_person_prior",
    "yolo_person_bbox_population_prior":  "population_prior",
    "pose_height_with_vlm_prior":         "vlm_person_prior",
    "pose_height_population_prior":       "population_prior",
    "frame_height_prior":                 "frame_heuristic",
    "none":                               "none",
}

def _scale_provenance(calibration_source: str) -> str:
    """Map calibration_source string to a provenance category."""
    cs = calibration_source or "none"
    for prefix, prov in _SCALE_PROVENANCE_MAP.items():
        if cs.startswith(prefix):
            return prov
    return "none"


def _motion_observability(dets: List, frame_h: int = 0) -> str:
    """Classify motion observability from a list of YOLO detections.

    dets: list of (frame_idx, cx, cy, bw, bh, cls_id) tuples.
    Returns one of: planar / depth_dominant / unknown.
    """
    if not dets or len(dets) < 4:
        return "unknown"
    dets_sorted = sorted(dets, key=lambda d: d[0])
    areas = [(d[3] * d[4]) for d in dets_sorted if d[3] > 0 and d[4] > 0]
    if len(areas) < 4:
        return "unknown"
    a_first = float(np.median(areas[:max(1, len(areas) // 4)]))
    a_last  = float(np.median(areas[-max(1, len(areas) // 4):]))
    if a_first < 1e-6:
        return "unknown"
    area_ratio = a_last / a_first
    if area_ratio > 1.30 or area_ratio < 0.77:
        return "depth_dominant"
    return "planar"


def _assign_tier(
    item: Dict[str, Any],
    calibration: Dict[str, Any],
    motion_class: str = "unknown",
) -> Dict[str, Any]:
    """Assign measurement_tier, scale_provenance, and _measurement_range to a CV item.

    Modifies item in-place and returns it.
    """
    prov = _scale_provenance(calibration.get("calibration_source", "none"))
    item["scale_provenance"] = prov
    item["motion_observability_class"] = motion_class

    # Depth-dominant motion → suppress metric speed, downgrade tier
    if motion_class == "depth_dominant" and item.get("estimated_speed_ms") is not None:
        item["speed_estimation_status"] = "suppressed_depth_motion"
        item["estimated_speed_ms"] = None

    conf = float(item.get("estimation_confidence", 0.0))

    # Tier D: rejected outright
    if prov == "none" or prov == "frame_heuristic":
        item["measurement_tier"] = "D"
        item["_measurement_range_pct"] = None
        return item

    # Tier assignment logic
    if prov == "independent_ref" and conf >= 0.72 and motion_class in ("planar", "unknown"):
        item["measurement_tier"] = "A"
        item["_measurement_range_pct"] = _TIER_A_TOLERANCE
    elif prov in ("independent_ref", "vlm_person_prior") and conf >= 0.55:
        item["measurement_tier"] = "B"
        item["_measurement_range_pct"] = _TIER_B_TOLERANCE
    elif prov == "population_prior" and conf >= 0.50:
        item["measurement_tier"] = "C"
        item["_measurement_range_pct"] = _TIER_C_TOLERANCE
    else:
        item["measurement_tier"] = "C"
        item["_measurement_range_pct"] = _TIER_C_TOLERANCE

    # Depth-dominant downgrades by one tier
    if motion_class == "depth_dominant":
        tier_map = {"A": "B", "B": "C", "C": "D"}
        item["measurement_tier"] = tier_map.get(item["measurement_tier"], "D")

    return item


# ===========================================================================
# Geometry helpers
# ===========================================================================

def _dist2d(a: Tuple, b: Tuple) -> float:
    return math.sqrt((a[0] - b[0]) ** 2 + (a[1] - b[1]) ** 2)


def _angle_at_vertex(a: Tuple, b: Tuple, c: Tuple) -> float:
    v1 = (a[0] - b[0], a[1] - b[1])
    v2 = (c[0] - b[0], c[1] - b[1])
    dot = v1[0] * v2[0] + v1[1] * v2[1]
    mag = (_dist2d((0, 0), v1) + 1e-9) * (_dist2d((0, 0), v2) + 1e-9)
    return math.degrees(math.acos(max(-1.0, min(1.0, dot / mag))))


def _vector_angle_from_horizontal(p_from: Tuple, p_to: Tuple) -> float:
    dx = p_to[0] - p_from[0]
    dy = p_to[1] - p_from[1]
    return math.degrees(math.atan2(-dy, dx))


def _parabola(x, a, b, c):
    return a * x ** 2 + b * x + c


def _fit_parabola(xs: np.ndarray, ys: np.ndarray) -> Optional[Tuple]:
    if len(xs) < 4:
        return None
    try:
        if _SCIPY_OK:
            popt, _ = _scipy_curve_fit(_parabola, xs, ys, maxfev=2000)
        else:
            popt = np.polyfit(xs, ys, 2)
        y_pred = np.polyval(popt, xs)
        ss_res = np.sum((ys - y_pred) ** 2)
        ss_tot = np.sum((ys - np.mean(ys)) ** 2) + 1e-9
        r2 = 1.0 - ss_res / ss_tot
        return (float(popt[0]), float(popt[1]), float(popt[2]), float(r2))
    except Exception:
        return None


def _smooth_trajectory(arr: np.ndarray, window: int = 7, polyorder: int = 2) -> np.ndarray:
    """Savitzky-Golay smoothing; returns original array if too short or scipy unavailable."""
    if not _SCIPY_OK or len(arr) < 4:
        return arr
    from scipy.signal import savgol_filter
    w = window if window % 2 == 1 else window + 1
    w = min(w, len(arr) if len(arr) % 2 == 1 else len(arr) - 1)
    if w < 3:
        return arr
    return savgol_filter(arr, window_length=w, polyorder=min(polyorder, w - 1))


# ===========================================================================
# Calibration layer
# ===========================================================================

def calibrate_scale(
    clip_path: str,
    sport_category: str,
    vlm_subjects: List[Dict],
    pose_height_px: Optional[float] = None,
    frame_h: int = 0,
    yolo_bbox_heights: Optional[Dict[str, float]] = None,
    ref_detections: Optional[List[Dict]] = None,
) -> Dict[str, Any]:
    """
    Derive pixel/metre scale for the clip.

    Priority:
      0. Detected reference object with known size (SPORT_CALIBRATION / GENERAL_REF_OBJECTS / VLM)
      1. YOLO-detected person bbox height + known athlete height
      2. Pose-derived athlete height (if VLM gave a height estimate)
      3. Pose-derived athlete height + population prior
      4. Frame-height rough estimate (last resort)

    yolo_bbox_heights: {"person": median_px, ...} from _detect_ref_objects()
    ref_detections:    list of dicts from _detect_ref_objects() with known_h_m filled

    Returns:
        {scale_px_per_m, calibration_source, calibration_confidence,
         reference_object, reference_height_m}
    """
    if yolo_bbox_heights is None:
        yolo_bbox_heights = {}

    sport_key = sport_category.lower() if sport_category else "default"

    # --- Priority 0: Detected reference object with known real-world size ---
    if ref_detections:
        # Person detections are normally deferred to Priority 1 (VLM / population prior).
        # Exception: when known_h_m comes from GT sensor data (A-class), treat person as
        # the highest-confidence reference object — it IS the anchor.
        def _is_gt_person(d: Dict) -> bool:
            return ("person" in d.get("label", "").lower()
                    and d.get("source") in ("sensor_gt", "gt_3d_pose"))

        eligible = [d for d in ref_detections
                    if "person" not in d.get("label", "").lower() or _is_gt_person(d)]
        # GT-anchored person detections get top priority within the eligible set
        eligible.sort(key=lambda x: (1 if _is_gt_person(x) else 0, x.get("confidence", 0)),
                      reverse=True)
        for det in eligible:
            is_gt = _is_gt_person(det)
            if det.get("known_h_m") and det.get("height_px", 0) > 10:
                scale = det["height_px"] / det["known_h_m"]
                conf  = 0.95 if is_gt else min(0.88, 0.68 + det.get("confidence", 0) * 0.20)
                return {
                    "scale_px_per_m": round(scale, 1),
                    "calibration_source": f"ref_object_{det['label']}" + ("_gt" if is_gt else ""),
                    "calibration_confidence": round(conf, 3),
                    "reference_object": det["label"],
                    "reference_height_m": det["known_h_m"],
                    "scale_provenance": "independent_ref",
                    "source_type": "sensor_gt" if is_gt else "known_object",
                    "reliability_tier": "A",
                    "is_anchor_based": True,
                }
            if det.get("known_w_m") and det.get("width_px", 0) > 10:
                scale = det["width_px"] / det["known_w_m"]
                conf  = 0.95 if is_gt else min(0.85, 0.65 + det.get("confidence", 0) * 0.20)
                return {
                    "scale_px_per_m": round(scale, 1),
                    "calibration_source": f"ref_object_{det['label']}_width" + ("_gt" if is_gt else ""),
                    "calibration_confidence": round(conf, 3),
                    "reference_object": det["label"],
                    "reference_height_m": det["known_w_m"],
                    "scale_provenance": "independent_ref",
                    "source_type": "sensor_gt" if is_gt else "known_object",
                    "reliability_tier": "A",
                    "is_anchor_based": True,
                }

    # --- Priority 1: YOLO person bbox height + VLM or population athlete height ---
    yolo_person_px = yolo_bbox_heights.get("person")
    if not yolo_person_px and ref_detections:
        for det in ref_detections:
            if "person" in det.get("label", "").lower() and det.get("height_px", 0) > 10:
                yolo_person_px = det["height_px"]
                break
    if yolo_person_px and yolo_person_px > 10:
        vlm_height_m = None
        for s in vlm_subjects:
            h = s.get("estimated_height_m")
            if h and isinstance(h, (int, float)) and 1.0 < h < 2.5:
                vlm_height_m = float(h)
                break
        ref_h = vlm_height_m if vlm_height_m else _HEIGHT_PRIOR_M
        scale = yolo_person_px / ref_h
        conf = 0.75 if vlm_height_m else 0.55
        return {
            "scale_px_per_m": round(scale, 1),
            "calibration_source": "yolo_person_bbox" + ("_vlm_height" if vlm_height_m else "_population_prior"),
            "calibration_confidence": conf,
            "reference_object": "athlete",
            "reference_height_m": ref_h,
            "scale_provenance": "vlm_person_prior" if vlm_height_m else "population_prior",
            "source_type": "human_prior",
            "reliability_tier": "B",
            "is_anchor_based": bool(vlm_height_m),
        }

    # --- Priority 2: Pose-derived athlete height + VLM prior ---
    vlm_height_m = None
    for s in vlm_subjects:
        h = s.get("estimated_height_m")
        if h and isinstance(h, (int, float)) and 1.0 < h < 2.5:
            vlm_height_m = float(h)
            break

    if pose_height_px and pose_height_px > 0 and vlm_height_m:
        scale = pose_height_px / vlm_height_m
        return {
            "scale_px_per_m": round(scale, 1),
            "calibration_source": "pose_height_with_vlm_prior",
            "calibration_confidence": 0.60,
            "reference_object": "athlete",
            "reference_height_m": vlm_height_m,
            "scale_provenance": "vlm_person_prior",
            "source_type": "human_prior",
            "reliability_tier": "B",
            "is_anchor_based": True,
        }

    # --- Priority 3: Pose-derived height + population prior ---
    if pose_height_px and pose_height_px > 0:
        scale = pose_height_px / _HEIGHT_PRIOR_M
        return {
            "scale_px_per_m": round(scale, 1),
            "calibration_source": "pose_height_population_prior",
            "calibration_confidence": 0.40,
            "reference_object": "athlete",
            "reference_height_m": _HEIGHT_PRIOR_M,
            "scale_provenance": "population_prior",
            "source_type": "human_prior",
            "reliability_tier": "C",
        }

    # --- Fallback: frame-height-based rough estimate ---
    if frame_h > 0:
        scale = (frame_h * 0.45) / _HEIGHT_PRIOR_M
        return {
            "scale_px_per_m": round(scale, 1),
            "calibration_source": "frame_height_prior",
            "calibration_confidence": 0.25,
            "reference_object": None,
            "reference_height_m": _HEIGHT_PRIOR_M,
            "scale_provenance": "frame_heuristic",
            "source_type": "frame_height",
            "reliability_tier": "D",
        }

    return {
        "scale_px_per_m": None,
        "calibration_source": "none",
        "calibration_confidence": 0.0,
        "reference_object": None,
        "reference_height_m": None,
        "scale_provenance": "none",
        "source_type": "none",
        "reliability_tier": "D",
    }


# ===========================================================================
# Route A — MediaPipe Pose (Tasks API)
# ===========================================================================

def _get_lm(lm_list, key: str) -> Optional[Tuple[float, float, float]]:
    idx = _LM.get(key)
    if idx is None or idx >= len(lm_list):
        return None
    l = lm_list[idx]
    if l is None:
        return None
    return l.x, l.y, l.visibility


def _body_height_norm(lm_list) -> Optional[float]:
    MIN_VIS = 0.5
    for side in [
        ("left_shoulder", "left_hip", "left_knee", "left_ankle"),
        ("right_shoulder", "right_hip", "right_knee", "right_ankle"),
    ]:
        pts = [_get_lm(lm_list, k) for k in side]
        if any(p is None for p in pts):
            continue
        if all(p[2] > MIN_VIS for p in pts):
            torso = abs(pts[1][1] - pts[0][1])
            thigh = abs(pts[2][1] - pts[1][1])
            shin  = abs(pts[3][1] - pts[2][1])
            lower = thigh + shin
            return torso + lower + (torso + lower) * 0.18
    return None


def _per_frame_stability(lm_list) -> float:
    """Mean visibility of shoulder/hip/knee/ankle keypoints for this frame (0-1)."""
    keys = ("left_shoulder", "right_shoulder", "left_hip", "right_hip",
            "left_knee", "right_knee", "left_ankle", "right_ankle")
    vis = [_get_lm(lm_list, k) for k in keys]
    valid = [v[2] for v in vis if v is not None]
    return float(np.mean(valid)) if valid else 0.0


def _run_pose(video_path: str) -> Tuple[List, float, Tuple[int, int]]:
    if not os.path.exists(_MODEL_PATH):
        raise RuntimeError(f"Pose model not found: {_MODEL_PATH}")

    base_options = _mp_tasks.BaseOptions(model_asset_path=_MODEL_PATH)
    options = _mp_vision.PoseLandmarkerOptions(
        base_options=base_options,
        running_mode=_mp_vision.RunningMode.VIDEO,
        num_poses=1,
        min_pose_detection_confidence=0.5,
        min_pose_presence_confidence=0.5,
        min_tracking_confidence=0.5,
    )

    cap = cv2.VideoCapture(video_path)
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    w   = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h   = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    lm_per_frame = []
    frame_idx    = 0

    with _mp_vision.PoseLandmarker.create_from_options(options) as landmarker:
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
            timestamp_ms = int(frame_idx * 1000 / fps)
            result = landmarker.detect_for_video(mp_image, timestamp_ms)
            if result.pose_landmarks:
                lm_per_frame.append(result.pose_landmarks[0])
            else:
                lm_per_frame.append(None)
            frame_idx += 1

    cap.release()
    return lm_per_frame, fps, (w, h)


def _get_yolo_pose():
    global _yolo_pose_model, _YOLO_POSE_OK
    if _yolo_pose_model is None:
        _yolo_pose_model = _YOLO("yolo11n-pose.pt")
        _YOLO_POSE_OK = True
    return _yolo_pose_model


def _run_yolo_pose(video_path: str) -> Tuple[List, float, Tuple[int, int]]:
    """Run YOLOv11-pose frame-by-frame; return MediaPipe-compatible landmark lists."""
    model = _get_yolo_pose()
    cap = cv2.VideoCapture(video_path)
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    fw  = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    fh  = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    upscale = fw > 0 and fw < 640

    lm_per_frame: List = []
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        if upscale:
            frame = cv2.resize(frame, (fw * 2, fh * 2), interpolation=cv2.INTER_CUBIC)

        result = model.predict(frame, verbose=False, conf=0.3)[0]

        if result.keypoints is None or len(result.keypoints.xy) == 0:
            lm_per_frame.append(None)
            continue

        kps = result.keypoints
        # Pick the highest-confidence person
        if kps.conf is not None and len(kps.conf) > 0:
            import torch
            best_idx = int(kps.conf.mean(dim=1).argmax())
        else:
            best_idx = 0

        kp_xy   = kps.xy[best_idx].cpu().numpy()    # (17, 2) pixel coords
        kp_conf = (kps.conf[best_idx].cpu().numpy()
                   if kps.conf is not None else np.ones(17))

        scale_w = fw * (2 if upscale else 1)
        scale_h = fh * (2 if upscale else 1)

        lm_list: List = [None] * 33
        for coco_idx, mp_idx in _COCO_TO_MP.items():
            x_px, y_px = kp_xy[coco_idx]
            lm_list[mp_idx] = _KP(
                x_px / (scale_w + 1e-9),
                y_px / (scale_h + 1e-9),
                float(kp_conf[coco_idx]),
            )
        lm_per_frame.append(lm_list)

    cap.release()
    return lm_per_frame, fps, (fw, fh)


def _estimate_camera_pan(tracks: Dict, fps: float, scale: float, clip_path: str = "") -> float:
    """Estimate camera horizontal pan speed (m/s).

    Primary: sparse Lucas-Kanade optical flow on sampled frame pairs (works even
    when background YOLO tracks are empty — the common case in tracking shots).
    Fallback: YOLO background-track drift if optical flow is unavailable.
    """
    # --- Primary: optical flow on a few sampled frame pairs ---
    if clip_path:
        try:
            cap = cv2.VideoCapture(clip_path)
            total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            fw    = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            fh    = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            cap.release()

            # Sample ~8 evenly-spaced frame pairs; avoid first/last 10%
            sample_start = max(0, int(total * 0.10))
            sample_end   = min(total - 1, int(total * 0.90))
            n_pairs = 8
            step    = max(1, (sample_end - sample_start) // n_pairs)

            cap = cv2.VideoCapture(clip_path)
            lk_params = dict(winSize=(15, 15), maxLevel=2,
                             criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 10, 0.03))
            feat_params = dict(maxCorners=80, qualityLevel=0.3, minDistance=10, blockSize=7)

            pan_samples: list = []
            for pi in range(n_pairs):
                fi = sample_start + pi * step
                cap.set(cv2.CAP_PROP_POS_FRAMES, fi)
                ok1, f1 = cap.read()
                cap.set(cv2.CAP_PROP_POS_FRAMES, fi + 1)
                ok2, f2 = cap.read()
                if not ok1 or not ok2:
                    continue
                g1 = cv2.cvtColor(f1, cv2.COLOR_BGR2GRAY)
                g2 = cv2.cvtColor(f2, cv2.COLOR_BGR2GRAY)
                pts = cv2.goodFeaturesToTrack(g1, **feat_params)
                if pts is None or len(pts) < 10:
                    continue
                pts2, status, _ = cv2.calcOpticalFlowPyrLK(g1, g2, pts, None, **lk_params)
                good = status.ravel() == 1
                if good.sum() < 6:
                    continue
                p1 = pts[good].reshape(-1, 2)
                p2 = pts2[good].reshape(-1, 2)
                dx = p2[:, 0] - p1[:, 0]  # horizontal displacement (pixels/frame)
                # Robust median — ignore outliers from moving athlete
                med_dx = float(np.median(dx))
                pan_samples.append(med_dx)
            cap.release()

            if pan_samples and scale > 0:
                # Median across sampled pairs; convert px/frame → m/s
                pan_px_per_frame = float(np.median(pan_samples))
                pan_ms = abs(pan_px_per_frame) * fps / scale
                logger.debug(f"Camera pan (optical flow): {pan_px_per_frame:.2f} px/frame → {pan_ms:.2f} m/s")
                return pan_ms
        except Exception as exc:
            logger.debug(f"Optical flow pan estimation failed: {exc}")

    # --- Fallback: YOLO background-track drift ---
    if not tracks or not scale or scale <= 0:
        return 0.0
    bg_speeds = []
    for tid, dets in tracks.items():
        if len(dets) < 4:
            continue
        sorted_dets = sorted(dets, key=lambda d: d[0])
        xs = [d[1] for d in sorted_dets]
        dx_px = abs(xs[-1] - xs[0])
        dt    = (sorted_dets[-1][0] - sorted_dets[0][0]) / fps
        if dt > 0:
            bg_speeds.append(dx_px / dt / scale)
    return float(np.median(bg_speeds)) if bg_speeds else 0.0


def route_a_pose_estimation(
    clip_path: str,
    calibration: Dict[str, Any],
    background_tracks: Optional[Dict] = None,
    gt_segments: Optional[Dict[str, float]] = None,
    gt_upright_frames: Optional[set] = None,
) -> Dict[str, Any]:
    # Try YOLOv11-pose first; fall back to MediaPipe
    lm_frames = None
    fps = 30.0
    fw = fh = 0

    if _YOLO_OK:
        try:
            _get_yolo_pose()
            if _YOLO_POSE_OK:
                lm_frames, fps, (fw, fh) = _run_yolo_pose(clip_path)
        except Exception as exc:
            logger.warning(f"Route A: YOLOv11-pose failed ({exc}), trying MediaPipe")
            lm_frames = None

    if lm_frames is None and _MEDIAPIPE_OK:
        try:
            lm_frames, fps, (fw, fh) = _run_pose(clip_path)
        except Exception as exc:
            logger.warning(f"Route A: MediaPipe pose run failed: {exc}")
            return {}

    if lm_frames is None:
        return {}

    valid = [(i, lm) for i, lm in enumerate(lm_frames) if lm is not None]
    if len(valid) < 5:
        return {}

    # Compute GT-segment-based calibration while lm_frames are in memory.
    # Returned in the result so augment_opq() can use it before the second Route A pass.
    gt_segment_calibration: Optional[Dict] = None
    if gt_segments:
        gt_segment_calibration = _gt_segment_scale_from_lm(lm_frames, fw, fh, gt_segments)

    scale_px_per_m = calibration.get("scale_px_per_m")
    cal_conf       = calibration.get("calibration_confidence", 0.0)

    # Height estimation: if GT upright frames are provided (A-class), restrict to those.
    # This ensures height is measured only from standing-upright GT-annotated frames.
    # gt_upright_frames=None means "no restriction" (B1/B2); empty set means "no valid frames".
    if gt_upright_frames is not None:
        height_valid = [(i, lm) for i, lm in valid if i in gt_upright_frames]
    else:
        height_valid = valid  # B1/B2: use all frames (stability filter applied below)

    if height_valid:
        frame_stabilities = [(_per_frame_stability(lm), _body_height_norm(lm))
                             for _, lm in height_valid]
    else:
        frame_stabilities = []
    frame_stabilities = [(s, h) for s, h in frame_stabilities if h is not None]

    subjects = []
    pose_height_px = None

    if frame_stabilities:
        # Sort by per-frame stability descending, take top-K
        frame_stabilities.sort(key=lambda x: x[0], reverse=True)
        K = max(5, min(15, len(frame_stabilities) // 3))
        top_k = frame_stabilities[:K]
        top_heights = [h for _, h in top_k]
        median_h  = float(np.median(top_heights))
        stability = float(max(0.0, 1.0 - np.std(top_heights) / (median_h + 1e-9) * 5))
        pose_height_px = median_h * fh  # pixels

        if scale_px_per_m:
            athlete_h_m = round(pose_height_px / scale_px_per_m, 2)
            height_conf = round(min(0.82, 0.50 + stability * 0.25 + cal_conf * 0.10), 2)
            basis = (
                f"Pose chain (shoulder-hip-knee-ankle), top_{K}_stable_frames of {len(frame_stabilities)}, "
                f"stability={stability:.2f}. Scale from {calibration.get('calibration_source')}."
            )
            subjects.append({
                "label": "athlete",
                "estimated_height_m": athlete_h_m,
                "estimated_mass_kg":  None,
                "estimated_length_m": None,
                "other_attributes": {"pose_stability": round(stability, 2)},
                "estimation_basis": basis,
                "estimation_confidence": height_conf,
                "estimation_method": "pose_estimation",
            })

    motions = []

    # Velocity from hip midpoint
    if scale_px_per_m:
        hip_traj = []
        for i, lm in valid:
            l = _get_lm(lm, "left_hip")
            r = _get_lm(lm, "right_hip")
            if l and r and l[2] > 0.5 and r[2] > 0.5:
                hip_traj.append((i, (l[0] + r[0]) / 2 * fw, (l[1] + r[1]) / 2 * fh))

        if len(hip_traj) >= 5:
            # Smooth hip coordinates before computing velocity
            hip_xs = _smooth_trajectory(np.array([h[1] for h in hip_traj]))
            hip_ys = _smooth_trajectory(np.array([h[2] for h in hip_traj]))
            hip_is = [h[0] for h in hip_traj]
            speeds = []
            for j in range(1, len(hip_traj)):
                i1, x1, y1 = hip_is[j - 1], hip_xs[j - 1], hip_ys[j - 1]
                i2, x2, y2 = hip_is[j],     hip_xs[j],     hip_ys[j]
                dt = (i2 - i1) / fps
                if dt > 0:
                    dx_m = (x2 - x1) / scale_px_per_m
                    dy_m = (y2 - y1) / scale_px_per_m
                    speeds.append(math.sqrt(dx_m**2 + dy_m**2) / dt)
            if speeds:
                med_speed = float(np.median(speeds))
                cam_pan = _estimate_camera_pan(background_tracks or {}, fps, scale_px_per_m, clip_path)
                corrected_speed = round(med_speed + cam_pan, 1)
                speed_conf = round(min(0.75, 0.50 + cal_conf * 0.25), 2)
                motions.append({
                    "subject": "athlete",
                    "event": "movement",
                    "time_point_s": round(hip_traj[len(hip_traj) // 2][0] / fps, 2),
                    "estimated_launch_angle_deg": None,
                    "estimated_speed_ms": corrected_speed,
                    "estimated_height_m": None,
                    "estimation_basis": (
                        f"Hip midpoint tracking, {len(speeds)} frame-pairs, scale={scale_px_per_m:.1f} px/m"
                        + (f", cam_pan_correction=+{cam_pan:.2f} m/s" if cam_pan > 0 else "") + "."
                    ),
                    "estimation_confidence": speed_conf,
                    "estimation_method": "pose_estimation",
                })
                # Peak velocity event detection (event frame = speed spike)
                if _SCIPY_OK and len(speeds) >= 6:
                    try:
                        from scipy.signal import find_peaks as _find_peaks
                        speed_arr = np.array(speeds)
                        peaks, props = _find_peaks(speed_arr, prominence=0.5)
                        if len(peaks):
                            peak_idx = peaks[np.argmax(speed_arr[peaks])]
                            peak_speed = round(float(speed_arr[peak_idx]) + cam_pan, 1)
                            peak_frame = hip_is[peak_idx + 1]  # speeds[j] corresponds to hip_is[j+1]
                            if peak_speed > corrected_speed * 1.2:  # only add if notably higher than median
                                motions.append({
                                    "subject": "athlete",
                                    "event": "peak_motion",
                                    "time_point_s": round(peak_frame / fps, 2),
                                    "estimated_launch_angle_deg": None,
                                    "estimated_speed_ms": peak_speed,
                                    "estimated_height_m": None,
                                    "estimation_basis": (
                                        f"Peak hip velocity at frame {peak_frame} "
                                        f"(prominence-based detection), scale={scale_px_per_m:.1f} px/m."
                                    ),
                                    "estimation_confidence": round(speed_conf * 0.9, 2),
                                    "estimation_method": "pose_estimation",
                                })
                    except Exception:
                        pass

    # Arm release angle
    arm_events = []
    for i, lm in valid:
        for side in [("left_shoulder", "left_elbow", "left_wrist"),
                     ("right_shoulder", "right_elbow", "right_wrist")]:
            pts = [_get_lm(lm, k) for k in side]
            if any(p is None for p in pts):
                continue
            if all(p[2] > 0.5 for p in pts):
                shoulder, _, wrist = pts
                ang = _vector_angle_from_horizontal(
                    (shoulder[0] * fw, shoulder[1] * fh),
                    (wrist[0] * fw, wrist[1] * fh),
                )
                arm_events.append((i, ang))

    if arm_events:
        best = max(arm_events, key=lambda x: x[1])
        if best[1] > 15:
            motions.append({
                "subject": "athlete",
                "event": "arm_release_or_raise",
                "time_point_s": round(best[0] / fps, 2),
                "estimated_launch_angle_deg": round(best[1], 1),
                "estimated_speed_ms": None,
                "estimated_height_m": None,
                "estimation_basis": f"Shoulder-to-wrist vector at peak elevation (frame {best[0]}).",
                "estimation_confidence": 0.72,
                "estimation_method": "pose_estimation",
            })

    # Knee flexion (takeoff/crouch)
    knee_series = []
    for i, lm in valid:
        for side in [("left_hip", "left_knee", "left_ankle"),
                     ("right_hip", "right_knee", "right_ankle")]:
            pts = [_get_lm(lm, k) for k in side]
            if any(p is None for p in pts):
                continue
            if all(p[2] > 0.5 for p in pts):
                ang = _angle_at_vertex(
                    (pts[0][0] * fw, pts[0][1] * fh),
                    (pts[1][0] * fw, pts[1][1] * fh),
                    (pts[2][0] * fw, pts[2][1] * fh),
                )
                knee_series.append((i, ang))

    if knee_series:
        min_k = min(knee_series, key=lambda x: x[1])
        if min_k[1] < 130:
            motions.append({
                "subject": "athlete",
                "event": "takeoff_crouch",
                "time_point_s": round(min_k[0] / fps, 2),
                "estimated_launch_angle_deg": None,
                "estimated_speed_ms": None,
                "estimated_height_m": None,
                "estimation_basis": f"Min knee angle = {min_k[1]:.1f}° at frame {min_k[0]}.",
                "estimation_confidence": 0.75,
                "estimation_method": "pose_estimation",
                "other_attributes": {"knee_angle_deg": round(min_k[1], 1)},
            })

    logger.info(f"Route A: {len(subjects)} subject(s), {len(motions)} motion(s)")
    result = {"subjects": subjects, "motions": motions, "pose_height_px": pose_height_px}
    if gt_segment_calibration is not None:
        result["gt_segment_calibration"] = gt_segment_calibration
    return result


# ===========================================================================
# Route B — YOLO detection + ByteTrack trajectory
# ===========================================================================

# Singleton YOLO model (loaded once)
_yolo_model = None

def _get_yolo():
    global _yolo_model
    if _yolo_model is None:
        _yolo_model = _YOLO("yolo11n.pt")  # auto-downloads on first use
    return _yolo_model


_yolo_world_model = None

def _get_yolo_world():
    global _yolo_world_model
    if _yolo_world_model is None:
        try:
            import torch
            from ultralytics.models import YOLOWorld as _YW
            device = "cuda:0" if torch.cuda.is_available() else "cpu"
            _yolo_world_model = _YW("yolov8s-worldv2.pt")
            _yolo_world_model.to(device)
        except Exception as exc:
            logger.warning(f"YOLOWorld load failed: {exc}")
            _yolo_world_model = None
    return _yolo_world_model


# ---------------------------------------------------------------------------
# VLM config (for _vlm_ref_object_size fallback)
# ---------------------------------------------------------------------------
_OPENROUTER_BASE_URL = os.environ.get("CHAT_API_BASE", "https://openrouter.ai/api/v1")
_OPENROUTER_API_KEY  = os.environ.get("VLM_API_KEY") or os.environ.get("OPENROUTER_API_KEY", "")
_VLM_MODEL           = os.environ.get("GEMINI_MODEL", "google/gemini-3.6-flash")


def _parse_json_simple(text: str) -> Optional[Dict]:
    """Extract first JSON object from text (handles fenced code blocks)."""
    import re
    text = re.sub(r"<think>.*?</think>", "", text, flags=re.DOTALL)
    m = re.search(r"```(?:json)?\s*(\{.*?\})\s*```", text, re.DOTALL)
    if m:
        text = m.group(1)
    m = re.search(r"\{.*\}", text, re.DOTALL)
    if m:
        try:
            return json.loads(m.group())
        except Exception:
            pass
    return None


def _vlm_ref_object_size(unknown_labels: List[str]) -> Optional[Dict]:
    """Ask VLM for world-knowledge size of unknown reference objects (text-only call)."""
    try:
        from openai import OpenAI  # sync client avoids event-loop conflict

        client = OpenAI(base_url=_OPENROUTER_BASE_URL, api_key=_OPENROUTER_API_KEY)
        labels_str = ", ".join(unknown_labels[:6])
        prompt = (
            f"These objects were detected in a video: {labels_str}. "
            "Pick the single most useful one for metric scale calibration "
            "(prefer objects with well-known standard dimensions). "
            "Give its standard real-world height and width in metres. "
            "Use world knowledge only — do NOT estimate from the image. "
            'Respond with JSON only: {"label": "...", "estimated_height_m": <float or null>, '
            '"estimated_width_m": <float or null>, "confidence": <0-1>}'
        )
        resp = client.chat.completions.create(
            model=_VLM_MODEL,
            messages=[{"role": "user", "content": prompt}],
            max_tokens=256,
            timeout=30,
            extra_headers={"HTTP-Referer": "https://github.com/OmniGAIA"},
        )
        return _parse_json_simple(resp.choices[0].message.content or "")
    except Exception as exc:
        logger.warning(f"VLM ref object size query failed: {exc}")
        return None


# ---------------------------------------------------------------------------
# Anchor label aliases (YOLO COCO names vs. natural-language anchor labels)
# ---------------------------------------------------------------------------

_ANCHOR_ALIASES: Dict[str, List[str]] = {
    "basketball":   ["sports ball", "ball"],
    "football":     ["sports ball", "ball"],
    "soccer ball":  ["sports ball", "ball"],
    "tennis ball":  ["sports ball", "ball"],
    "volleyball":   ["sports ball", "ball"],
    "baseball":     ["sports ball", "ball"],
}


def _label_matches(spec_label: str, det_label: str) -> bool:
    """Return True if spec_label and det_label refer to the same object,
    considering YOLO COCO class aliases (e.g. 'basketball' ↔ 'sports ball')."""
    s, d = spec_label.lower().strip(), det_label.lower().strip()
    if s in d or d in s:
        return True
    for alias in _ANCHOR_ALIASES.get(s, []):
        if alias in d or d in alias:
            return True
    return False


def _vlm_locate_bbox(
    clip_path: str,
    label: str,
    n_frames: int = 3,
    frame_indices: Optional[set] = None,
) -> Optional[Dict]:
    """Ask VLM to locate *label* in up to n_frames sampled from clip.

    Returns {"height_px": float, "width_px": float, "confidence": float} using
    the first frame where the object is found, or None if not found in any frame.
    Uses a sync OpenAI client (same pattern as _vlm_ref_object_size) to avoid
    nested event-loop issues.
    """
    import base64 as _b64
    try:
        cap = cv2.VideoCapture(clip_path)
        total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT)) or 1
        fh = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        fw = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))

        # Prefer GT/quality frame indices; fall back to uniform sampling
        if frame_indices:
            indices = sorted(f for f in frame_indices if 0 <= f < total)
            if not indices:
                indices = [int(total * i / max(1, n_frames - 1)) for i in range(n_frames)]
        else:
            indices = [int(total * i / max(1, n_frames - 1)) for i in range(n_frames)]
        indices = sorted(set(min(max(0, idx), total - 1) for idx in indices))

        frames_b64 = []
        for idx in indices:
            cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
            ret, frame = cap.read()
            if not ret:
                continue
            # Downsample large frames to max 1280px wide to reduce image tokens
            # and avoid truncated responses from VLM token limits.
            h_orig, w_orig = frame.shape[:2]
            if w_orig > 1280:
                scale = 1280.0 / w_orig
                frame = cv2.resize(frame, (1280, int(h_orig * scale)), interpolation=cv2.INTER_AREA)
            _, buf = cv2.imencode(".jpg", frame, [cv2.IMWRITE_JPEG_QUALITY, 80])
            frames_b64.append(_b64.b64encode(buf).decode())
        cap.release()

        if not frames_b64:
            return None

        from openai import OpenAI as _OAI
        client = _OAI(base_url=_OPENROUTER_BASE_URL, api_key=_OPENROUTER_API_KEY)

        for b64 in frames_b64:
            content = [
                {
                    "type": "image_url",
                    "image_url": {"url": f"data:image/jpeg;base64,{b64}"},
                },
                {
                    "type": "text",
                    "text": (
                        f"Locate the '{label}' in this image. "
                        "Output ONLY a single JSON object. No explanation. No reasoning. Just JSON: "
                        '{"found": true/false, "bbox_xywh_norm": [cx, cy, w, h], "confidence": 0.0-1.0} '
                        "where cx,cy are the center and w,h are size, all as 0-1 fractions of image dimensions."
                    ),
                },
            ]
            resp = client.chat.completions.create(
                model=_VLM_MODEL,
                messages=[{"role": "user", "content": content}],
                max_tokens=2048,
                timeout=60,
                extra_headers={"HTTP-Referer": "https://github.com/OmniGAIA"},
            )
            raw_content = resp.choices[0].message.content or ""
            logger.debug(f"_vlm_locate_bbox '{label}' frame={idx}: raw={raw_content[:200]!r}")
            parsed = _parse_json_simple(raw_content)
            if not parsed or not parsed.get("found"):
                logger.info(f"_vlm_locate_bbox '{label}' frame={idx}: found=False or parse_fail. parsed={parsed}")
                continue
            bbox = parsed.get("bbox_xywh_norm")
            conf = float(parsed.get("confidence", 0.5))
            if not bbox or len(bbox) < 4:
                continue
            _, _, w_norm, h_norm = bbox
            if h_norm <= 0 or w_norm <= 0:
                continue
            return {
                "height_px": round(h_norm * fh, 1),
                "width_px":  round(w_norm * fw, 1),
                "confidence": round(conf, 3),
            }

    except Exception as exc:
        logger.warning(f"_vlm_locate_bbox '{label}' failed: {exc}")
    return None


def _select_flow_event_frames(clip_path: str, n_event: int = 3, n_uniform: int = 2) -> List[int]:
    """Select frames near optical flow spikes (collisions, bounces) + uniform context frames.

    Computes Farneback optical flow on a downsampled version of the clip, picks the
    n_event frames with the highest mean flow magnitude, and adds n_uniform evenly-spaced
    frames for context. Deduplicates and returns a sorted list of frame indices.

    Falls back to uniform sampling if OpenCV flow fails.
    """
    try:
        cap = cv2.VideoCapture(clip_path)
        total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT)) or 1
        n_total = n_event + n_uniform

        if total <= n_total:
            cap.release()
            return list(range(total))

        # Sample every step-th frame — ~40 probes is cheap
        step = max(1, total // 40)
        flow_samples: List[tuple] = []  # (magnitude, frame_idx)
        prev_gray = None

        for i in range(0, total, step):
            cap.set(cv2.CAP_PROP_POS_FRAMES, i)
            ret, frame = cap.read()
            if not ret:
                continue
            # Downscale to 160×90 for fast flow computation
            small = cv2.resize(frame, (160, 90), interpolation=cv2.INTER_AREA)
            gray = cv2.cvtColor(small, cv2.COLOR_BGR2GRAY)
            if prev_gray is not None:
                flow = cv2.calcOpticalFlowFarneback(
                    prev_gray, gray, None, 0.5, 3, 9, 3, 5, 1.1, 0
                )
                mag = float(np.mean(np.sqrt(flow[..., 0] ** 2 + flow[..., 1] ** 2)))
                flow_samples.append((mag, i))
            prev_gray = gray

        cap.release()

        if not flow_samples:
            raise ValueError("no flow samples")

        # Top n_event frames by flow magnitude
        flow_samples.sort(key=lambda x: x[0], reverse=True)
        event_frames = sorted(s[1] for s in flow_samples[:n_event])

        # Evenly spaced uniform frames (avoid 0 and total-1 — usually uninformative)
        uniform_frames = [int(total * i / (n_uniform + 1)) for i in range(1, n_uniform + 1)]

        all_frames = sorted(set(event_frames + uniform_frames))
        return all_frames

    except Exception as exc:
        logger.warning(f"_select_flow_event_frames failed ({exc}), falling back to uniform")
        n_total = n_event + n_uniform
        try:
            cap2 = cv2.VideoCapture(clip_path)
            total2 = int(cap2.get(cv2.CAP_PROP_FRAME_COUNT)) or n_total
            cap2.release()
        except Exception:
            total2 = n_total
        return [int(total2 * i / max(1, n_total - 1)) for i in range(n_total)]


def _vlm_extract_b_physics(
    clip_path: str,
    clip_summary: str = "",
    scene_objects: Optional[List[str]] = None,
    sport_category: str = "",
    n_frames: int = 5,
) -> Dict[str, Any]:
    """Send event-aware frames from a B-class clip to Gemini and ask it to
    estimate physical quantities observable from the video.

    Frame selection uses optical flow to prioritise collision/bounce moments
    (high motion frames) over uniformly-spaced frames, ensuring the VLM sees
    the key physical events rather than static scenes.

    Only scale-free quantities are requested (angles, counts, relative ratios).
    Absolute metric values (heights, speeds) are explicitly excluded.

    Returns OPQ-compatible dict: {"subjects": [...], "motions": [...]} where
    every entry has estimation_method="vlm".
    """
    import base64 as _b64

    try:
        # Event-aware frame selection: 3 high-flow frames + 2 uniform context frames
        indices = _select_flow_event_frames(clip_path, n_event=3, n_uniform=2)

        cap = cv2.VideoCapture(clip_path)
        total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT)) or 1
        indices = sorted(set(min(max(0, idx), total - 1) for idx in indices))

        frames_b64 = []
        for idx in indices:
            cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
            ret, frame = cap.read()
            if not ret:
                continue
            h0, w0 = frame.shape[:2]
            if w0 > 1280:
                frame = cv2.resize(frame, (1280, int(h0 * 1280.0 / w0)),
                                   interpolation=cv2.INTER_AREA)
            _, buf = cv2.imencode(".jpg", frame, [cv2.IMWRITE_JPEG_QUALITY, 80])
            frames_b64.append(_b64.b64encode(buf).decode())
        cap.release()

        if not frames_b64:
            logger.warning(f"_vlm_extract_b_physics: no frames read from {clip_path}")
            return {"subjects": [], "motions": []}

        from openai import OpenAI as _OAI
        client = _OAI(base_url=_OPENROUTER_BASE_URL, api_key=_OPENROUTER_API_KEY)

        ctx_parts = []
        if sport_category:
            ctx_parts.append(f"Sport/activity: {sport_category}.")
        if clip_summary:
            ctx_parts.append(f"Clip description: {clip_summary[:300]}")
        if scene_objects:
            ctx_parts.append(f"Objects seen: {', '.join(scene_objects[:8])}.")
        context_str = " ".join(ctx_parts) if ctx_parts else ""

        prompt = (
            f"{context_str}\n\n"
            "You are a physics observer. Analyze the sequence of frames from this video clip.\n\n"
            "Identify ALL moving objects and physical events. For each, provide quantitative "
            "estimates where possible.\n\n"
            "COORDINATE SYSTEM CONVENTION:\n"
            "  - Angles are measured in the image plane (2D projection of the 3D scene)\n"
            "  - Horizontal reference: use a VISIBLE horizontal edge (table edge, floor line, court baseline)\n"
            "  - Vertical reference: use a VISIBLE vertical edge (wall, goal post, vertical line)\n"
            "  - Counter-clockwise from reference = positive angle; clockwise = negative angle\n"
            "  - Always specify which direction is 0° for your measurement\n\n"
            "ALLOWED quantities (scale-free only):\n"
            "  - Angles in degrees (0–360): deflection angle, launch angle, incline angle, spin\n"
            "  - Counts: number of bounces, collisions, rotations, steps\n"
            "  - Dimensionless ratios: speed ratio (after/before collision), size ratio\n"
            "  - Qualitative direction: 'left', 'upward', 'clockwise'\n\n"
            "FORBIDDEN quantities:\n"
            "  - Absolute heights, distances, or speeds (no meters, cm, m/s)\n"
            "  - Any value that requires real-world scale to compute\n\n"
            "ANGLE REFERENCE REQUIREMENTS (CRITICAL):\n"
            "  1. Reference MUST be a VISIBLE, STATIC, LINEAR feature in the video\n"
            "  2. Valid examples:\n"
            "     - 'horizontal edge of the snooker table' ✓\n"
            "     - 'vertical left goal post' ✓\n"
            "     - 'floor baseline parallel to camera' ✓\n"
            "  3. Invalid examples:\n"
            "     - 'screen horizontal' ✗ (not in video scene)\n"
            "     - 'camera axis' ✗ (not visible)\n"
            "     - 'initial trajectory' ✗ (not static)\n"
            "  4. For deflection angles: specify BOTH incoming and outgoing reference\n"
            "     Example: 'incoming trajectory at 25° above table edge, outgoing at 45° above same edge'\n"
            "  5. Always state which direction is 0° (e.g., '0° = horizontal table edge, positive = upward')\n\n"
            "MULTI-OBJECT TRACKING:\n"
            "  - If multiple similar objects appear (e.g., 2 basketballs, 3 players), assign unique IDs\n"
            "  - Use descriptive IDs: 'basketball_orange', 'basketball_left', 'player_red_shirt'\n"
            "  - Track the SAME object across frames using color, position, or distinguishing features\n"
            "  - In motions, reference the specific object: {\"subject\": \"basketball_orange\", ...}\n\n"
            "For estimation_confidence: 0.9=certain, 0.7=confident, 0.5=uncertain, 0.3=guess.\n\n"
            "Output ONLY valid JSON, no explanation:\n"
            "{\n"
            '  "subjects": [\n'
            '    {"label": "unique_object_id", "color": "color or unknown",\n'
            '     "relative_size_note": "e.g. larger than opponent or similar size",\n'
            '     "estimation_confidence": 0.8, "estimation_basis": "visible in frames"}\n'
            "  ],\n"
            '  "motions": [\n'
            '    {"subject": "unique_object_id (must match subjects[].label)",\n'
            '     "event": "deflection_angle|bounce_count|trajectory_direction|speed_ratio|rotation_count|other",\n'
            '     "estimated_angle_deg": null,\n'
            '     "angle_reference": "MUST be specific visible static feature + 0° direction",\n'
            '     "estimated_count": null,\n'
            '     "estimated_ratio": null,\n'
            '     "direction_desc": null,\n'
            '     "estimation_basis": "brief explanation referencing the named reference object",\n'
            '     "estimation_confidence": 0.7}\n'
            "  ]\n"
            "}"
        )

        content = []
        for b64 in frames_b64:
            content.append({"type": "image_url",
                             "image_url": {"url": f"data:image/jpeg;base64,{b64}"}})
        content.append({"type": "text", "text": prompt})

        resp = client.chat.completions.create(
            model=_VLM_MODEL,
            messages=[{"role": "user", "content": content}],
            max_tokens=2048,
            temperature=0.1,  # Low temperature for consistent physical measurements
            timeout=90,
            extra_headers={"HTTP-Referer": "https://github.com/OmniGAIA"},
        )
        raw = resp.choices[0].message.content or ""
        parsed = _parse_json_simple(raw)
        if not parsed:
            logger.warning(f"_vlm_extract_b_physics: JSON parse failed for {clip_path}. raw={raw[:200]!r}")
            return {"subjects": [], "motions": []}

        subjects = []
        for s in parsed.get("subjects", []):
            if not isinstance(s, dict):
                continue
            s["estimation_method"] = "vlm"
            subjects.append(s)

        motions = []
        for m in parsed.get("motions", []):
            if not isinstance(m, dict):
                continue
            m["estimation_method"] = "vlm"
            # Normalise: drop null-valued optional fields to keep output clean
            for field in ("estimated_angle_deg", "angle_reference", "estimated_count",
                          "estimated_ratio", "direction_desc"):
                if m.get(field) is None:
                    m.pop(field, None)
            motions.append(m)

        logger.info(
            f"_vlm_extract_b_physics: {len(subjects)} subjects, {len(motions)} motions"
            f" from {len(frames_b64)} frames of {os.path.basename(clip_path)}"
        )
        return {"subjects": subjects, "motions": motions}

    except Exception as exc:
        logger.warning(f"_vlm_extract_b_physics failed for {clip_path}: {exc}")
        return {"subjects": [], "motions": []}


# ---------------------------------------------------------------------------
# Minimum half-widths for fused ranges (prevents collapse to point estimates)
# ---------------------------------------------------------------------------
_FUSED_MIN_HALF = {
    "estimated_height_m":       0.015,
    "estimated_speed_ms":       0.20,
    "estimated_launch_angle_deg": 1.5,
    "estimated_mass_kg":        2.0,
}
_FUSED_DEFAULT_MIN_HALF = 0.05


def _fuse_cv_vlm_ranges(
    cv_value: float,
    cv_conf: float,
    cv_range: Tuple[float, float],
    vlm_center: float,
    vlm_conf: float,
    vlm_range: Tuple[float, float],
    status: str,
    typicality: str,
    quantity_name: str,
) -> Dict[str, Any]:
    """Inverse-variance weighted fusion of CV and VLM estimates.

    Returns {"fused_value": float, "fused_range": [lo, hi], "fused_confidence": float}.
    """
    eps = 1e-9
    h_min = _FUSED_MIN_HALF.get(quantity_name, _FUSED_DEFAULT_MIN_HALF)

    h_c = max((cv_range[1] - cv_range[0]) / 2.0, eps)
    h_v = max((vlm_range[1] - vlm_range[0]) / 2.0, eps)
    sigma_c = max(h_c / 1.96, eps)
    sigma_v = max(h_v / 1.96, eps)
    w_c = cv_conf / sigma_c ** 2
    w_v = vlm_conf / sigma_v ** 2

    mid = (abs(cv_value) + abs(vlm_center)) / 2.0
    d = abs(cv_value - vlm_center) / max(mid, eps)

    # Implausible CV or unphysical — trust VLM only
    if status in ("implausible_cv", "likely_unphysical") or typicality == "likely_unphysical":
        mu = vlm_center
        lo = max(vlm_range[0], mu - h_min)
        hi = min(vlm_range[1], mu + max(h_min, (vlm_range[1] - vlm_range[0]) / 2.0))
        lo = min(lo, mu - h_min)
        hi = max(hi, mu + h_min)
        return {
            "fused_value":      round(mu, 4),
            "fused_range":      [round(lo, 4), round(hi, 4)],
            "fused_confidence": round(min(0.55, 0.55 * vlm_conf), 3),
        }

    mu = (w_c * cv_value + w_v * vlm_center) / (w_c + w_v)
    base_h = max(h_min, 1.64 / (w_c + w_v) ** 0.5)

    if d <= 0.08:
        h = base_h
        conf = min(0.95, 0.5 * cv_conf + 0.5 * vlm_conf + 0.10)
    elif d <= 0.20:
        h = max(h_min, base_h * (1.0 + 0.75 * d))
        conf = min(0.88, 0.5 * cv_conf + 0.5 * vlm_conf)
    elif d <= 0.35:
        h = max(h_min, base_h * (1.0 + 1.5 * d))
        lo = min(mu - h, cv_value, vlm_center)
        hi = max(mu + h, cv_value, vlm_center)
        conf = max(0.40, min(cv_conf, vlm_conf) - 0.10)
        return {
            "fused_value":      round(mu, 4),
            "fused_range":      [round(lo, 4), round(hi, 4)],
            "fused_confidence": round(conf, 3),
        }
    else:
        lo = min(cv_range[0], vlm_range[0])
        hi = max(cv_range[1], vlm_range[1])
        conf = max(0.30, min(cv_conf, vlm_conf) - 0.20)
        return {
            "fused_value":      round(mu, 4),
            "fused_range":      [round(lo, 4), round(hi, 4)],
            "fused_confidence": round(conf, 3),
        }

    lo = mu - h
    hi = mu + h
    return {
        "fused_value":      round(mu, 4),
        "fused_range":      [round(lo, 4), round(hi, 4)],
        "fused_confidence": round(conf, 3),
    }


def _vlm_validate_cv_quantities(
    clip_path: str,
    merged_opq: Dict[str, Any],
    vlm_initial: Dict[str, Any],
) -> Optional[Dict[str, Any]]:
    """One VLM call per clip to cross-validate CV-extracted physical quantities.

    Sends a representative frame + structured JSON with CV values and the VLM's
    own initial estimates. VLM outputs per-quantity agreement status and a
    plausible range from visual inspection.

    Returns a dict keyed by entity label (subject or motion subject:event) with
    per-quantity validation results, or None on failure.
    """
    try:
        import base64
        from openai import OpenAI

        # Extract a representative frame (25% into clip)
        cap = cv2.VideoCapture(clip_path)
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        target_frame = max(0, int(total_frames * 0.25))
        cap.set(cv2.CAP_PROP_POS_FRAMES, target_frame)
        ret, frame = cap.read()
        cap.release()
        if not ret:
            logger.warning("_vlm_validate_cv_quantities: could not read frame")
            return None

        ok, buf = cv2.imencode(".jpg", frame, [cv2.IMWRITE_JPEG_QUALITY, 80])
        if not ok:
            return None
        frame_b64 = base64.b64encode(buf.tobytes()).decode()

        # Build the quantities payload
        quantities = []
        for subj in merged_opq.get("subjects", []):
            label = subj.get("label", "")
            if not label:
                continue
            h_cv  = subj.get("estimated_height_m")
            h_vlm = None
            for vs in vlm_initial.get("subjects", []):
                if vs.get("label", "").lower() == label.lower():
                    h_vlm = vs.get("estimated_height_m")
                    break
            if h_cv is not None:
                pct = subj.get("_measurement_range_pct", 0.12)
                cv_range = [round(h_cv * (1.0 - (pct or 0.12)), 3),
                            round(h_cv * (1.0 + (pct or 0.12)), 3)]
                quantities.append({
                    "entity":       label,
                    "quantity":     "estimated_height_m",
                    "cv_value":     h_cv,
                    "cv_confidence": subj.get("estimation_confidence", 0.5),
                    "cv_raw_range": cv_range,
                    "vlm_initial_estimate": h_vlm,
                })

        for mot in merged_opq.get("motions", []):
            entity_key = f"{mot.get('subject', '')}:{mot.get('event', '')}"
            for qty_name in ("estimated_speed_ms", "estimated_launch_angle_deg", "estimated_height_m"):
                cv_val = mot.get(qty_name)
                if cv_val is None:
                    continue
                pct = mot.get("_measurement_range_pct", 0.12)
                cv_range = [round(cv_val * (1.0 - (pct or 0.12)), 3),
                            round(cv_val * (1.0 + (pct or 0.12)), 3)]
                vlm_val = None
                for vm in vlm_initial.get("motions", []):
                    if (vm.get("subject", "") == mot.get("subject", "")
                            and vm.get("event", "") == mot.get("event", "")):
                        vlm_val = vm.get(qty_name)
                        break
                quantities.append({
                    "entity":        entity_key,
                    "quantity":      qty_name,
                    "cv_value":      cv_val,
                    "cv_confidence": mot.get("estimation_confidence", 0.5),
                    "cv_raw_range":  cv_range,
                    "vlm_initial_estimate": vlm_val,
                })

        if not quantities:
            return None

        prompt = (
            "You are a sports-science measurement validator. "
            "I have extracted physical quantities from a video using computer vision (CV). "
            "For each quantity below, use visual evidence from the provided frame "
            "to assess whether the CV value is plausible. "
            "Do NOT rely solely on the CV numbers — actually look at the image.\n\n"
            "Quantities to validate:\n"
            f"```json\n{json.dumps(quantities, indent=2)}\n```\n\n"
            "For each entry, output:\n"
            '- "agreement_status": one of "agree" | "weak_agree" | "disagree" | '
            '"implausible_cv" | "insufficient_visual_evidence"\n'
            '- "plausible_range_from_video": [lo, hi] (tight interval from visual inspection; '
            "must be ≤±15% of your estimate — do NOT give a wide fallback range)\n"
            '- "typicality_flag": "typical" | "unusual_but_plausible" | "likely_unphysical"\n'
            '- "preferred_center": your best single-value estimate (float)\n'
            '- "confidence_from_vlm": 0.0–1.0\n'
            '- "reason_short": one sentence\n\n'
            "Respond with a JSON array (same order as input), each element containing "
            "all the fields above plus the original entity and quantity fields."
        )

        client = OpenAI(base_url=_OPENROUTER_BASE_URL, api_key=_OPENROUTER_API_KEY)
        resp = client.chat.completions.create(
            model=_VLM_MODEL,
            messages=[{
                "role": "user",
                "content": [
                    {
                        "type":      "image_url",
                        "image_url": {"url": f"data:image/jpeg;base64,{frame_b64}"},
                    },
                    {"type": "text", "text": prompt},
                ],
            }],
            max_tokens=2048,
            timeout=60,
            extra_headers={"HTTP-Referer": "https://github.com/OmniGAIA"},
        )
        raw = resp.choices[0].message.content or ""

        # Parse — accept JSON array possibly wrapped in code fence
        import re as _re
        raw = _re.sub(r"<think>.*?</think>", "", raw, flags=_re.DOTALL)
        m = _re.search(r"```(?:json)?\s*(\[.*?\])\s*```", raw, _re.DOTALL)
        arr_text = m.group(1) if m else _re.search(r"\[.*\]", raw, _re.DOTALL)
        if hasattr(arr_text, "group"):
            arr_text = arr_text.group()
        if not arr_text or not isinstance(arr_text, str):
            logger.warning("_vlm_validate_cv_quantities: could not parse VLM response array")
            return None
        try:
            validated = json.loads(arr_text)
        except Exception as exc:
            logger.warning(f"_vlm_validate_cv_quantities: JSON parse error: {exc}")
            return None

        # Index results by (entity, quantity)
        result_index: Dict[Tuple[str, str], Dict] = {}
        for entry in validated:
            if isinstance(entry, dict):
                key = (entry.get("entity", ""), entry.get("quantity", ""))
                result_index[key] = entry

        return result_index

    except Exception as exc:
        logger.warning(f"_vlm_validate_cv_quantities failed: {exc}")
        return None


def _apply_vlm_validation(
    merged_opq: Dict[str, Any],
    validation_index: Dict[Tuple[str, str], Dict],
) -> None:
    """Inject vlm_validated_range and final_fused_range into merged subjects/motions in-place."""
    def _process_item(item: Dict, entity_key: str, qty_names: List[str]) -> None:
        for qty in qty_names:
            cv_val = item.get(qty)
            if cv_val is None:
                continue
            vr = validation_index.get((entity_key, qty))
            if not vr:
                continue

            plausible = vr.get("plausible_range_from_video")
            preferred = vr.get("preferred_center")
            vlm_conf  = float(vr.get("confidence_from_vlm", 0.5))
            status    = vr.get("agreement_status", "insufficient_visual_evidence")
            typicality = vr.get("typicality_flag", "typical")

            if not (isinstance(plausible, list) and len(plausible) == 2):
                continue
            try:
                vlo, vhi = float(plausible[0]), float(plausible[1])
            except (TypeError, ValueError):
                continue
            if preferred is None:
                preferred = (vlo + vhi) / 2.0
            vlm_center = float(preferred)

            item["vlm_validated_range"]  = [round(vlo, 4), round(vhi, 4)]
            item["vlm_validation_status"] = status
            item["vlm_typicality_flag"]   = typicality

            pct = item.get("_measurement_range_pct", 0.12)
            cv_conf = float(item.get("estimation_confidence", 0.5))
            cv_lo = cv_val * (1.0 - (pct or 0.12))
            cv_hi = cv_val * (1.0 + (pct or 0.12))

            fused = _fuse_cv_vlm_ranges(
                cv_value=cv_val,   cv_conf=cv_conf,   cv_range=(cv_lo, cv_hi),
                vlm_center=vlm_center, vlm_conf=vlm_conf, vlm_range=(vlo, vhi),
                status=status, typicality=typicality, quantity_name=qty,
            )
            item["final_fused_range"]      = fused["fused_range"]
            item["final_fused_confidence"] = fused["fused_confidence"]
            # Update estimation_confidence to reflect fusion
            item["estimation_confidence"]  = fused["fused_confidence"]
            logger.debug(f"Fused {entity_key}/{qty}: CV={cv_val} VLM={vlm_center} "
                         f"→ fused={fused['fused_value']} range={fused['fused_range']}")

    subject_qtys = ["estimated_height_m", "estimated_mass_kg", "estimated_length_m"]
    for subj in merged_opq.get("subjects", []):
        _process_item(subj, subj.get("label", ""), subject_qtys)

    motion_qtys = ["estimated_speed_ms", "estimated_launch_angle_deg", "estimated_height_m"]
    for mot in merged_opq.get("motions", []):
        entity_key = f"{mot.get('subject', '')}:{mot.get('event', '')}"
        _process_item(mot, entity_key, motion_qtys)


_PERSON_KEYWORDS = {
    "athlete", "player", "person", "man", "woman", "boy", "girl",
    "runner", "jumper", "thrower", "gymnast", "swimmer", "cyclist",
    "goalkeeper", "striker", "defender", "coach", "referee",
    "mechanic", "technician", "worker", "operator", "instructor",
    "trainer", "staff", "judge", "umpire", "volunteer", "photographer",
}


def _is_person_label(label: str) -> bool:
    """Return True if the VLM subject label refers to a person."""
    low = label.lower()
    return any(kw in low for kw in _PERSON_KEYWORDS)


def _run_yolo_tracks(clip_path: str) -> Dict[int, List]:
    """Run YOLO11n ByteTrack on full clip; return {track_id: [(fi, cx, cy, bw, bh, cls_id)]}."""
    if not _YOLO_OK:
        return {}
    try:
        model = _get_yolo()
        results = model.track(
            source=clip_path,
            persist=True,
            tracker="bytetrack.yaml",
            conf=0.25,
            iou=0.5,
            verbose=False,
            stream=True,
            classes=None,  # reset any sticky class filter from prior detection calls
        )
        tracks: Dict[int, List] = {}
        frame_idx = 0
        for result in results:
            if result.boxes is not None and result.boxes.id is not None:
                ids   = result.boxes.id.cpu().numpy().astype(int)
                boxes = result.boxes.xyxy.cpu().numpy()
                clss  = result.boxes.cls.cpu().numpy().astype(int)
                for tid, box, cls_id in zip(ids, boxes, clss):
                    cx = (box[0] + box[2]) / 2
                    cy = (box[1] + box[3]) / 2
                    bw = box[2] - box[0]
                    bh = box[3] - box[1]
                    tracks.setdefault(int(tid), []).append(
                        (frame_idx, cx, cy, bw, bh, cls_id)
                    )
            frame_idx += 1
        return tracks
    except Exception as exc:
        logger.warning(f"YOLO tracking failed: {exc}")
        return {}


def route_b_yolo_trajectory(
    clip_path: str,
    fps: float,
    calibration: Dict[str, Any],
    sport_category: str = "",
    vlm_subjects: Optional[List[Dict]] = None,
    precomputed_tracks: Optional[Dict[int, List]] = None,
) -> Dict[str, Any]:
    """
    Route B: YOLO detection + ByteTrack tracking -> trajectory -> physics.

    If precomputed_tracks is provided, skips the tracking pass (reuses shared tracks).
    """
    if not _YOLO_OK:
        return {}

    scale_px_per_m = calibration.get("scale_px_per_m")
    cal_conf       = calibration.get("calibration_confidence", 0.0)

    if precomputed_tracks is not None:
        tracks = precomputed_tracks
    else:
        tracks = _run_yolo_tracks(clip_path)

    if not tracks:
        logger.info("Route B: no tracks found")
        return {}

    motions = []
    subjects_b = []

    # --- Collect all person tracks first, then match to VLM subjects ---
    person_track_candidates: List[Dict] = []
    non_person_tracks: List[Tuple] = []

    for tid, dets in tracks.items():
        if len(dets) < 4:
            continue
        dets.sort(key=lambda d: d[0])
        cls_id = dets[len(dets) // 2][5]
        if cls_id == 0:
            bh_arr_tmp = np.array([d[4] for d in dets], dtype=float)
            bw_arr_tmp = np.array([d[3] for d in dets], dtype=float)
            median_bh  = float(np.median(bh_arr_tmp))
            median_bw  = float(np.median(bw_arr_tmp))
            person_track_candidates.append({
                "tid": tid,
                "dets": dets,
                "median_bh": median_bh,
                "median_area": median_bh * median_bw,
                "track_length": len(dets),
            })
        else:
            non_person_tracks.append((tid, dets))

    # Sort person candidates: largest bbox area first (foreground = main subject)
    person_track_candidates.sort(key=lambda x: x["median_area"], reverse=True)

    # Determine which VLM person subjects to match against
    vlm_person_subjects = [s for s in (vlm_subjects or []) if _is_person_label(s.get("label", ""))]
    if not vlm_person_subjects:
        # No VLM guidance: keep only the single largest (foreground) person track
        vlm_person_subjects = [{"label": "athlete"}]

    # One-to-one assignment: rank-i VLM person subject → rank-i YOLO track
    for i, vs in enumerate(vlm_person_subjects):
        if i >= len(person_track_candidates):
            break
        cand = person_track_candidates[i]
        tid  = cand["tid"]
        median_bh = cand["median_bh"]
        label = vs.get("label", "athlete")

        if scale_px_per_m and median_bh > 0:
            h_m = round(median_bh / scale_px_per_m, 2)
            # Sanity-check against the calibration reference height if available.
            # The YOLO person that was used for calibration should yield ~reference_height_m;
            # other person tracks are typically similar build, so cap at 1.3x the reference.
            ref_h = calibration.get("reference_height_m")
            h_upper = min(2.2, ref_h * 1.30) if ref_h else 2.2
            h_lower = max(1.0, ref_h * 0.70) if ref_h else 1.0
            if not (h_lower <= h_m <= h_upper):
                logger.debug(
                    f"Route B: track {tid} height {h_m}m outside [{h_lower:.2f}, {h_upper:.2f}]m — skipped"
                )
            else:
                subjects_b.append({
                    "label": label,
                    "estimated_height_m": h_m,
                    "estimated_mass_kg": None,
                    "estimated_length_m": None,
                    "other_attributes": {"track_id": int(tid), "track_length": cand["track_length"]},
                    "estimation_basis": (
                        f"YOLO bounding box height, track {tid} (matched to VLM subject '{label}'), "
                        f"scale={scale_px_per_m:.1f} px/m."
                    ),
                    "estimation_confidence": round(min(0.78, 0.52 + cal_conf * 0.35), 2),
                    "estimation_method": "yolo_trajectory",
                })

    logger.info(
        f"Route B: {len(person_track_candidates)} person track(s) detected, "
        f"{len(subjects_b)} matched to VLM subject(s)"
    )

    # --- Person speed from YOLO track displacement (camera-pan corrected) ---
    # Use the track with the highest median displacement speed (not longest),
    # since in tracking shots the main athlete may move most on screen.
    # Camera pan is estimated via optical flow (works even with no BG tracks).
    if scale_px_per_m and person_track_candidates:
        cam_pan_b = _estimate_camera_pan({}, fps, scale_px_per_m, clip_path)

        # Compute median screen speed for each candidate track (min 8 frames)
        cand_speeds: List[Tuple[float, Dict]] = []
        for cand in person_track_candidates:
            if cand["track_length"] < 8:
                continue
            d = cand["dets"]
            xs = _smooth_trajectory(np.array([det[1] for det in d]))
            fi = np.array([det[0] for det in d], dtype=float)
            p_sp = []
            for j in range(1, len(d)):
                dt = (fi[j] - fi[j - 1]) / fps
                if dt > 0:
                    p_sp.append(abs((xs[j] - xs[j - 1]) / scale_px_per_m) / dt)
            if p_sp:
                cand_speeds.append((float(np.median(p_sp)), cand))

        if cand_speeds:
            med_screen_spd, best_cand = max(cand_speeds, key=lambda t: t[0])
            best_dets = best_cand["dets"]
            athlete_spd = round(med_screen_spd + cam_pan_b, 1)
            spd_conf    = round(min(0.78, 0.50 + cal_conf * 0.38), 2)
            vlm_lbl = (vlm_person_subjects[0].get("label", "athlete")
                       if vlm_person_subjects else "athlete")
            pers_mot_class = _motion_observability(best_cand["dets"])
            motions.append({
                "subject": vlm_lbl,
                "event":   "running_speed",
                "time_point_s": round(best_dets[len(best_dets) // 2][0] / fps, 2),
                "estimated_launch_angle_deg": None,
                "estimated_speed_ms": athlete_spd,
                "estimated_height_m": None,
                "estimation_basis": (
                    f"YOLO track {best_cand['tid']} center-x displacement, "
                    f"{best_cand['track_length']} frames, scale={scale_px_per_m:.1f} px/m"
                    + (f", cam_pan=+{cam_pan_b:.2f} m/s" if cam_pan_b > 0 else "") + "."
                ),
                "estimation_confidence": spd_conf,
                "estimation_method": "yolo_trajectory",
                "_motion_observability_class": pers_mot_class,
            })
            logger.info(
                f"Route B: athlete speed (track {best_cand['tid']}) "
                f"screen={med_screen_spd:.1f} + pan={cam_pan_b:.1f} = {athlete_spd:.1f} m/s"
            )

            # Peak velocity event detection (scipy only)
            if _SCIPY_OK:
                try:
                    from scipy.signal import find_peaks as _find_peaks
                    bd = best_cand["dets"]
                    bxs = _smooth_trajectory(np.array([det[1] for det in bd]))
                    bfi = np.array([det[0] for det in bd], dtype=float)
                    b_sps = []
                    b_ts  = []
                    for j in range(1, len(bd)):
                        dt2 = (bfi[j] - bfi[j-1]) / fps
                        if dt2 > 0:
                            b_sps.append(abs((bxs[j] - bxs[j-1]) / scale_px_per_m) / dt2 + cam_pan_b)
                            b_ts.append((bfi[j-1] + bfi[j]) / 2 / fps)
                    if len(b_sps) >= 6:
                        peaks_b, _ = _find_peaks(b_sps, prominence=0.5)
                        if len(peaks_b):
                            pk_idx = peaks_b[int(np.argmax([b_sps[p] for p in peaks_b]))]
                            pk_spd = round(float(b_sps[pk_idx]), 1)
                            pk_t   = round(float(b_ts[pk_idx]), 2)
                            if pk_spd > athlete_spd * 1.2:
                                motions.append({
                                    "subject": vlm_lbl,
                                    "event":   "peak_motion",
                                    "time_point_s": pk_t,
                                    "estimated_launch_angle_deg": None,
                                    "estimated_speed_ms": pk_spd,
                                    "estimated_height_m": None,
                                    "estimation_basis": (
                                        f"Peak velocity at frame t={pk_t:.2f}s "
                                        f"(YOLO track {best_cand['tid']}, prominence-based detection)."
                                    ),
                                    "estimation_confidence": round(spd_conf * 0.9, 2),
                                    "estimation_method": "yolo_trajectory",
                                    "_motion_observability_class": pers_mot_class,
                                })
                except Exception:
                    pass



    # --- Non-person tracks: projectile trajectory fitting ---
    for tid, dets in non_person_tracks:
        if len(dets) < 6:
            continue

        dets.sort(key=lambda d: d[0])
        cls_id   = dets[len(dets) // 2][5]
        cls_name = YOLO_SPORTS_CLASSES.get(cls_id)
        if cls_name is None:
            continue  # Skip unknown YOLO classes — object_N labels have no semantic value in QA

        fi_arr = np.array([d[0] for d in dets], dtype=float)
        cx_arr = np.array([d[1] for d in dets], dtype=float)
        cy_arr = np.array([d[2] for d in dets], dtype=float)

        # Smooth trajectories before fitting to reduce detection noise
        cx_arr = _smooth_trajectory(cx_arr)
        cy_arr = _smooth_trajectory(cy_arr)

        # Try fitting cy vs cx (spatial parabola)
        fit = _fit_parabola(cx_arr, cy_arr)
        use_time = False
        if fit is None or fit[3] < 0.80:
            fit = _fit_parabola(fi_arr, cy_arr)
            use_time = True

        if fit is None or fit[3] < 0.75:
            continue

        a, b, c, r2 = fit

        # Gate: require ≥8 frames and meaningful vertical excursion for reliable launch angle
        _vert_excursion = float(np.max(cy_arr) - np.min(cy_arr))
        _bbox_h_median = float(np.median([d[4] for d in dets if d[4] > 0])) if dets else 0.0
        _min_vert_px = max(15.0, _bbox_h_median * 0.10)
        _launch_angle_reliable = (
            len(fi_arr) >= 8
            and _vert_excursion >= _min_vert_px
            and r2 >= 0.85
        )

        # Launch angle
        if use_time:
            dx_total = cx_arr[-1] - cx_arr[0]
            dt_total = (fi_arr[-1] - fi_arr[0]) / fps
            vx_px = dx_total / (dt_total + 1e-9)
            vy_px = -b * fps
        else:
            cx_start = cx_arr[0]
            slope    = 2 * a * cx_start + b
            vx_px    = fps
            vy_px    = -slope * fps

        launch_angle = (
            round(math.degrees(math.atan2(abs(vy_px), abs(vx_px) + 1e-9)), 1)
            if _launch_angle_reliable else None
        )

        # Speed — use parabola derivative at launch point (not average over trajectory)
        speed_ms = None
        if scale_px_per_m:
            if not use_time:
                # Spatial parabola cy = a*cx^2 + b*cx + c
                # initial slope dy/dx at cx[0] = 2a*cx[0] + b
                n_frames_track = max(len(fi_arr) - 1, 1)
                vx_ppf = (cx_arr[-1] - cx_arr[0]) / n_frames_track
                initial_slope = 2 * a * cx_arr[0] + b
                vy_ppf = -initial_slope * abs(vx_ppf)  # screen y flipped
            else:
                # Time-based parabola cy = a*t^2 + b*t + c
                # vy at t0 = b + 2a*t0  (px/frame)
                t0     = fi_arr[0]
                vy_ppf = -(b + 2 * a * t0)            # screen y flipped
                vx_ppf = (cx_arr[-1] - cx_arr[0]) / max(len(fi_arr) - 1, 1)
            vx_ms    = abs(vx_ppf) * fps / scale_px_per_m
            vy_ms    = abs(vy_ppf) * fps / scale_px_per_m
            speed_ms = round(math.sqrt(vx_ms**2 + vy_ms**2), 1)

        # Height at launch
        height_m = None
        if scale_px_per_m:
            cap_tmp = cv2.VideoCapture(clip_path)
            fh = int(cap_tmp.get(cv2.CAP_PROP_FRAME_HEIGHT))
            cap_tmp.release()
            height_m = round((fh - cy_arr[0]) / scale_px_per_m, 1) if fh > 0 else None

        conf = round(min(0.85, 0.60 + r2 * 0.20 + cal_conf * 0.05), 2)

        # Check motion observability for depth-motion suppression
        raw_dets = tracks.get(tid, [])
        mot_class = _motion_observability(raw_dets)

        mot_entry = {
            "subject": cls_name,
            "event": "trajectory",
            "time_point_s": round(float(fi_arr[0]) / fps, 2),
            "estimated_launch_angle_deg": launch_angle,
            "estimated_speed_ms": speed_ms,
            "estimated_height_m": height_m,
            "estimation_basis": (
                f"YOLO track {tid} ({cls_name}), {len(dets)} detections, "
                f"parabola R^2={r2:.3f}, {'time' if use_time else 'spatial'} fit."
                + (f" Scale={scale_px_per_m:.1f} px/m." if scale_px_per_m else " No metric scale.")
            ),
            "estimation_confidence": conf,
            "estimation_method": "yolo_trajectory",
            "_motion_observability_class": mot_class,
        }
        motions.append(mot_entry)

    logger.info(f"Route B: {len(subjects_b)} person subject(s), {len(motions)} trajectory(ies)")
    # _background_tracks is consumed by augment_opq for camera-pan correction in Route A
    bg_tracks = {tid: dets for tid, dets in tracks.items()
                 if len(dets) >= 4 and dets[len(dets) // 2][5] != 0}
    return {"subjects": subjects_b, "motions": motions, "_background_tracks": bg_tracks}


# ===========================================================================
# Route C — Merge VLM + CV results
# ===========================================================================

def route_c_merge(
    vlm_opq: Dict[str, Any],
    route_a: Dict[str, Any],
    route_b: Dict[str, Any],
    calibration: Dict[str, Any],
) -> Dict[str, Any]:
    cv_subjects = route_a.get("subjects", []) + route_b.get("subjects", [])
    cv_motions  = route_a.get("motions",  []) + route_b.get("motions",  [])

    # Assign measurement tier to every CV item before merging
    for item in cv_subjects:
        mot_class = item.pop("_motion_observability_class", "unknown")
        _assign_tier(item, calibration, mot_class)
    for item in cv_motions:
        mot_class = item.pop("_motion_observability_class", "unknown")
        _assign_tier(item, calibration, mot_class)

    merged_subjects = _merge_subjects(vlm_opq.get("subjects", []), cv_subjects)
    merged_motions  = _merge_motions(vlm_opq.get("motions",  []), cv_motions)

    uncertain = []
    for s in merged_subjects:
        if s.get("estimation_confidence", 1.0) < 0.7:
            uncertain.append(f"subject:{s.get('label','?')}:height")
    for m in merged_motions:
        conf = m.get("estimation_confidence", 1.0)
        if conf < 0.7:
            if m.get("estimated_launch_angle_deg") is not None:
                uncertain.append(f"motion:{m.get('subject','?')}:{m.get('event','?')}:angle")
            if m.get("estimated_speed_ms") is not None:
                uncertain.append(f"motion:{m.get('subject','?')}:{m.get('event','?')}:speed")

    result = dict(vlm_opq)
    result["subjects"]                = merged_subjects
    result["motions"]                 = merged_motions
    result["cv_extraction_available"] = bool(cv_subjects or cv_motions)
    result["uncertain_quantities"]    = uncertain
    result["calibration"]             = calibration

    return result


def _merge_subjects(vlm_list, cv_list):
    cv_by_label = {}
    for cs in cv_list:
        key = cs.get("label", "").lower()
        if cs.get("estimation_confidence", 0) > cv_by_label.get(key, {}).get("estimation_confidence", -1):
            cv_by_label[key] = cs

    merged = []
    for vs in vlm_list:
        label_key = vs.get("label", "").lower()
        cv = cv_by_label.get(label_key) or _fuzzy_match_subject(vs, cv_by_label)

        if cv and cv.get("estimation_confidence", 0) >= 0.7:
            s = dict(cv)
            for field in ("estimated_mass_kg", "estimated_length_m", "other_attributes"):
                if s.get(field) is None and vs.get(field) is not None:
                    s[field] = vs[field]
            merged.append(s)
        else:
            s = dict(vs)
            if "estimation_method" not in s:
                s["estimation_method"] = "vlm"
            if "estimation_confidence" not in s:
                s["estimation_confidence"] = _vlm_confidence(vs, cv)
            if cv:
                s["cv_notes"] = {
                    "cv_method": cv.get("estimation_method"),
                    "cv_confidence": cv.get("estimation_confidence"),
                    "cv_height_m": cv.get("estimated_height_m"),
                }
            merged.append(s)

    existing_labels = {s.get("label", "").lower() for s in merged}
    for cs in cv_list:
        if cs.get("label", "").lower() not in existing_labels:
            merged.append(cs)

    return merged


def _merge_motions(vlm_list, cv_list):
    merged = list(vlm_list)
    for m in merged:
        if "estimation_method" not in m:
            m["estimation_method"] = "vlm"
        if "estimation_confidence" not in m:
            m["estimation_confidence"] = 0.50

    for cv_m in cv_list:
        best_vlm = _find_best_vlm_motion_match(cv_m, merged)
        if best_vlm is not None and cv_m.get("estimation_confidence", 0) >= 0.7:
            for field in ("estimated_launch_angle_deg", "estimated_speed_ms", "estimated_height_m"):
                if cv_m.get(field) is not None:
                    best_vlm[field] = cv_m[field]
            best_vlm["estimation_method"]    = cv_m["estimation_method"]
            best_vlm["estimation_confidence"] = cv_m["estimation_confidence"]
            best_vlm["estimation_basis"]      = cv_m.get("estimation_basis", "")
        else:
            merged.append(cv_m)

    return merged


def _fuzzy_match_subject(vs, cv_by_label):
    label = vs.get("label", "").lower()
    for key in cv_by_label:
        if key in label or label in key:
            return cv_by_label[key]
    return None


def _vlm_confidence(vs, cv):
    if cv is None:
        return 0.45
    cv_h = cv.get("estimated_height_m")
    vs_h = vs.get("estimated_height_m")
    if cv_h and vs_h:
        if abs(cv_h - vs_h) / (cv_h + 1e-9) < 0.10:
            return 0.65
        return 0.40
    return 0.45


def _find_best_vlm_motion_match(cv_m, vlm_list):
    cv_t   = cv_m.get("time_point_s", 0) or 0
    cv_sub = (cv_m.get("subject") or "").lower()
    best, best_score = None, float("inf")
    for vm in vlm_list:
        vm_t   = vm.get("time_point_s", 0) or 0
        vm_sub = (vm.get("subject") or "").lower()
        time_diff = abs(cv_t - vm_t)
        label_ok  = cv_sub in vm_sub or vm_sub in cv_sub or cv_sub == "" or vm_sub == ""
        if label_ok and time_diff < best_score:
            best_score = time_diff
            best = vm
    return best if best_score < 3.0 else None


# ===========================================================================
# Top-level entry point
# ===========================================================================

def _lookup_ref_size(label: str, sport_category: str) -> Tuple[Optional[float], Optional[float]]:
    """Return (known_h_m, known_w_m) for a detected label from all calibration tables."""
    label_low = label.lower()
    # 1. Sport-specific table
    sport_key = sport_category.lower() if sport_category else "default"
    for entries in [SPORT_CALIBRATION.get(sport_key, []), SPORT_CALIBRATION.get("default", [])]:
        for keywords, h_m, w_m in entries:
            if any(kw in label_low for kw in keywords):
                return h_m, w_m
    # 2. Vehicle / traffic table
    for _key, (keywords, h_m, w_m, _var) in VEHICLE_REF_OBJECTS.items():
        if any(kw in label_low for kw in keywords):
            return h_m, w_m
    # 3. Indoor / daily life table
    for _key, (keywords, h_m, w_m, _var) in INDOOR_REF_OBJECTS.items():
        if any(kw in label_low for kw in keywords):
            return h_m, w_m
    # 4. General objects table (legacy)
    for _key, (keywords, h_m, w_m, *_) in GENERAL_REF_OBJECTS.items():
        if any(kw in label_low for kw in keywords):
            return h_m, w_m
    return None, None


def _detect_ref_objects(
    clip_path: str,
    sport_category: str = "",
    n_frames: int = 12,
) -> List[Dict]:
    """
    Sample n_frames from clip, run YOLO11n + YOLOWorld, return reference object detections
    with median bbox dimensions and known real-world sizes where available.

    Returns List[Dict] with keys:
        label, height_px, width_px, known_h_m, known_w_m, confidence, source
    """
    if not _YOLO_OK:
        return []

    all_detections: Dict[str, List[Tuple[float, float, float]]] = {}  # label -> [(h_px, w_px, conf)]

    try:
        yolo11 = _get_yolo()
        cap = cv2.VideoCapture(clip_path)
        total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT)) or 1
        step  = max(1, total // n_frames)

        for i in range(0, total, step):
            cap.set(cv2.CAP_PROP_POS_FRAMES, i)
            ret, frame = cap.read()
            if not ret:
                break
            res = yolo11.predict(frame, verbose=False, conf=0.3)[0]
            if res.boxes is None:
                continue
            boxes = res.boxes.xyxy.cpu().numpy()
            clss  = res.boxes.cls.cpu().numpy().astype(int)
            confs = res.boxes.conf.cpu().numpy()
            for box, cls_id, conf in zip(boxes, clss, confs):
                bh = float(box[3] - box[1])
                bw = float(box[2] - box[0])
                name = YOLO_SPORTS_CLASSES.get(cls_id)
                if name is None:
                    continue  # Skip unknown classes — object_N labels pollute QA
                all_detections.setdefault(name, []).append((bh, bw, float(conf)))

        cap.release()
    except Exception as exc:
        logger.warning(f"YOLO11n ref detection failed: {exc}")

    # Run YOLOWorld for sport-specific + general vocabulary
    yw_model = _get_yolo_world()
    if yw_model is not None:
        sport_key = sport_category.lower() if sport_category else "default"
        vocab: List[str] = []
        for entries in [SPORT_CALIBRATION.get(sport_key, []), SPORT_CALIBRATION.get("default", [])]:
            for keywords, _, _ in entries:
                vocab.extend(keywords)
        for _, (keywords, _, _, *_rest) in GENERAL_REF_OBJECTS.items():
            vocab.extend(keywords)
        vocab = list(dict.fromkeys(v for v in vocab if v))  # deduplicate, preserve order

        if vocab:
            try:
                yw_model.set_classes(vocab)
                # set_classes() re-creates text embeddings on CPU; move back to device
                try:
                    import torch as _torch
                    _dev = "cuda:0" if _torch.cuda.is_available() else "cpu"
                    yw_model.model.to(_dev)
                except Exception:
                    pass
                cap = cv2.VideoCapture(clip_path)
                total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT)) or 1
                step  = max(1, total // n_frames)
                for i in range(0, total, step):
                    cap.set(cv2.CAP_PROP_POS_FRAMES, i)
                    ret, frame = cap.read()
                    if not ret:
                        break
                    res = yw_model.predict(frame, verbose=False, conf=0.25)[0]
                    if res.boxes is None:
                        continue
                    boxes = res.boxes.xyxy.cpu().numpy()
                    clss  = res.boxes.cls.cpu().numpy().astype(int)
                    confs = res.boxes.conf.cpu().numpy()
                    for box, cls_id, conf in zip(boxes, clss, confs):
                        if cls_id < len(vocab):
                            bh   = float(box[3] - box[1])
                            bw   = float(box[2] - box[0])
                            name = vocab[cls_id]
                            all_detections.setdefault(name, []).append((bh, bw, float(conf)))
                cap.release()
            except Exception as exc:
                logger.warning(f"YOLOWorld ref detection failed: {exc}")

    # Build result list
    result: List[Dict] = []
    for label, samples in all_detections.items():
        heights = [s[0] for s in samples]
        widths  = [s[1] for s in samples]
        confs   = [s[2] for s in samples]
        med_h   = float(np.median(heights))
        med_w   = float(np.median(widths))
        med_c   = float(np.median(confs))
        known_h, known_w = _lookup_ref_size(label, sport_category)
        result.append({
            "label":    label,
            "height_px": round(med_h, 1),
            "width_px":  round(med_w, 1),
            "known_h_m": known_h,
            "known_w_m": known_w,
            "confidence": round(med_c, 3),
            "source": "yolo11n+yoloworld",
        })

    return result


# ---------------------------------------------------------------------------
# Scene-object physics helpers (Stage 1 visual_elements → augment_opq)
# ---------------------------------------------------------------------------

_PERSON_VISUAL_KEYWORDS: set = {
    "player", "athlete", "person", "woman", "man", "girl", "boy",
    "female", "male", "coach", "referee", "official", "spectator",
    "crowd", "audience", "people", "human", "participant", "competitor",
    "runner", "gymnast", "cyclist", "swimmer",
    "mechanic", "technician", "worker", "operator", "instructor",
    "trainer", "staff", "judge", "umpire", "volunteer", "photographer",
}

_VISUAL_STOP_WORDS: set = {
    "a", "an", "the", "of", "in", "on", "at", "to", "for", "from",
    "by", "with", "and", "or", "is", "are", "was", "were",
}

# Labels that produce physically meaningless VLM queries (not discrete measurable objects)
_VISUAL_SCENE_WORDS: set = {
    "background", "floor", "ceiling", "wall", "room", "area", "environment",
    "setting", "scene", "field", "court", "ground", "surface", "space",
    "interior", "exterior", "shop", "workshop", "gym", "stadium", "arena",
}

_VISUAL_VERB_WORDS: set = {
    "being", "dribbling", "dribbled", "bouncing", "bounced", "reflecting",
    "hanging", "decorated", "standing", "walking", "practicing", "running",
    "wearing", "executing", "gathering", "raising", "lowering", "releasing",
    "marking", "sitting", "floating", "placed", "positioned",
}


def _parse_visual_elements(visual_elements: List[str]) -> List[str]:
    """Extract non-person object labels from Stage 1 visual_elements descriptions.

    Returns at most 5 simple labels usable as VLM bbox query strings.

    Key rule: only skip a description if it STARTS with a person keyword.
    Descriptions like "Detached bicycle wheel held by the mechanic" must still
    yield the object label ("detached bicycle wheel") even though they mention
    a person later.
    """
    results: List[str] = []
    seen: set = set()
    for desc in visual_elements:
        words = desc.strip().split()
        if not words:
            continue
        # Skip only when description STARTS with a person keyword
        first_clean = words[0].strip(".,!?;:'\"").lower()
        # Also strip possessive suffix so "mechanic's" matches "mechanic"
        first_stem = first_clean.rstrip("'s") if first_clean.endswith("'s") else first_clean
        if first_clean in _PERSON_VISUAL_KEYWORDS or first_stem in _PERSON_VISUAL_KEYWORDS:
            continue
        # Extract object label: take first N significant words, stopping at
        # stop words (after at least one word collected) or verb words.
        label_words: List[str] = []
        for w in words[:6]:
            clean = w.strip(".,!?;:'\"").lower()
            if not clean:
                continue
            if clean in _VISUAL_STOP_WORDS:
                if label_words:
                    break  # "of", "on", etc. after a noun phrase ends it
                continue   # skip leading articles
            if clean in _VISUAL_VERB_WORDS:
                if label_words:
                    break
                continue
            # Stop if a person keyword appears mid-description (object name ended)
            if clean in _PERSON_VISUAL_KEYWORDS:
                break
            label_words.append(clean)
            if len(label_words) >= 3:
                break
        if not label_words:
            continue
        # Skip scene/environment labels that are not discrete measurable objects
        if any(sw in label_words for sw in _VISUAL_SCENE_WORDS):
            continue
        label = " ".join(label_words)
        if label not in seen:
            seen.add(label)
            results.append(label)
        if len(results) >= 5:
            break
    return results


def _extract_object_physics(
    clip_path: str,
    label: str,
    calibration: Dict[str, Any],
    shared_tracks: Dict,
    fps: float,
    frame_indices: Optional[set] = None,
) -> Optional[Dict[str, Any]]:
    """Extract size (VLM bbox) and speed (YOLO track) for a named scene object.

    Returns {"subject": {...}, "motion": {...} | None}, or None if object not found.
    Metric values are suppressed when calibration is not anchor-based.
    """
    vlm_bbox = _vlm_locate_bbox(clip_path, label, frame_indices=frame_indices)
    if vlm_bbox is None:
        return None

    height_px = float(vlm_bbox.get("height_px", 0))
    width_px  = float(vlm_bbox.get("width_px", 0))
    vlm_conf  = float(vlm_bbox.get("confidence", 0.5))

    scale     = calibration.get("scale_px_per_m")
    is_anchor = calibration.get("is_anchor_based", False)

    subj: Dict[str, Any] = {
        "label": label,
        "source": "cv_scene_object",
        "estimation_method": "vlm_bbox" + ("+calibration" if is_anchor and scale else ""),
        "estimation_confidence": round(vlm_conf * (0.85 if is_anchor else 0.50), 3),
    }

    if is_anchor and scale and scale > 0:
        subj["estimated_height_m"] = round(height_px / scale, 3) if height_px > 0 else None
        subj["estimated_width_m"]  = round(width_px / scale, 3)  if width_px > 0  else None
        subj["estimation_basis"]   = (
            f"VLM bbox h={height_px:.1f}px w={width_px:.1f}px, scale={scale:.1f}px/m"
        )
    else:
        subj["estimated_height_m"] = None
        subj["estimated_width_m"]  = None
        subj["estimation_basis"]   = (
            "VLM bbox only; metric suppressed (no anchor-based calibration)"
        )

    # Speed from YOLO track whose label matches
    motion: Optional[Dict[str, Any]] = None
    if is_anchor and scale and scale > 0 and shared_tracks:
        best_speed: Optional[float] = None
        best_tid: Optional[int] = None
        for tid, dets in shared_tracks.items():
            if len(dets) < 5:
                continue
            mid   = dets[len(dets) // 2]
            cls_id = int(mid[5]) if len(mid) > 5 else -1
            # _YOLO_CLS_NAMES resolved at call time (defined later in module)
            track_label = _YOLO_CLS_NAMES.get(cls_id, f"obj_{cls_id}")  # type: ignore[name-defined]
            if not _label_matches(label, track_label):
                continue
            sorted_dets = sorted(dets, key=lambda x: x[0])
            cxs = _smooth_trajectory(np.array([float(d[1]) for d in sorted_dets]))
            cys = _smooth_trajectory(np.array([float(d[2]) for d in sorted_dets]))
            fis = np.array([float(d[0]) for d in sorted_dets])
            speeds_local: List[float] = []
            for j in range(1, len(sorted_dets)):
                dt = (fis[j] - fis[j - 1]) / fps
                if dt > 0:
                    dx = (cxs[j] - cxs[j - 1]) / scale
                    dy = (cys[j] - cys[j - 1]) / scale
                    speeds_local.append(math.sqrt(dx ** 2 + dy ** 2) / dt)
            if speeds_local:
                spd = float(np.median(speeds_local))
                if best_speed is None or spd > best_speed:
                    best_speed = spd
                    best_tid = tid

        if best_speed is not None:
            motion = {
                "subject":                   label,
                "event":                     "movement",
                "estimated_speed_ms":        round(best_speed, 2),
                "estimated_height_m":        None,
                "estimated_launch_angle_deg": None,
                "estimation_confidence":     round(min(0.70, vlm_conf * 0.85), 3),
                "estimation_method":         "yolo_track_displacement",
                "estimation_basis": (
                    f"YOLO track {best_tid} cx/cy displacement, scale={scale:.1f}px/m"
                ),
                "_motion_observability_class": "directly_visible",
            }

    return {"subject": subj, "motion": motion}


# ===========================================================================
# B2 Scale-Free Extraction
# ===========================================================================

def _b2_joint_angle_deg(lm, a_key: str, v_key: str, b_key: str, fw: int, fh: int) -> Optional[float]:
    """Compute angle at vertex v_key between joints a_key and b_key (degrees)."""
    import math
    a = _get_lm(lm, a_key)
    v = _get_lm(lm, v_key)
    b = _get_lm(lm, b_key)
    if not (a and v and b and a[2] > 0.5 and v[2] > 0.5 and b[2] > 0.5):
        return None
    ax = (a[0] - v[0]) * fw
    ay = (a[1] - v[1]) * fh
    bx = (b[0] - v[0]) * fw
    by = (b[1] - v[1]) * fh
    denom = math.sqrt(ax * ax + ay * ay) * math.sqrt(bx * bx + by * by)
    if denom < 1e-9:
        return None
    cos_a = (ax * bx + ay * by) / denom
    return round(math.degrees(math.acos(max(-1.0, min(1.0, cos_a)))), 1)


def _b2_extract_pose_angles(
    clip_path: str,
    fps: float,
    frame_indices: Optional[List[int]] = None,
) -> List[Dict]:
    """Extract peak joint angles from pose estimation. Returns subject-format OPQ items."""
    import math

    lm_frames = None
    fw = fh = 0
    if _YOLO_OK:
        try:
            _get_yolo_pose()
            if _YOLO_POSE_OK:
                lm_frames, _, (fw, fh) = _run_yolo_pose(clip_path)
        except Exception as exc:
            logger.warning(f"B2 pose angles: YOLO-pose failed ({exc})")

    if lm_frames is None and _MEDIAPIPE_OK:
        try:
            lm_frames, _, (fw, fh) = _run_pose(clip_path)
        except Exception as exc:
            logger.warning(f"B2 pose angles: MediaPipe failed ({exc})")

    if lm_frames is None or fw == 0:
        return []

    valid = [(i, lm) for i, lm in enumerate(lm_frames) if lm is not None]
    if frame_indices:
        fi_set = set(frame_indices)
        valid = [(i, lm) for i, lm in valid if i in fi_set]
    if len(valid) < 3:
        return []

    # Angle definitions: (side_a, vertex, side_b, canonical_name, peak_is="max"|"min")
    ANGLE_DEFS = [
        ("left_shoulder",  "left_elbow",  "left_wrist",  "peak_elbow_angle",  "max"),
        ("right_shoulder", "right_elbow", "right_wrist", "peak_elbow_angle",  "max"),
        ("left_hip",       "left_knee",   "left_ankle",  "peak_knee_angle",   "min"),
        ("right_hip",      "right_knee",  "right_ankle", "peak_knee_angle",   "min"),
    ]

    angle_series: Dict[str, List[float]] = {}
    for _, lm in valid:
        # Trunk lean first — gate joint angles on body uprightness
        ls = _get_lm(lm, "left_shoulder")
        rs = _get_lm(lm, "right_shoulder")
        lh = _get_lm(lm, "left_hip")
        rh = _get_lm(lm, "right_hip")
        body_upright = False
        if all(x and x[2] > 0.3 for x in [ls, rs, lh, rh]):
            sh_x = (ls[0] + rs[0]) / 2
            sh_y = (ls[1] + rs[1]) / 2
            hp_x = (lh[0] + rh[0]) / 2
            hp_y = (lh[1] + rh[1]) / 2
            dx = (sh_x - hp_x) * fw
            dy = (sh_y - hp_y) * fh
            lean = round(math.degrees(math.atan2(abs(dx), abs(dy) + 1e-9)), 1)
            angle_series.setdefault("trunk_lean_angle", []).append(lean)
            body_upright = lean < 45.0  # only compute limb angles when roughly upright

        # Elbow / knee angles: skip for bent-over postures (lean ≥ 45° from vertical)
        if not body_upright:
            continue
        for a_key, v_key, b_key, name, _ in ANGLE_DEFS:
            ang = _b2_joint_angle_deg(lm, a_key, v_key, b_key, fw, fh)
            if ang is not None:
                angle_series.setdefault(name, []).append(ang)

    # Select peak value per angle name and deduplicate left/right
    best_by_name: Dict[str, Dict] = {}
    for a_key, v_key, b_key, name, peak_dir in ANGLE_DEFS:
        vals = angle_series.get(name, [])
        if not vals:
            continue
        peak = max(vals) if peak_dir == "max" else min(vals)
        stability = max(0.0, 1.0 - float(np.std(vals)) / (abs(peak) + 1e-9))
        conf = round(min(0.84, 0.72 + stability * 0.12), 2)
        if conf < 0.80:
            continue
        existing = best_by_name.get(name)
        if existing is None or conf > existing["estimation_confidence"]:
            best_by_name[name] = {
                "label": "athlete",
                "estimated_height_m": None,
                "estimated_angle_deg": peak,
                "angle_name": name,
                "estimation_basis": (
                    f"Pose estimation: {name} = {peak}° "
                    f"(n_frames={len(vals)}, stability={stability:.2f})"
                ),
                "estimation_confidence": conf,
                "estimation_method": "pose_estimation",
            }

    # Trunk lean
    trunk_vals = angle_series.get("trunk_lean_angle", [])
    if trunk_vals:
        peak_trunk = round(float(np.mean(trunk_vals)), 1)
        stability = max(0.0, 1.0 - float(np.std(trunk_vals)) / (abs(peak_trunk) + 1e-9))
        conf = round(min(0.82, 0.72 + stability * 0.10), 2)
        if conf >= 0.80:
            best_by_name["trunk_lean_angle"] = {
                "label": "athlete",
                "estimated_height_m": None,
                "estimated_angle_deg": peak_trunk,
                "angle_name": "trunk_lean_angle",
                "estimation_basis": (
                    f"Pose estimation: trunk lean = {peak_trunk}° from vertical "
                    f"(n_frames={len(trunk_vals)}, stability={stability:.2f})"
                ),
                "estimation_confidence": conf,
                "estimation_method": "pose_estimation",
            }

    return list(best_by_name.values())


def _resolve_label_from_scene(cls_id: int, generic_label: str, scene_objects: List[str]) -> str:
    """Match YOLO COCO class to the most specific VLM-described object from scene_objects.

    For example, cls 32 ("sports ball") + scene_objects containing "white cue ball"
    → returns "white cue ball" instead of the generic label.
    """
    if not scene_objects:
        return generic_label
    # COCO class → keywords to look for in VLM scene descriptions
    _CLS_KEYWORDS: Dict[int, List[str]] = {
        32: ["ball", "puck"],
        36: ["skateboard", "board"],
        37: ["surfboard"],
        38: ["racket", "racquet"],
        43: ["knife", "blade"],
        46: ["banana"],
        47: ["apple"],
        50: ["clock"],
        56: ["chair"],
        57: ["couch", "sofa"],
        63: ["laptop"],
        67: ["phone", "mobile"],
        73: ["book"],
        76: ["scissors"],
    }
    keywords = _CLS_KEYWORDS.get(cls_id, [generic_label.lower()])
    for kw in keywords:
        for obj in scene_objects:
            if kw in obj.lower():
                # Return VLM description (trimmed to last 4 words for conciseness)
                words = obj.strip().split()
                return " ".join(words[-min(4, len(words)):]).strip()
    return generic_label


def _detect_scene_reference_line(clip_path: str) -> Dict[str, Any]:
    """Detect the dominant structural line in the scene for use as an angle reference.

    Runs Hough line transform on the middle frame of the clip.
    The dominant line is used to express trajectory_angle relative to a physical
    scene feature rather than arbitrary screen coordinates.

    Returns:
        {
          "angle_deg": float,   # line angle from horizontal, range (-90, 90]
          "name": str,          # human-readable reference name for question text
          "confidence": float,  # 0.0 (fallback) to ~0.85
        }
    """
    _FALLBACK = {"angle_deg": 0.0, "name": "horizontal (default)", "confidence": 0.0}
    try:
        cap = cv2.VideoCapture(clip_path)
        total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT)) or 1
        cap.set(cv2.CAP_PROP_POS_FRAMES, total // 2)
        ret, frame = cap.read()
        cap.release()
        if not ret or frame is None:
            return _FALLBACK

        h_img, w_img = frame.shape[:2]
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        edges = cv2.Canny(gray, 50, 150, apertureSize=3)
        lines = cv2.HoughLinesP(
            edges, 1, np.pi / 180, threshold=80,
            minLineLength=int(min(w_img, h_img) * 0.15),
            maxLineGap=20,
        )
        if lines is None or len(lines) == 0:
            return _FALLBACK

        # Compute angle and length for each detected segment
        angles, lengths = [], []
        for seg in lines:
            x1, y1, x2, y2 = seg[0]
            a = math.degrees(math.atan2(y2 - y1, x2 - x1))
            # Normalise to (-90, 90]: lines have no direction sense
            a = a % 180
            if a > 90:
                a -= 180
            length = math.sqrt((x2 - x1) ** 2 + (y2 - y1) ** 2)
            angles.append(a)
            lengths.append(length)

        angles_arr = np.array(angles)
        lengths_arr = np.array(lengths)
        total_len = float(lengths_arr.sum()) or 1.0

        # Histogram over 18 bins of 10° each spanning (-90°, 90°)
        hist = np.zeros(18)
        for a, ln in zip(angles_arr, lengths_arr):
            bucket = int((a + 90) / 10) % 18
            hist[bucket] += ln

        best_bucket = int(np.argmax(hist))
        dominant_angle = float(best_bucket * 10 - 90 + 5)   # centre of bucket

        # Confidence: fraction of total line length in the dominant bucket
        conf_raw = hist[best_bucket] / total_len
        confidence = float(min(0.82, 0.45 + conf_raw * 0.5))

        # Human-readable reference name
        abs_a = abs(dominant_angle)
        if abs_a <= 15:
            name = "horizontal surface (ground/table)"
        elif abs_a >= 75:
            name = "vertical surface (wall)"
        else:
            sign_str = "upward-right" if dominant_angle > 0 else "upward-left"
            name = f"inclined surface at {abs_a:.0f}° from horizontal ({sign_str})"

        return {"angle_deg": dominant_angle, "name": name, "confidence": confidence}

    except Exception as exc:
        logger.debug(f"_detect_scene_reference_line failed ({clip_path}): {exc}")
        return _FALLBACK


def _pca_axis_from_stationary(shared_tracks: Dict[int, List]) -> Optional[float]:
    """Return principal axis angle (degrees) of stationary objects via PCA.

    Identifies tracks with net displacement < 0.5 × mean bbox width (stationary
    objects, e.g. snooker balls at rest), collects their mean center positions,
    and returns the first principal component direction.

    Convention: physics angle (0°=right, 90°=up); normalised to (-90°, 90°] so
    both ±directions of the same axis map to the same value — no sign ambiguity.

    Returns None when fewer than 2 stationary objects are found (caller falls
    back to Hough-based _detect_scene_reference_line).
    """
    import math
    centers = []
    for tid, dets in shared_tracks.items():
        if len(dets) < 4:
            continue
        cls = int(dets[len(dets) // 2][5])
        if cls == 0:          # skip persons
            continue
        cx_vals = [float(d[1]) for d in dets]
        cy_vals = [float(d[2]) for d in dets]
        w_vals  = [float(d[3]) for d in dets]
        mean_cx = float(np.mean(cx_vals))
        mean_cy = float(np.mean(cy_vals))
        mean_w  = max(1.0, float(np.mean(w_vals)))
        net_disp = math.sqrt((cx_vals[-1] - cx_vals[0]) ** 2 +
                             (cy_vals[-1] - cy_vals[0]) ** 2)
        if net_disp < mean_w * 0.5:      # stationary: displacement < half bbox width
            centers.append([mean_cx, mean_cy])

    if len(centers) < 2:
        return None

    pts = np.array(centers, dtype=float)
    pts_c = pts - pts.mean(axis=0)
    cov = np.cov(pts_c.T)
    if cov.ndim < 2:          # degenerate (all points collinear on one axis)
        return None
    eigvals, eigvecs = np.linalg.eigh(cov)   # eigh: symmetric, real eigenvalues
    primary = eigvecs[:, np.argmax(eigvals)]
    # Physics convention: flip screen-y (screen y increases downward)
    angle = math.degrees(math.atan2(-primary[1], primary[0]))
    # Normalise to (-90°, 90°]: axis has no preferred direction
    if angle <= -90.0:
        angle += 180.0
    elif angle > 90.0:
        angle -= 180.0
    logger.debug(
        f"PCA axis: {angle:.1f}° from {len(centers)} stationary objects "
        f"(centers: {[(round(c[0],0), round(c[1],0)) for c in centers[:5]]})"
    )
    return round(angle, 1)


_MULTI_TRACK_CLASSES: set = {32, 34, 35}   # sports ball, frisbee, puck variants
_MAX_BALL_TRACKS: int = 3                  # max tracks to keep per ball-like class


def _dominant_ball_color(clip_path: str, dets: List, n_samples: int = 5) -> str:
    """Identify the dominant color of a ball from a YOLO track.

    Samples N evenly-spaced detections, extracts the central 60% of each
    bounding box (avoids background contamination), converts to HSV, and
    takes the per-channel median.  Returns one of:
        'white' | 'red' | 'pink' | 'black' | 'yellow' | 'brown' |
        'blue' | 'green' | 'orange' | 'purple' | 'unknown'
    """
    if not clip_path or not dets:
        return "unknown"
    try:
        import cv2 as _cv2
        cap = _cv2.VideoCapture(clip_path)
        fh  = int(cap.get(_cv2.CAP_PROP_FRAME_HEIGHT))
        fw  = int(cap.get(_cv2.CAP_PROP_FRAME_WIDTH))

        step    = max(1, len(dets) // n_samples)
        sampled = dets[::step][:n_samples]

        hsv_meds: List[Tuple[float, float, float]] = []
        for det in sampled:
            frame_idx = int(det[0])
            cx = float(det[1]); cy = float(det[2])
            bw = float(det[3]); bh = float(det[4])

            # Shrink bbox to central 60% to avoid playing-surface bleed
            px = bw * 0.2; py = bh * 0.2
            x1 = int(max(0,      cx - bw / 2 + px))
            x2 = int(min(fw - 1, cx + bw / 2 - px))
            y1 = int(max(0,      cy - bh / 2 + py))
            y2 = int(min(fh - 1, cy + bh / 2 - py))
            if x2 <= x1 or y2 <= y1:
                continue

            cap.set(_cv2.CAP_PROP_POS_FRAMES, frame_idx)
            ret, frame = cap.read()
            if not ret:
                continue

            roi = frame[y1:y2, x1:x2]
            if roi.size == 0:
                continue

            hsv = _cv2.cvtColor(roi, _cv2.COLOR_BGR2HSV)
            hsv_meds.append((
                float(np.median(hsv[:, :, 0])),
                float(np.median(hsv[:, :, 1])),
                float(np.median(hsv[:, :, 2])),
            ))
        cap.release()

        if not hsv_meds:
            return "unknown"

        H = float(np.mean([m[0] for m in hsv_meds]))  # 0–180 in OpenCV
        S = float(np.mean([m[1] for m in hsv_meds]))  # 0–255
        V = float(np.mean([m[2] for m in hsv_meds]))  # 0–255

        if V < 55:
            return "black"
        # White ball: very bright regardless of slight yellowish cast from table reflection.
        # Snooker yellow ball has S > 130; white ball may have S up to ~90 due to green bleed.
        if V > 170 and S < 90:
            return "white"
        if S > 100:
            if H < 10 or H > 165:             # red hue range
                return "red" if S > 150 else "pink"
            if 10 <= H < 25:
                return "orange" if S > 170 else "brown"
            if 25 <= H < 38:
                return "yellow"
            if 38 <= H < 85:
                return "green"
            if 85 <= H < 135:
                return "blue"
            if 135 <= H <= 165:
                return "purple"
        return "unknown"
    except Exception:
        return "unknown"


def _ball_label_from_color(color: str, scene_objects: List[str]) -> str:
    """Map a detected color to the best-matching ball name from scene_objects.

    Searches scene_objects for a phrase containing both a color keyword and
    'ball'.  Falls back to '<color> ball' if no match found.
    """
    _COLOR_KEYS: Dict[str, List[str]] = {
        "white":  ["white", "cue"],
        "red":    ["red"],
        "pink":   ["pink", "rose"],
        "black":  ["black"],
        "yellow": ["yellow"],
        "brown":  ["brown"],
        "blue":   ["blue"],
        "green":  ["green"],
        "orange": ["orange"],
        "purple": ["purple", "violet"],
    }
    keywords = _COLOR_KEYS.get(color, [])
    for kw in keywords:
        for obj in scene_objects:
            ol = obj.lower()
            if kw in ol and "ball" in ol:
                words = obj.strip().split()
                return " ".join(words[-min(4, len(words)):]).strip()
    return f"{color} ball" if color != "unknown" else "sports ball"


def _fit_piecewise_tangent(dets: List) -> Optional[Dict]:
    """Detect the deflection breakpoint in a YOLO track and return tangent angles.

    Uses LOCAL velocity vectors (mean of consecutive-frame differences) around
    the breakpoint rather than global chord direction (first→last position).
    This handles curved trajectories correctly: parabolic flight, rolling arcs, etc.

    Breakpoint = frame of maximum angular velocity (steepest direction change).

    ID-switch indicator: if the apparent speed INCREASES at the breakpoint the
    track likely switched from the projectile to the struck (stationary→moving)
    object, which would have been at rest before the collision.

    Returns dict with keys:
        pre_angle, post_angle, deflection, breakpoint_frame,
        is_suspicious, pre_disp, post_disp
    or None if no significant deflection found.
    """
    import math
    n = len(dets)
    if n < 12:
        return None

    xs = np.array([float(d[1]) for d in dets])
    ys = np.array([float(d[2]) for d in dets])

    # Smooth bbox-center jitter with Savitzky-Golay
    try:
        from scipy.signal import savgol_filter
        win = max(3, min(7, (n // 4) * 2 + 1))
        if win % 2 == 0:
            win -= 1
        xs = savgol_filter(xs, win, 2)
        ys = savgol_filter(ys, win, 2)
    except Exception:
        pass

    # Per-frame velocity in physics coords (x right, y up = flip screen-y)
    vx = np.diff(xs)
    vy = -np.diff(ys)
    speed = np.hypot(vx, vy)

    # Angular velocity: how fast the direction is turning between consecutive frames
    ang = np.unwrap(np.arctan2(vy, vx))
    d_ang = np.abs(np.diff(ang))   # len = n - 2

    # Find breakpoint in the interior (skip edge frames to avoid noisy endpoints)
    margin = max(3, n // 8)
    interior = d_ang[margin:-margin] if len(d_ang) > 2 * margin else d_ang
    if len(interior) == 0:
        return None
    bp = int(np.argmax(interior)) + margin

    if np.degrees(d_ang[bp]) < 15:
        return None  # no significant deflection

    # --- ID-switch check #1: speed increase at breakpoint ---
    # A real collision makes the projectile SLOW DOWN (transfers energy to target).
    # If instead the track appears to speed up, it has likely switched to the
    # struck object (which accelerates from rest) — flag as suspicious.
    K = max(3, n // 8)
    pre_sl  = slice(max(0, bp - K), bp)
    post_sl = slice(bp + 1, min(len(speed), bp + 1 + K))
    pre_speed  = float(np.mean(speed[pre_sl]))  if pre_sl.stop > pre_sl.start  else 0.0
    post_speed = float(np.mean(speed[post_sl])) if post_sl.stop > post_sl.start else 0.0
    is_suspicious = (pre_speed > 0.5 and post_speed > pre_speed * 1.3)

    # Tangent angle = mean velocity direction over the K-frame window
    def _tangent(sl):
        mvx = float(np.mean(vx[sl])) if sl.stop > sl.start else 0.0
        mvy = float(np.mean(vy[sl])) if sl.stop > sl.start else 0.0
        return math.degrees(math.atan2(mvy, mvx))

    pre_ang  = _tangent(pre_sl)
    post_ang = _tangent(post_sl)
    deflection = float((post_ang - pre_ang + 180) % 360 - 180)

    # Displacement proxy: sum of per-frame speeds in each window
    pre_disp  = float(np.sum(speed[pre_sl]))
    post_disp = float(np.sum(speed[post_sl]))

    return {
        "pre_angle":        round(pre_ang, 1),
        "post_angle":       round(post_ang, 1),
        "deflection":       round(deflection, 1),
        "breakpoint_frame": int(dets[min(bp, n - 1)][0]),
        "is_suspicious":    is_suspicious,
        "pre_disp":         round(pre_disp, 1),
        "post_disp":        round(post_disp, 1),
    }


def _b2_extract_track_physics(
    shared_tracks: Dict[int, List],
    fps: float,
    vlm_opq: Dict[str, Any],
    scene_objects: Optional[List[str]] = None,
    clip_path: Optional[str] = None,
) -> List[Dict]:
    """Extract scale-free trajectory physics from YOLO tracks.

    For B-class videos, persons (cls 0) are excluded — only physical objects are tracked.
    scene_objects (VLM captions) are used to label non-person tracks with specific names.

    Outputs per significant track (all general — no category-specific branching):
    - trajectory_angle: initial motion direction relative to detected scene reference line
    - speed_phase_ratio: second-half vs first-half displacement ratio
    - deflection_angle: direction change within track (camera-independent)
    - bounce_height_ratio: consecutive peak height ratio (e² for elastic bounce)
    - oscillation_period_proxy: period estimate from direction reversals
    - bbox_area_change_ratio: bbox size change ratio (growing/shrinking objects)
    After all tracks: multi_object_speed_ratio for 2+ tracked objects.

    All angles are expressed relative to the scene reference direction (PCA
    of stationary objects when available, Hough fallback), NOT raw screen coords.
    """
    import math

    if not shared_tracks or fps <= 0:
        return []

    # Detect scene reference direction: PCA of stationary objects (preferred),
    # fall back to Hough transform on the raw clip frame.
    pca_angle = _pca_axis_from_stationary(shared_tracks)
    if pca_angle is not None:
        ref_angle = pca_angle
        # Translate to a human-readable description based on scene_objects VLM captions
        surface_name = "scene"
        if scene_objects:
            text = " ".join(scene_objects).lower()
            if any(k in text for k in ("snooker", "pool", "billiard", "ping-pong", "table tennis")):
                surface_name = "snooker/pool table"
            elif any(k in text for k in ("court", "basketball", "volleyball", "badminton")):
                surface_name = "court"
            elif any(k in text for k in ("field", "pitch", "grass", "soccer", "football")):
                surface_name = "field"
            elif "table" in text:
                surface_name = "table"
            elif any(k in text for k in ("floor", "mat", "ground")):
                surface_name = "floor"
        ref_name = (
            f"long axis of the {surface_name} "
            f"({pca_angle:.1f}° from image horizontal, "
            f"estimated from positions of stationary objects)"
        )
        logger.debug(f"Reference: {ref_name}")
    elif clip_path:
        scene_ref = _detect_scene_reference_line(clip_path)
        ref_angle = scene_ref["angle_deg"]
        ref_name  = scene_ref["name"]
    else:
        ref_angle = 0.0
        ref_name  = "horizontal (default)"

    motions = []
    # Collect (label, dets) for multi-object ratio computed after main loop
    significant_tracks: List[Tuple[str, List]] = []

    # Group non-person tracks by class.
    # Objects (cls != 0): keep the track with maximum net displacement
    # (most likely the actively moving object, e.g. cue ball vs stationary balls).
    # Person (cls 0) is excluded for B-class — people are not the physics subject.
    # YOLO COCO class labels for physics-relevant objects
    _CLS_LABELS = {
        0: "athlete",
        32: "sports ball",
        36: "skateboard",
        37: "surfboard",
        38: "tennis racket",
        39: "bottle",
        40: "wine glass",
        43: "knife",
        46: "banana",
        47: "apple",
        50: "clock",
        56: "chair",
        57: "couch",
        63: "laptop",
        67: "cell phone",
        73: "book",
        76: "scissors",
    }

    # --- Build per-class track list -----------------------------------------------
    # Ball-like classes (sports ball, frisbee/puck variants) keep the top-3 tracks
    # by net displacement so we can label the cue ball AND the struck ball separately.
    # All other object classes keep only the single most-displaced track.
    # Persons (cls 0) are excluded entirely for B-class.

    _class_tracks_raw: Dict[int, List[Tuple[float, int, List]]] = {}
    for tid, dets in shared_tracks.items():
        if len(dets) < 4:
            continue
        cls = int(dets[len(dets) // 2][5])
        if cls == 0:
            continue
        # Exclude very large bounding boxes (tables, walls, floor areas mis-detected
        # as physics objects).  Threshold 200px covers the largest real subject (a person
        # is excluded by cls==0 above; equipment like rackets may be ~150px on long axis).
        mid = dets[len(dets) // 2]
        w_mid, h_mid = float(mid[3]), float(mid[4])
        if max(w_mid, h_mid) > 200:
            continue
        cx1 = float(dets[0][1]); cy1 = float(dets[0][2])
        cx2 = float(dets[-1][1]); cy2 = float(dets[-1][2])
        net_disp = math.sqrt((cx2 - cx1) ** 2 + (cy2 - cy1) ** 2)
        _class_tracks_raw.setdefault(cls, []).append((net_disp, tid, dets))

    # Flatten to list of (cls_id, tid, dets), sorted by displacement within class
    class_best_flat: List[Tuple[int, int, List]] = []
    for cls, entries in _class_tracks_raw.items():
        entries.sort(key=lambda x: x[0], reverse=True)
        max_keep = _MAX_BALL_TRACKS if cls in _MULTI_TRACK_CLASSES else 1
        for _, tid, dets in entries[:max_keep]:
            # Skip unknown classes that don't pass the ball-like size/shape heuristic:
            # they are likely mis-detections of furniture or body parts.
            is_known_cls = cls in _CLS_LABELS
            mid2 = dets[len(dets) // 2]
            w2, h2 = float(mid2[3]), float(mid2[4])
            aspect2 = w2 / h2 if h2 > 0 else 0
            is_ball_like = 0.55 < aspect2 < 1.4 and max(w2, h2) < 80
            if not is_known_cls and not is_ball_like:
                continue
            class_best_flat.append((cls, tid, dets))

    for cls_id, best_tid, dets in class_best_flat:
        generic  = _CLS_LABELS.get(cls_id, f"object_cls{cls_id}")
        is_ball  = cls_id in _MULTI_TRACK_CLASSES or "ball" in generic.lower()
        # Also treat compact, near-square unknown objects as ball-like
        # (YOLO sometimes assigns unexpected class IDs to snooker/billiard balls).
        # Use tight thresholds to avoid mis-classifying body parts or equipment.
        if not is_ball and dets:
            mid = dets[len(dets) // 2]
            w_px, h_px = float(mid[3]), float(mid[4])
            if w_px > 0 and h_px > 0:
                aspect = w_px / h_px
                if 0.55 < aspect < 1.4 and max(w_px, h_px) < 80:
                    is_ball = True
        if is_ball and clip_path:
            _color = _dominant_ball_color(clip_path, dets)
            label  = _ball_label_from_color(_color, scene_objects or [])
        else:
            label  = _resolve_label_from_scene(cls_id, generic, scene_objects or [])

        n = len(dets)

        # Record this track for later multi-object ratio computation
        # (only tracks with meaningful displacement — stationary objects excluded below)

        # --- Trajectory angle: initial direction relative to scene reference line ---
        # Expressed as "X° from [ref_name]", NOT raw screen coordinates.
        # deflection_angle is camera-independent; trajectory_angle requires ref_name context.
        if n >= 4:
            seg = max(2, n // 5)          # first 20% of track, min 2 frames
            early = dets[:seg]
            cx1 = float(np.mean([d[1] for d in early[:max(1, seg // 2)]]))
            cy1 = float(np.mean([d[2] for d in early[:max(1, seg // 2)]]))
            cx2 = float(np.mean([d[1] for d in early[max(1, seg // 2):]]))
            cy2 = float(np.mean([d[2] for d in early[max(1, seg // 2):]]))
            dx = cx2 - cx1
            dy = -(cy2 - cy1)  # flip: screen y increases down, physics y increases up
            displacement = math.sqrt(dx * dx + dy * dy)
            if displacement > 3:
                # Screen angle (0°=right, 90°=up in physics convention)
                screen_angle = math.degrees(math.atan2(dy, dx))
                # Angle relative to detected scene reference line
                rel_angle = screen_angle - ref_angle
                rel_angle = (rel_angle + 180) % 360 - 180   # normalise to (-180, 180]
                rel_angle = round(rel_angle, 1)
                ang_conf = 0.82 if displacement >= 8 else 0.78
                motions.append({
                    "subject": label,
                    "event": "trajectory_angle",
                    "estimated_angle_deg": rel_angle,
                    "estimated_speed_ms": None,
                    "scene_reference": ref_name,
                    "estimation_basis": (
                        f"YOLO track {best_tid}: initial motion = {rel_angle}° "
                        f"from [{ref_name}] "
                        f"(screen angle {round(screen_angle,1)}°, ref={ref_angle}°; "
                        f"displacement={displacement:.0f}px, first {seg} frames)"
                    ),
                    "estimation_confidence": ang_conf,
                    "estimation_method": "yolo_trajectory",
                })
                # Collect for multi-object ratio if displacement is meaningful
                if displacement >= 8:
                    significant_tracks.append((label, dets))

        # --- Phase speed ratio (≥20 frames) ---
        if n >= 20:
            xs = [float(d[1]) for d in dets]
            mid = n // 2
            disp_first  = [abs(xs[j] - xs[j - 1]) for j in range(1, mid)]
            disp_second = [abs(xs[j] - xs[j - 1]) for j in range(mid + 1, n)]
            if disp_first and disp_second:
                mean1 = float(np.mean(disp_first))
                mean2 = float(np.mean(disp_second))
                if mean1 > 0.5:  # avoid near-zero division
                    ratio = round(mean2 / mean1, 2)
                    motions.append({
                        "subject": label,
                        "event": "speed_phase_ratio",
                        "estimated_ratio": ratio,
                        "estimated_speed_ms": None,
                        "estimation_basis": (
                            f"YOLO track {best_tid}: second-half / first-half mean displacement "
                            f"ratio = {ratio:.2f} (>1 = accelerating, <1 = decelerating)"
                        ),
                        "estimation_confidence": 0.80,
                        "estimation_method": "yolo_trajectory",
                    })

        # --- Deflection angle: piecewise tangent fitting (handles curved trajectories) ---
        # Uses local velocity vectors (mean np.diff) around the breakpoint, NOT chords.
        # Breakpoint = frame of maximum angular velocity.
        # Suspicious if post-breakpoint speed > 1.3× pre-speed (possible ID-switch).
        if n >= 12:
            pw = _fit_piecewise_tangent(dets)
            if pw is not None:
                defl_conf = 0.82 if min(pw["pre_disp"], pw["post_disp"]) >= 15 else 0.80
                basis_note = "piecewise tangent fit; camera-independent relative angle"

                # Secondary ID-switch check: neighboring track appears near breakpoint
                bp_frame = pw["breakpoint_frame"]
                bp_cx = float(dets[min(int(len(dets) * 0.5), n - 1)][1])
                bp_cy = float(dets[min(int(len(dets) * 0.5), n - 1)][2])
                for other_tid, other_dets in shared_tracks.items():
                    if other_tid == best_tid or len(other_dets) < 4:
                        continue
                    if int(other_dets[len(other_dets) // 2][5]) == 0:
                        continue  # skip persons
                    other_first = int(other_dets[0][0])
                    if abs(other_first - bp_frame) <= 8:
                        ox = float(other_dets[0][1])
                        oy = float(other_dets[0][2])
                        if math.sqrt((ox - bp_cx)**2 + (oy - bp_cy)**2) < 80:
                            pw["is_suspicious"] = True
                            break

                if pw["is_suspicious"]:
                    defl_conf = max(0.30, defl_conf - 0.5)
                    basis_note = (
                        "CAUTION: possible_id_switch — post-breakpoint speed increased or "
                        "another object appeared at collision point; pre/post may be "
                        "different objects (e.g. struck ball vs projectile)"
                    )
                motions.append({
                    "subject": label,
                    "event": "deflection_angle",
                    "estimated_angle_deg": pw["deflection"],
                    "estimated_speed_ms": None,
                    "estimation_basis": (
                        f"YOLO track {best_tid}: piecewise tangent fitting — "
                        f"pre={pw['pre_angle']}° → post={pw['post_angle']}°, "
                        f"deflection={pw['deflection']}°, "
                        f"breakpoint at frame {bp_frame}; "
                        f"pre_disp={pw['pre_disp']:.0f}px, post_disp={pw['post_disp']:.0f}px; "
                        f"{basis_note}"
                    ),
                    "estimation_confidence": defl_conf,
                    "estimation_method": "yolo_trajectory",
                })

        # --- Bounce height ratio: e² for elastic bounce (general) ---
        # Detects successive floor contacts and apex heights from y-coordinate.
        # Works for any bouncing object; gives coefficient of restitution e = sqrt(ratio).
        if n >= 30:
            try:
                from scipy.signal import savgol_filter, find_peaks
                y_raw = np.array([float(d[2]) for d in dets])   # screen y (down=larger)
                win = min(11, (n // 4) * 2 + 1)
                if win < 3:
                    win = 3
                if win % 2 == 0:
                    win -= 1
                y_sm = savgol_filter(y_raw, win, 2)

                # Floor contacts = local maxima in screen y (lowest physical points)
                contacts, _ = find_peaks(y_sm, prominence=10, distance=max(4, n // 8))
                # Apex points = local minima in screen y (highest physical points)
                apexes, _   = find_peaks(-y_sm, prominence=10, distance=max(4, n // 8))

                if len(contacts) >= 2 and len(apexes) >= 1:
                    heights = []
                    for i in range(len(contacts) - 1):
                        c1, c2 = contacts[i], contacts[i + 1]
                        mid_ap = [a for a in apexes if c1 < a < c2]
                        if not mid_ap:
                            continue
                        floor_y = float((y_sm[c1] + y_sm[c2]) / 2)
                        apex_y  = float(y_sm[mid_ap[0]])
                        h = floor_y - apex_y    # positive: object rose above floor
                        if h > 8:               # minimum 8px height to suppress noise
                            heights.append(h)

                    if len(heights) >= 2:
                        bh_ratio = round(heights[1] / heights[0], 3)
                        e_est    = round(math.sqrt(max(0, bh_ratio)), 3)
                        bh_conf  = 0.82 if len(heights) >= 3 else 0.80
                        motions.append({
                            "subject": label,
                            "event": "bounce_height_ratio",
                            "estimated_ratio": bh_ratio,
                            "estimated_angle_deg": None,
                            "estimated_speed_ms": None,
                            "bounce_count": len(contacts),
                            "restitution_estimate": e_est,
                            "estimation_basis": (
                                f"YOLO track {best_tid}: bounce height ratio = "
                                f"{heights[1]:.0f}px / {heights[0]:.0f}px = {bh_ratio:.3f} "
                                f"({len(contacts)} contacts detected; "
                                f"e ≈ √{bh_ratio:.3f} = {e_est:.3f}; "
                                f"heights in screen pixels, ratio is scale-free)"
                            ),
                            "estimation_confidence": bh_conf,
                            "estimation_method": "yolo_trajectory",
                        })
            except Exception:
                pass

        # --- Oscillation period proxy: period estimate from direction reversals (general) ---
        # Counts reversals in x or y coordinate. Works for spring, pendulum, circular motion.
        # Period in frames = clip_frames / (reversals / 2). Period in seconds = / fps.
        if n >= 30:
            try:
                from scipy.signal import savgol_filter, find_peaks
                win = min(11, (n // 4) * 2 + 1)
                if win < 3: win = 3
                if win % 2 == 0: win -= 1

                y_raw = np.array([float(d[2]) for d in dets])
                x_raw = np.array([float(d[1]) for d in dets])
                y_sm  = savgol_filter(y_raw, win, 2)
                x_sm  = savgol_filter(x_raw, win, 2)

                min_dist = max(3, n // 12)
                y_peaks, _   = find_peaks( y_sm, prominence=8, distance=min_dist)
                y_valleys, _ = find_peaks(-y_sm, prominence=8, distance=min_dist)
                x_peaks, _   = find_peaks( x_sm, prominence=8, distance=min_dist)
                x_valleys, _ = find_peaks(-x_sm, prominence=8, distance=min_dist)

                y_rev = len(y_peaks) + len(y_valleys)
                x_rev = len(x_peaks) + len(x_valleys)
                best_rev = max(y_rev, x_rev)
                primary_axis = "vertical" if y_rev >= x_rev else "horizontal"

                if best_rev >= 4:   # require ≥2 full cycles
                    period_frames = round(n / (best_rev / 2), 1)
                    period_s      = round(period_frames / fps, 3) if fps > 0 else None
                    freq_hz       = round(1.0 / period_s, 3) if period_s else None
                    osc_conf      = 0.82 if best_rev >= 6 else 0.80
                    motions.append({
                        "subject": label,
                        "event": "oscillation_period_proxy",
                        "estimated_angle_deg": None,
                        "estimated_speed_ms": None,
                        "estimated_ratio": period_frames,
                        "oscillation_reversal_count": best_rev,
                        "period_frames": period_frames,
                        "period_seconds": period_s,
                        "frequency_hz": freq_hz,
                        "primary_axis": primary_axis,
                        "estimation_basis": (
                            f"YOLO track {best_tid}: {best_rev} direction reversals along "
                            f"{primary_axis} axis → period ≈ {period_frames} frames "
                            f"({period_s}s, {freq_hz}Hz at {fps}fps; "
                            f"period_seconds = period_frames / fps — no metric calibration needed)"
                        ),
                        "estimation_confidence": osc_conf,
                        "estimation_method": "yolo_trajectory",
                    })
            except Exception:
                pass

        # --- Bbox area change ratio: growing or shrinking objects (general) ---
        # Captures melting (decreasing), expanding explosion fragments (increasing),
        # objects approaching/receding camera. Ratio = area(2nd half) / area(1st half).
        if n >= 20:
            areas = [float(d[3]) * float(d[4]) for d in dets]
            mid   = n // 2
            mean1 = float(np.mean(areas[:mid]))
            mean2 = float(np.mean(areas[mid:]))
            if mean1 >= 100:    # bbox must be large enough (≥10×10px) to be reliable
                area_ratio = round(mean2 / mean1, 3)
                if area_ratio < 0.85 or area_ratio > 1.15:  # only emit if notable change
                    direction = "growing" if area_ratio > 1.0 else "shrinking"
                    motions.append({
                        "subject": label,
                        "event": "bbox_area_change_ratio",
                        "estimated_ratio": area_ratio,
                        "estimated_angle_deg": None,
                        "estimated_speed_ms": None,
                        "estimation_basis": (
                            f"YOLO track {best_tid}: bbox area ratio = "
                            f"{mean2:.0f} / {mean1:.0f} = {area_ratio:.3f} "
                            f"(object is {direction}; "
                            f"useful for phase-change, expansion, approach)"
                        ),
                        "estimation_confidence": 0.78,
                        "estimation_method": "yolo_trajectory",
                    })

    # --- Multi-object speed ratio: displacement ratio between 2 tracked objects ---
    # Applies to any video with 2+ significant moving objects (pulley, collision, etc.).
    # Ratio = |total_displacement_A| / |total_displacement_B|.
    if len(significant_tracks) >= 2:
        for i in range(len(significant_tracks)):
            for j in range(i + 1, len(significant_tracks)):
                label_a, dets_a = significant_tracks[i]
                label_b, dets_b = significant_tracks[j]

                def _total_disp(det_list):
                    xs = [float(d[1]) for d in det_list]
                    ys = [float(d[2]) for d in det_list]
                    ddx = xs[-1] - xs[0]
                    ddy = ys[-1] - ys[0]
                    return math.sqrt(ddx * ddx + ddy * ddy)

                disp_a = _total_disp(dets_a)
                disp_b = _total_disp(dets_b)
                if min(disp_a, disp_b) < 5:
                    continue
                mo_ratio = round(disp_a / disp_b, 3) if disp_b > 0 else None
                if mo_ratio is None:
                    continue
                motions.append({
                    "subject": f"{label_a} vs {label_b}",
                    "event": "multi_object_speed_ratio",
                    "estimated_ratio": mo_ratio,
                    "estimated_angle_deg": None,
                    "estimated_speed_ms": None,
                    "estimation_basis": (
                        f"Displacement ratio: {label_a} ({disp_a:.0f}px) / "
                        f"{label_b} ({disp_b:.0f}px) = {mo_ratio:.3f} "
                        f"(useful for pulley MA, coupled-motion systems, "
                        f"relative approach/recession)"
                    ),
                    "estimation_confidence": 0.78,
                    "estimation_method": "yolo_trajectory",
                })

    return motions


def _b2_extract_size_ratios(
    shared_tracks: Dict[int, List],
) -> List[Dict]:
    """Compute dimensionless bounding-box area ratios between tracked object classes.

    Prefers ball-to-ball or ball-to-equipment ratios over ball-to-person.
    Ball/person ratio is kept as fallback when no other pairing is available.
    """
    if not shared_tracks:
        return []

    class_areas: Dict[int, float] = {}
    for tid, dets in shared_tracks.items():
        if len(dets) < 5:
            continue
        cls = int(dets[len(dets) // 2][5])
        areas = [float(d[3]) * float(d[4]) for d in dets]
        # Keep the largest-area track per class (most prominent instance)
        med = float(np.median(areas))
        if cls not in class_areas or med > class_areas[cls]:
            class_areas[cls] = med

    subjects = []

    # --- Ball-to-ball ratio (most physically meaningful for ball sports) ---
    # YOLO sports_ball = cls 32; use two distinct tracks of the same class
    ball_tracks = [(tid, dets) for tid, dets in shared_tracks.items()
                   if len(dets) >= 5 and int(dets[len(dets) // 2][5]) == 32]
    if len(ball_tracks) >= 2:
        areas_32 = sorted(
            [float(np.median([float(d[3]) * float(d[4]) for d in dets]))
             for _, dets in ball_tracks],
            reverse=True,
        )
        if areas_32[1] > 0:
            ratio = round(areas_32[0] / areas_32[1], 4)
            subjects.append({
                "label": "sports ball",
                "estimated_height_m": None,
                "estimated_ratio": ratio,
                "ratio_description": "largest_ball_bbox_area / second_ball_bbox_area (perspective-only, NOT physical size)",
                "estimation_basis": (
                    f"YOLO: largest sports-ball bbox area / second-largest = {ratio:.4f}. "
                    f"WARNING: For sports with regulation same-size balls (snooker, pool, billiards, "
                    f"tennis, etc.), this ratio reflects PERSPECTIVE/DEPTH only, NOT physical size or "
                    f"mass difference. Do NOT use as mass ratio."
                ),
                "estimation_confidence": 0.55,  # below 0.80 threshold — perspective artifact for same-size sports
                "estimation_method": "yolo_trajectory",
            })
            return subjects  # ball-to-ball extracted but below confidence threshold; skip person-based

    # --- Fallback: ball-to-person ratio (when only one ball class detected) ---
    if 0 in class_areas and 32 in class_areas and class_areas[0] > 0:
        ratio = round(class_areas[32] / class_areas[0], 4)
        subjects.append({
            "label": "sports ball",
            "estimated_height_m": None,
            "estimated_ratio": ratio,
            "ratio_description": "sports_ball_bbox_area / person_bbox_area (pixel units, dimensionless)",
            "estimation_basis": (
                f"YOLO: sports ball bounding box area / person bounding box area = {ratio:.4f} "
                f"(with known ball diameter from web-search, person height can be cross-checked)"
            ),
            "estimation_confidence": 0.75,
            "estimation_method": "yolo_trajectory",
        })
    return subjects


def _extract_b2_scalefree(
    clip_path: str,
    fps: float,
    vlm_opq: Dict[str, Any],
    shared_tracks: Dict[int, List],
    web_search_specs: Optional[Dict] = None,
    scene_objects: Optional[List[str]] = None,
    measurement_frame_indices: Optional[List[int]] = None,
) -> Dict[str, Any]:
    """Extract scale-free physical measurements for B2 videos.

    Produces OPQ-compatible entries with:
      estimated_angle_deg  — joint angles (pose) and trajectory angles (YOLO)
      estimated_duration_s — event durations from YOLO track lengths
      estimated_ratio      — dimensionless size / speed ratios

    No pixel-to-metre scale conversion is used or required.
    """
    new_subjects: List[Dict] = []
    new_motions:  List[Dict] = []

    # B-class: skip person pose angles — only physical objects matter
    # (pose angles are kept for A-class via route_a_pose_estimation)

    try:
        new_motions.extend(
            _b2_extract_track_physics(
                shared_tracks, fps, vlm_opq,
                scene_objects=scene_objects, clip_path=clip_path,
            )
        )
    except Exception as exc:
        logger.warning(f"B2 track physics failed: {exc}")

    try:
        new_subjects.extend(_b2_extract_size_ratios(shared_tracks))
    except Exception as exc:
        logger.warning(f"B2 size ratios failed: {exc}")

    merged = dict(vlm_opq)
    merged["subjects"] = merged.get("subjects", []) + new_subjects
    merged["motions"]  = merged.get("motions",  []) + new_motions
    merged["calibration"] = {
        "scale_px_per_m":        None,
        "calibration_confidence": 0.0,
        "calibration_source":    "b2_scalefree",
        "is_anchor_based":       False,
    }
    merged["web_search_specs"] = web_search_specs or {}
    logger.info(
        f"B2 scale-free extraction: "
        f"{len(new_subjects)} subjects, {len(new_motions)} motions"
    )
    return merged


def augment_opq(
    clip_path: str,
    fps: float,
    vlm_opq: Dict[str, Any],
    sport_category: str = "",
    web_search_specs: Optional[Dict[str, Dict]] = None,
    video_class: str = "B",
    scene_objects: Optional[List[str]] = None,
    gt_segments: Optional[Dict[str, float]] = None,
    measurement_frame_indices: Optional[List[int]] = None,
    object_frame_indices: Optional[List[int]] = None,
    clip_summary: str = "",
) -> Dict[str, Any]:
    """
    Augment VLM observable_physical_quantities with CV-based estimates.

    Args:
        clip_path:        path to the video clip
        fps:              video frames-per-second
        vlm_opq:          raw OPQ dict from VLM (may be {})
        sport_category:   sport name for calibration priors (e.g. "hurdling")
        web_search_specs: {label: {"h_m": float|None, "w_m": float|None,
                                   "mass_kg": float|None, "source_type": str,
                                   "confidence": float}}
                          Pre-fetched standard specs for B1 class; injected as
                          Priority 0 reference objects in calibrate_scale().
        video_class:      "A" | "B"  — B suppresses metric extraction.
        scene_objects:    Non-person object labels from Stage 1 visual_elements
                          (parsed by _parse_visual_elements). Each object is
                          localised via VLM bbox and tracked via YOLO for speed.
                          Results are appended to subjects/motions.

    Returns:
        Augmented OPQ dict with estimation_confidence, estimation_method,
        calibration, and uncertain_quantities fields.
    """
    if vlm_opq is None:
        vlm_opq = {}

    # Get frame dimensions
    cap = cv2.VideoCapture(clip_path)
    frame_h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    cap.release()

    # --- Single shared YOLO tracking pass (used by calibration, Route B, and camera-pan) ---
    # B-class uses VLM physics extraction — skip YOLO entirely.
    if video_class != "B":
        shared_tracks = _run_yolo_tracks(clip_path)
        logger.info(f"Shared YOLO tracks: {len(shared_tracks)} tracks")
    else:
        shared_tracks = {}

    # --- Detect reference objects using sampled frames (YOLO11n + YOLOWorld) ---
    if video_class != "B":
        ref_detections = _detect_ref_objects(clip_path, sport_category)
        logger.info(f"Ref detections: {[(d['label'], d['height_px'], d['known_h_m']) for d in ref_detections]}")
    else:
        ref_detections = []

    # --- B-class: VLM-only physics extraction (no CV, no YOLO) ---
    if video_class == "B":
        vlm_result = _vlm_extract_b_physics(
            clip_path,
            clip_summary=clip_summary,
            scene_objects=scene_objects,
            sport_category=sport_category,
            n_frames=5,
        )
        return {
            "subjects": vlm_result.get("subjects", []),
            "motions": vlm_result.get("motions", []),
            "calibration": {
                "scale_px_per_m": None,
                "calibration_confidence": 0.0,
                "calibration_source": "b2_scalefree",
                "is_anchor_based": False,
            },
            "web_search_specs": web_search_specs or {},
        }

    # --- A-class: inject web_search standard specs into ref_detections (Priority 0) ---
    # web_search_specs are pre-fetched by extract_video_info.py before calling augment_opq.
    # They provide known_h_m / known_w_m for scene-specific objects (e.g. baseball bat,
    # bowling ball) that are not in the static calibration tables.
    if web_search_specs and video_class == "A":
        for spec_label, spec in web_search_specs.items():
            h_m  = spec.get("h_m")
            w_m  = spec.get("w_m")
            conf = float(spec.get("confidence", 0.75))
            if h_m is None and w_m is None:
                continue
            matched = False
            for det in ref_detections:
                if _label_matches(spec_label, det["label"]):
                    if h_m is not None and det.get("known_h_m") is None:
                        det["known_h_m"] = h_m
                        det["confidence"] = max(det.get("confidence", 0.0), conf)
                        det["source"] = "web_search"
                        logger.info(f"Web-search spec injected → '{det['label']}' h={h_m}m (conf={conf})")
                    if w_m is not None and det.get("known_w_m") is None:
                        det["known_w_m"] = w_m
                        det["source"] = "web_search"
                    matched = True
                    break
            if not matched:
                # Anchor label not found by YOLO — try VLM bbox localisation
                logger.info(f"Anchor '{spec_label}' not in YOLO detections — trying VLM bbox")
                vlm_bbox = _vlm_locate_bbox(clip_path, spec_label, frame_indices=obj_set)
                if vlm_bbox and (vlm_bbox.get("height_px", 0) > 5 or vlm_bbox.get("width_px", 0) > 5):
                    synthetic = {
                        "label":     spec_label,
                        "height_px": vlm_bbox.get("height_px", 0),
                        "width_px":  vlm_bbox.get("width_px", 0),
                        "known_h_m": h_m,
                        "known_w_m": w_m,
                        "confidence": min(conf, vlm_bbox["confidence"]),
                        "source": "web_search+vlm_bbox",
                    }
                    ref_detections.append(synthetic)
                    logger.info(
                        f"VLM bbox fallback → '{spec_label}' "
                        f"h_px={vlm_bbox.get('height_px',0):.1f} w_px={vlm_bbox.get('width_px',0):.1f} "
                        f"known_h={h_m}m known_w={w_m}m conf={synthetic['confidence']:.2f}"
                    )
                else:
                    logger.info(f"VLM bbox failed for '{spec_label}' — anchor dropped (vlm_bbox={vlm_bbox})")
        logger.info(f"After web-search injection: {[(d['label'], d['height_px'], d.get('known_h_m'), d.get('known_w_m')) for d in ref_detections if d.get('known_h_m') or d.get('known_w_m')]}")

    # VLM knowledge-based size estimation is intentionally NOT used here.
    # A-class calibration must use GT 3D joints and web_search specs.
    # Falling back to VLM knowledge (equivalent to a model's internal database) would
    # violate the anchor requirement and produce measurements that cannot be verified.

    # Build yolo_bbox_heights dict for calibration Priority 1 compatibility
    # Prefer tracks data (full-video median) over sampled-frame detections
    yolo_bbox_heights: Dict[str, float] = {}
    for det in ref_detections:
        yolo_bbox_heights[det["label"]] = det["height_px"]
    # Override person height from shared_tracks for better accuracy (full-video median)
    person_heights = []
    for tid, dets in shared_tracks.items():
        if len(dets) < 3:
            continue
        mid_det = dets[len(dets) // 2]
        if mid_det[5] == 0:  # COCO class 0 = person
            person_heights.extend([d[4] for d in dets])  # bh values
    if person_heights:
        yolo_bbox_heights["person"] = float(np.median(person_heights))

    # --- Route A first pass (pose detection + optional GT segment calibration) ---
    # Pass gt_segments so GT-based scale is computed in the same pose-detection pass.
    upright_set = set(measurement_frame_indices) if measurement_frame_indices else None
    obj_set     = set(object_frame_indices) if object_frame_indices else None

    a_first = route_a_pose_estimation(
        clip_path,
        {"scale_px_per_m": None, "calibration_confidence": 0.0},
        gt_segments=gt_segments,
        gt_upright_frames=upright_set,
    )
    pose_height_px = a_first.pop("pose_height_px", None)
    gt_segment_cal = a_first.pop("gt_segment_calibration", None)

    # --- Calibration ---
    # A-class: prefer GT segment-based scale (sensor_gt, highest reliability).
    # B1-class / fallback: use calibrate_scale() with web_search refs + person anchor.
    if gt_segment_cal:
        calibration = gt_segment_cal
        logger.info(
            f"Calibration (GT segments): scale={calibration['scale_px_per_m']} px/m "
            f"segs={calibration.get('gt_segments_used')}  conf={calibration['calibration_confidence']}"
        )
    else:
        calibration = calibrate_scale(
            clip_path,
            sport_category,
            vlm_opq.get("subjects", []),
            pose_height_px=pose_height_px,
            frame_h=frame_h,
            yolo_bbox_heights=yolo_bbox_heights,
            ref_detections=ref_detections,
        )
        logger.info(
            f"Calibration: {calibration['calibration_source']} "
            f"scale={calibration['scale_px_per_m']} px/m "
            f"conf={calibration['calibration_confidence']}"
        )

    # --- Route B — reuse shared tracks, no second YOLO pass ---
    b_result = route_b_yolo_trajectory(
        clip_path, fps, calibration, sport_category,
        vlm_subjects=vlm_opq.get("subjects", []),
        precomputed_tracks=shared_tracks,
    )
    b_result.pop("_background_tracks", None)  # no longer needed; using shared_tracks

    # Background tracks = non-person tracks for camera-pan correction
    background_tracks = {
        tid: dets for tid, dets in shared_tracks.items()
        if len(dets) >= 4 and dets[len(dets) // 2][5] != 0
    }

    # --- Route A second pass with calibration + camera-pan correction ---
    # a_first still contains the first-pass subjects/motions (no scale) — discard them.
    a_result: Dict[str, Any] = {}
    if calibration.get("scale_px_per_m"):
        a_result = route_a_pose_estimation(
            clip_path, calibration,
            background_tracks=background_tracks,
            gt_upright_frames=upright_set,
        )
        a_result.pop("pose_height_px", None)
        a_result.pop("gt_segment_calibration", None)

    # --- Merge ---
    merged = route_c_merge(vlm_opq, a_result, b_result, calibration)

    # --- VLM cross-validation of CV quantities ---
    validation_index = _vlm_validate_cv_quantities(clip_path, merged, vlm_opq)
    if validation_index:
        logger.info(f"VLM validation: {len(validation_index)} quantity entries validated")
        _apply_vlm_validation(merged, validation_index)
    else:
        logger.info("VLM validation skipped or failed — using raw CV estimates")

    # --- Anchor constraint: A-class must suppress metric quantities if no anchor ---
    # scale_provenance values that count as "anchor-based":
    #   independent_ref, vlm_person_prior, sensor_gt, computed_from_gt
    # Values that do NOT count: population_prior, frame_heuristic, none
    if video_class == "A" and not calibration.get("is_anchor_based", False):
        metric_keys = ("estimated_height_m", "estimated_speed_ms", "estimated_mass_kg",
                       "estimated_length_m")
        for section in ("subjects", "motions"):
            for item in merged.get(section, []):
                for k in metric_keys:
                    if k in item:
                        item[k] = None
        # Flag in calibration so downstream can see why values are null
        merged.setdefault("calibration", {})
        merged["calibration"]["metric_suppressed"] = True
        merged["calibration"]["suppression_reason"] = (
            f"no_anchor: scale_provenance={calibration.get('scale_provenance','none')}"
        )
        merged.setdefault("uncertain_quantities", []).append({
            "quantity": "all_metric_quantities",
            "reason": "A/B1 class requires anchor-based calibration; "
                      f"only {calibration.get('scale_provenance','none')} available",
            "no_anchor_metric_suppressed": True,
        })
        logger.info(
            f"Anchor constraint: metric quantities suppressed "
            f"(scale_provenance={calibration.get('scale_provenance','none')})"
        )

    # --- Scene object physics (Stage 1 visual_elements) ---
    if scene_objects:
        logger.info(f"Scene objects to extract: {scene_objects}")
        for obj_label in scene_objects:
            if _is_person_label(obj_label):
                continue
            try:
                obj_result = _extract_object_physics(
                    clip_path, obj_label, calibration, shared_tracks, fps,
                    frame_indices=obj_set,
                )
                if obj_result is None:
                    logger.info(f"  Scene obj '{obj_label}': not visible in clip")
                    continue
                merged.setdefault("subjects", []).append(obj_result["subject"])
                if obj_result.get("motion") is not None:
                    merged.setdefault("motions", []).append(obj_result["motion"])
                logger.info(
                    f"  Scene obj '{obj_label}': "
                    f"h={obj_result['subject'].get('estimated_height_m')}m "
                    f"w={obj_result['subject'].get('estimated_width_m')}m "
                    + (f"speed={obj_result['motion']['estimated_speed_ms']}m/s"
                       if obj_result.get('motion') else "speed=n/a")
                )
            except Exception as e_obj:
                logger.warning(f"  Scene obj physics '{obj_label}': {e_obj}")

    return merged


# ===========================================================================
# Stage 2.5 CV — batch extraction with track_id + bbox_region
# ===========================================================================

# YOLO COCO class names (subset used for track_id matching)
_YOLO_CLS_NAMES: Dict[int, str] = {
    0: "person", 1: "bicycle", 2: "car", 3: "motorcycle",
    24: "backpack", 25: "umbrella", 32: "sports ball", 36: "skateboard",
    38: "tennis racket", 39: "bottle", 40: "wine glass", 41: "cup",
    56: "chair", 57: "couch", 58: "potted plant", 59: "bed",
    60: "dining table", 62: "tv", 63: "laptop", 67: "cell phone",
}


# ===========================================================================
# Stage 2.5 CV extraction entry point
# ===========================================================================

def _is_upright_from_gt_joints(joints_3d: Dict[str, Any], threshold_deg: float = 20.0) -> bool:
    """Return True if person is standing upright based on EgoExo4D GT 3D joints.

    In EgoExo4D the Z axis points up.  The spine vector (hip_mid → shoulder_mid)
    must be within threshold_deg of vertical for the frame to count as upright.
    joints_3d format: {joint_name: [x, y, z]}  (list, not dict).
    """
    import math

    def _xyz(name: str):
        j = joints_3d.get(name)
        return list(j[:3]) if j and len(j) >= 3 else None

    ls, rs = _xyz("left-shoulder"),  _xyz("right-shoulder")
    lh, rh = _xyz("left-hip"),       _xyz("right-hip")
    if not all([ls, rs, lh, rh]):
        return False

    sh = [(ls[i] + rs[i]) / 2 for i in range(3)]  # shoulder midpoint
    hp = [(lh[i] + rh[i]) / 2 for i in range(3)]  # hip midpoint
    spine = [sh[i] - hp[i] for i in range(3)]      # hip→shoulder
    L = math.sqrt(sum(s ** 2 for s in spine))
    if L < 0.1:
        return False
    return spine[2] / L > math.cos(math.radians(threshold_deg))


def _gt_height_from_3d_joints(joints_per_frame_sample: Dict[str, Any]) -> Optional[float]:
    """Compute body height in metres from EgoExo4D 3D joint world coordinates.

    Uses segmental method (ankle→knee→hip→shoulder→nose) on both sides,
    averaged across all sampled frames.  Returns None if joints insufficient.
    """
    import math

    def dist3(a: list, b: list) -> float:
        return math.sqrt(sum((ai - bi) ** 2 for ai, bi in zip(a, b)))

    LEFT_CHAIN  = [("left-ankle", "left-knee"), ("left-knee", "left-hip"),
                   ("left-hip", "left-shoulder"), ("left-shoulder", "nose")]
    RIGHT_CHAIN = [("right-ankle", "right-knee"), ("right-knee", "right-hip"),
                   ("right-hip", "right-shoulder"), ("right-shoulder", "nose")]

    heights: List[float] = []
    for _fid, frame_data in joints_per_frame_sample.items():
        joints = frame_data.get("joints", {})
        for chain in (LEFT_CHAIN, RIGHT_CHAIN):
            try:
                h = sum(dist3(joints[a], joints[b]) for a, b in chain)
                if 1.0 < h < 2.5:
                    heights.append(h)
            except (KeyError, TypeError, ValueError):
                pass

    if not heights:
        return None
    return round(float(sum(heights) / len(heights)), 3)


# Map GT joint names (hyphen) → MediaPipe landmark keys (underscore)
_GT_TO_MP: Dict[str, str] = {
    "left-shoulder": "left_shoulder",   "right-shoulder": "right_shoulder",
    "left-elbow":    "left_elbow",       "right-elbow":    "right_elbow",
    "left-wrist":    "left_wrist",       "right-wrist":    "right_wrist",
    "left-hip":      "left_hip",         "right-hip":      "right_hip",
    "left-knee":     "left_knee",        "right-knee":     "right_knee",
    "left-ankle":    "left_ankle",       "right-ankle":    "right_ankle",
}

# Body segments used for GT-based scale calibration.
# Upper-limb segments (forearms + upper arms) are preferred: they are visible across
# most EgoExo4D activities, shorter (less perspective distortion), and less likely to
# be occluded by equipment.  Lower-limb segments are omitted by default — they are
# longer and often blocked by bikes, benches, or other apparatus.
# Format: {segment_key: (GT_joint_A, GT_joint_B, MP_key_A, MP_key_B)}
_GT_SEGMENTS_DEF: Dict[str, Tuple[str, str, str, str]] = {
    "left_forearm":    ("left-elbow",     "left-wrist",     "left_elbow",     "left_wrist"),
    "right_forearm":   ("right-elbow",    "right-wrist",    "right_elbow",    "right_wrist"),
    "left_upper_arm":  ("left-shoulder",  "left-elbow",     "left_shoulder",  "left_elbow"),
    "right_upper_arm": ("right-shoulder", "right-elbow",    "right_shoulder", "right_elbow"),
    # Lower-limb segments — only add back when upper-limb coverage is insufficient:
    # "left_shin":   ("left-knee",  "left-ankle",  "left_knee",  "left_ankle"),
    # "right_shin":  ("right-knee", "right-ankle", "right_knee", "right_ankle"),
    # "left_thigh":  ("left-hip",   "left-knee",   "left_hip",   "left_knee"),
    # "right_thigh": ("right-hip",  "right-knee",  "right_hip",  "right_knee"),
}


def _compute_gt_segment_lengths(
    joints_per_frame_sample: Dict[str, Any],
) -> Dict[str, float]:
    """Compute GT 3D segment lengths (metres) from EgoExo4D joint coordinates.

    Averages over all sampled frames for robustness.
    Returns {segment_key: length_m} for segments with valid data.
    """
    seg_vals: Dict[str, List[float]] = {k: [] for k in _GT_SEGMENTS_DEF}

    for _fid, frame_data in joints_per_frame_sample.items():
        j = frame_data.get("joints", {})
        for seg_key, (gt_a, gt_b, _mp_a, _mp_b) in _GT_SEGMENTS_DEF.items():
            if gt_a in j and gt_b in j:
                try:
                    d = math.sqrt(sum((ai - bi) ** 2 for ai, bi in zip(j[gt_a], j[gt_b])))
                    if 0.05 < d < 0.80:
                        seg_vals[seg_key].append(d)
                except (TypeError, ValueError):
                    pass

    return {k: round(float(sum(v) / len(v)), 4)
            for k, v in seg_vals.items() if v}


def _gt_segment_scale_from_lm(
    lm_frames: List,
    fw: int,
    fh: int,
    gt_segments: Dict[str, float],
    min_vis: float = 0.55,
) -> Optional[Dict[str, Any]]:
    """Compute px/m scale by matching GT 3D segment lengths to 2D pose landmarks.

    For each (frame, segment) pair where both endpoints are visible, computes:
        scale_px_per_m = pixel_distance / gt_3d_length_m

    Uses robust median + outlier rejection across all valid measurements.
    Returns a calibration dict (compatible with calibrate_scale() output) or None.
    """
    if not lm_frames or fw == 0 or fh == 0 or not gt_segments:
        return None

    scale_samples: List[Tuple[float, str]] = []

    for lm in lm_frames:
        if lm is None:
            continue
        for seg_key, gt_len in gt_segments.items():
            if gt_len <= 0:
                continue
            seg_def = _GT_SEGMENTS_DEF.get(seg_key)
            if seg_def is None:
                continue
            _gt_a, _gt_b, mp_a, mp_b = seg_def
            la = _get_lm(lm, mp_a)
            lb = _get_lm(lm, mp_b)
            if la is None or lb is None:
                continue
            if la[2] < min_vis or lb[2] < min_vis:
                continue
            dx_px = (la[0] - lb[0]) * fw
            dy_px = (la[1] - lb[1]) * fh
            px_dist = math.sqrt(dx_px ** 2 + dy_px ** 2)
            if px_dist < 8:
                continue  # likely depth-foreshortened or occluded
            scale_samples.append((px_dist / gt_len, seg_key))

    if len(scale_samples) < 5:
        return None

    scales = [s for s, _ in scale_samples]
    median_s = float(np.median(scales))

    # Reject outliers: keep within 60% of the median (handles foreshortened segments)
    filtered = [(s, name) for s, name in scale_samples
                if 0.40 * median_s < s < 1.60 * median_s]
    if len(filtered) < 3:
        return None

    final_scales = [s for s, _ in filtered]
    final_scale = float(np.median(final_scales))
    n = len(final_scales)
    cv = float(np.std(final_scales)) / (final_scale + 1e-9)
    # Confidence: improves with count (cap at 30 samples) and falls with variability
    conf = round(min(0.92, max(0.62, 0.72 + min(n, 30) / 150.0 - cv * 0.4)), 3)

    segs_used = sorted(set(name for _, name in filtered))
    logger.info(
        f"GT segment scale: {final_scale:.1f} px/m  "
        f"n={n} samples  cv={cv:.2f}  segs={segs_used}  conf={conf}"
    )
    return {
        "scale_px_per_m": round(final_scale, 1),
        "calibration_source": "ref_object_person_gt_segments",
        "calibration_confidence": conf,
        "reference_object": "person_gt_segments",
        "reference_height_m": None,
        "scale_provenance": "independent_ref",
        "source_type": "sensor_gt",
        "reliability_tier": "A",
        "is_anchor_based": True,
        "gt_segments_used": segs_used,
        "gt_measurements_count": n,
    }



def _yolo_quality_select_frames(
    clip_path: str, n_sample: int = 10, top_k: int = 5
) -> List[int]:
    """Sample n_sample frames from clip, score by YOLO person detection quality,
    return top_k clip-relative frame indices.

    Score = max(person_bbox_area_fraction * detection_confidence) per frame.
    Returns [] if YOLO unavailable (caller falls back to uniform sampling).
    """
    if not _YOLO_OK:
        return []
    try:
        cap = cv2.VideoCapture(clip_path)
        total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT)) or 1
        fh = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        fw = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        indices = [int(total * i / max(1, n_sample - 1)) for i in range(n_sample)]
        indices = sorted(set(min(max(0, idx), total - 1) for idx in indices))

        model = _get_yolo()
        scores: List[tuple] = []
        for idx in indices:
            cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
            ret, frame = cap.read()
            if not ret:
                continue
            results = model(frame, verbose=False, classes=[0])  # person only
            best = 0.0
            for r in results:
                for box in r.boxes:
                    conf = float(box.conf[0])
                    x1, y1, x2, y2 = box.xyxy[0].tolist()
                    area = (x2 - x1) * (y2 - y1) / max(fw * fh, 1)
                    best = max(best, area * conf)
            scores.append((best, idx))
        cap.release()

        scores.sort(reverse=True)
        return [idx for _, idx in scores[:top_k]]
    except Exception as e:
        logger.warning(f"_yolo_quality_select_frames failed: {e}")
        return []


def select_measurement_frames(
    omni_info: Dict[str, Any],
    video_class: str,
    fps: float,
) -> Dict[str, Dict[str, List[int]]]:
    """Determine per-clip frame indices suitable for physics measurement.

    A-class: converts GT upright frames (measurement) and all GT frames (fallback)
             to clip-relative indices.
    B1/B2:   no frame selection (empty lists): B-class extraction (_vlm_extract_b_physics)
             samples its own 5 frames and never reads these indices.

    Returns:
        {clip_key: {"measurement": [...], "fallback": [...]}}
        Empty lists mean no suitable frames found; caller should fall back to
        uniform sampling.
    """
    clips = omni_info.get("mid_level", {}).get("clips", [])
    result: Dict[str, Dict[str, List[int]]] = {}

    if video_class == "A":
        gt_physics      = omni_info.get("gt_physics", {})
        upright_frames  = gt_physics.get("measurement_frame_indices", [])
        all_gt_frames   = gt_physics.get("all_gt_frame_indices", [])
        upright_set     = set(upright_frames)
        gt_frame_set    = set(all_gt_frames)

        for i, clip in enumerate(clips):
            key        = f"clip_{i}"
            clip_start = int(clip.get("start_time", 0) * fps)
            clip_end   = int(clip.get("end_time", 999999) * fps)

            clip_upright = sorted(f - clip_start for f in upright_set
                                  if clip_start <= f < clip_end)
            clip_all_gt  = sorted(f - clip_start for f in gt_frame_set
                                  if clip_start <= f < clip_end)

            result[key] = {"measurement": clip_upright, "fallback": clip_all_gt}
            logger.info(
                f"  [FrameSelect] {key}: {len(clip_upright)} upright / "
                f"{len(clip_all_gt)} total GT frames (clip-relative)"
            )

    else:  # B1 / B2
        # No YOLO frame scoring for B-class: augment_opq returns from its VLM-only B branch
        # before measurement/object frame indices are used.
        for i, clip in enumerate(clips):
            result[f"clip_{i}"] = {"measurement": [], "fallback": []}

    return result


def run_stage25_cv(
    omni_info: Dict[str, Any],
    video_class: str = "B",
    anchor_specs: Optional[Dict[str, Dict]] = None,
) -> Dict[str, Any]:
    """Run CV physics extraction over all clips in omni_info (Stage 2.5).

    Each measurement carries track_id + bbox_region so Stage 2 graph builder
    can match physical quantities to specific object instances and avoid
    confusing multiple similar objects in the same scene.

    Args:
        omni_info:     Stage 1 output dict (contains mid_level.clips with clip_path).
        video_class:   "A" | "B"  — B uses scale-free extraction.
        anchor_specs:  {label: {"h_m", "w_m", "mass_kg", "confidence"}} — optional
                       pre-fetched web-search specs (dimensions + physics constants).
                       For A-class: used for calibration. For B2: passed as context
                       to the Stage 3 LLM prompt (hidden from solver).

    Returns:
        {"clip_cv": {"clip_0": {...}, ...}, "video_class": str,
         "anchor_specs_used": [...], "web_search_specs": {...}}
    """
    clips = omni_info.get("mid_level", {}).get("clips", [])
    sport_category = omni_info.get("high_level", {}).get("sport_category", "")
    fps = omni_info.get("meta", {}).get("fps", 30.0)

    # Make a local copy so we don't mutate the caller's dict
    anchor_specs = dict(anchor_specs or {})
    gt_segments: Dict[str, float] = {}

    # A-class: derive calibration anchors from EgoExo4D 3D joint coordinates.
    # Primary: per-segment scale from MediaPipe 2D landmarks vs GT 3D lengths (gt_segments).
    # Fallback: GT body height injected as anchor_specs["person"] for calibrate_scale().
    if video_class == "A":
        gt_physics = omni_info.get("gt_physics", {})
        joints_sample = gt_physics.get("joints_per_frame_sample", {})
        if joints_sample:
            gt_segments = _compute_gt_segment_lengths(joints_sample)
            if gt_segments:
                logger.info(
                    f"  [GT anchor] Computed {len(gt_segments)} segment lengths: "
                    + ", ".join(f"{k}={v:.3f}m" for k, v in sorted(gt_segments.items()))
                )
            # Also inject GT body height as fallback in case segment method fails
            gt_height_m = _gt_height_from_3d_joints(joints_sample)
            if gt_height_m:
                anchor_specs["person"] = {
                    "h_m": gt_height_m,
                    "confidence": 0.90,
                    "source_type": "sensor_gt",
                    "source": "gt_3d_pose",
                }
                logger.info(f"  [GT anchor] Fallback body height: {gt_height_m:.3f}m")
        else:
            logger.warning("  [GT anchor] No joints_per_frame_sample in gt_physics")

    # Frame selection: determine per-clip measurement-worthy frame indices before any extraction.
    frame_selection = select_measurement_frames(omni_info, video_class, fps)

    clip_cv: Dict[str, Any] = {}
    for i, clip in enumerate(clips):
        key = f"clip_{i}"
        clip_path = clip.get("clip_path", "")
        if not clip_path or not os.path.exists(clip_path):
            clip_cv[key] = {"error": f"clip not found: {clip_path}", "clip_idx": i}
            continue

        vlm_opq = clip.get("observable_physical_quantities", {})
        # Extract non-person scene objects from Stage 1 visual_elements for physics extraction
        raw_ve = clip.get("visual_elements", [])
        if isinstance(raw_ve, list):
            scene_objs = _parse_visual_elements(raw_ve)
        else:
            scene_objs = []
        if scene_objs:
            logger.info(f"  [Stage 2.5] clip_{i} scene objects: {scene_objs}")
        clip_summary = clip.get("global_summary", "")
        try:
            fs = frame_selection.get(key, {})
            result = augment_opq(
                clip_path, fps, vlm_opq, sport_category,
                web_search_specs=anchor_specs or {},
                video_class=video_class,
                scene_objects=scene_objs,
                gt_segments=gt_segments if video_class == "A" else None,
                measurement_frame_indices=fs.get("measurement") or None,
                object_frame_indices=fs.get("fallback") or None,
                clip_summary=clip_summary,
            )
            result["frame_selection"] = fs
            # Attach track_id + bbox_region to each subject/motion for cross-stage traceability.
            # augment_opq builds shared_tracks internally; expose the per-track bbox here.
            shared_tracks = _run_yolo_tracks(clip_path)
            track_map: Dict[str, Dict] = {}
            for tid, dets in shared_tracks.items():
                if not dets:
                    continue
                mid = dets[len(dets) // 2]
                cls_id = int(mid[5]) if len(mid) > 5 else -1
                label = _YOLO_CLS_NAMES.get(cls_id, f"obj_{cls_id}")
                # mid = (x1, y1, x2, y2, bh, cls_id)
                track_map[str(tid)] = {
                    "track_id": tid,
                    "label_hint": label,
                    "bbox_region": [round(float(mid[0]), 1), round(float(mid[1]), 1),
                                    round(float(mid[2]), 1), round(float(mid[3]), 1)],
                }

            # Tag subjects with nearest track
            cap = cv2.VideoCapture(clip_path)
            fw = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            fh = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            cap.release()
            for subj in result.get("subjects", []):
                best_tid, best_d = None, 1e9
                lbl = subj.get("label", "").lower()
                for tid_s, tm in track_map.items():
                    if lbl and lbl not in tm["label_hint"] and tm["label_hint"] not in lbl:
                        continue
                    bx = (tm["bbox_region"][0] + tm["bbox_region"][2]) / 2 / max(fw, 1)
                    by = (tm["bbox_region"][1] + tm["bbox_region"][3]) / 2 / max(fh, 1)
                    d = abs(bx - 0.5) + abs(by - 0.5)
                    if d < best_d:
                        best_d, best_tid = d, tid_s
                if best_tid:
                    tm = track_map[best_tid]
                    subj["track_id"] = tm["track_id"]
                    subj["bbox_region"] = tm["bbox_region"]
                    subj["label_hint"] = tm["label_hint"]

            result["clip_idx"] = i
            result["clip_path"] = clip_path
            clip_cv[key] = result
            logger.info(
                f"  [Stage 2.5 CV] clip_{i}: "
                f"{len(result.get('subjects', []))} subjects, "
                f"{len(result.get('motions', []))} motions, "
                f"tracks={len(track_map)}"
            )
        except Exception as exc:
            logger.warning(f"  [Stage 2.5 CV] clip_{i} failed: {exc}")
            clip_cv[key] = {"error": str(exc), "clip_idx": i}

    anchor_labels = sorted(anchor_specs.keys()) if anchor_specs else []
    return {
        "clip_cv": clip_cv,
        "video_class": video_class,
        "anchor_specs_used": anchor_labels,
        "web_search_specs": dict(anchor_specs) if anchor_specs else {},
    }


# ===========================================================================
# EgoExo4D A-class GT Adapter
# ===========================================================================

def load_egoexo_gt(annotation_path: str, video_fps: float = 30.0) -> Dict[str, Any]:
    """
    Load EgoExo4D ego_pose body annotation JSON and compute derived physics quantities.

    Expected annotation format:
      {"frames": {"0": {"joints": {"right_wrist": [x, y, z], ...}, "confidence": 0.99}, ...}}
    or flat list format:
      [{"frame_idx": 0, "joints": {...}, "confidence": 0.99}, ...]

    Returns a gt_physics dict with joints_per_frame, velocities, accelerations,
    and derived_quantities.
    """
    if not os.path.exists(annotation_path):
        logger.warning(f"EgoExo4D GT annotation not found: {annotation_path}")
        return {}

    with open(annotation_path, encoding="utf-8") as f:
        raw = json.load(f)

    # Normalize to {frame_idx: {joints: {name: [x,y,z]}, confidence: float}}
    frames_dict: Dict[int, Dict] = {}
    if isinstance(raw, dict) and "frames" in raw:
        for k, v in raw["frames"].items():
            frames_dict[int(k)] = v
    elif isinstance(raw, list):
        for entry in raw:
            fidx = entry.get("frame_idx", entry.get("frame", 0))
            frames_dict[int(fidx)] = entry
    elif isinstance(raw, dict):
        # EgoExo4D actual format: {"32": [{"annotation3D": {"left-wrist": {"x":..,"y":..,"z":..}, ...}}], ...}
        # where keys are frame number strings and values are lists with one annotation entry
        first_val = next(iter(raw.values()), None)
        if isinstance(first_val, list):
            for frame_str, entries in raw.items():
                try:
                    fidx = int(frame_str)
                except ValueError:
                    continue
                if not entries:
                    continue
                ann3d = entries[0].get("annotation3D", {})
                # Convert {"left-wrist": {"x":..,"y":..,"z":..}} → {"left-wrist": [x,y,z]}
                joints = {
                    jname: [jval["x"], jval["y"], jval["z"]]
                    for jname, jval in ann3d.items()
                    if isinstance(jval, dict) and "x" in jval
                }
                frames_dict[fidx] = {"joints": joints, "confidence": entries[0].get("confidence", 0.9)}
        else:
            # take_uid → frames structure
            for take_uid, take_data in raw.items():
                if isinstance(take_data, dict) and "frames" in take_data:
                    for k, v in take_data["frames"].items():
                        frames_dict[int(k)] = v
                    break

    if not frames_dict:
        logger.warning(f"No frames parsed from {annotation_path}")
        return {}

    sorted_frames = sorted(frames_dict.keys())
    joints_per_frame: Dict[str, Dict[str, list]] = {}
    for fidx in sorted_frames:
        entry = frames_dict[fidx]
        joints = entry.get("joints", entry.get("body_3d", {}))
        joints_per_frame[str(fidx)] = {"joints": joints, "confidence": entry.get("confidence", 0.9)}

    # Collect all joint names
    all_joint_names: set = set()
    for entry in joints_per_frame.values():
        all_joint_names.update(entry["joints"].keys())

    dt = 1.0 / video_fps if video_fps > 0 else 1.0 / 30.0

    # Compute per-joint velocities (finite difference, m/s)
    velocities: Dict[str, Dict[str, float]] = {}  # joint -> {mean_speed, max_speed, ...}
    accelerations: Dict[str, Dict[str, float]] = {}

    for joint in all_joint_names:
        positions = []
        for fidx in sorted_frames:
            pos = joints_per_frame[str(fidx)]["joints"].get(joint)
            if pos and len(pos) >= 3:
                positions.append((fidx, np.array(pos[:3], dtype=float)))

        if len(positions) < 2:
            continue

        # Velocity: central differences where possible
        speeds = []
        vels = []
        for i in range(1, len(positions)):
            df = positions[i][0] - positions[i-1][0]
            if df <= 0:
                continue
            dpos = positions[i][1] - positions[i-1][1]
            v_vec = dpos / (df * dt)
            speed = float(np.linalg.norm(v_vec))
            speeds.append(speed)
            vels.append(v_vec)

        if speeds:
            # Also track which frame had the peak speed
            max_speed_val = 0.0
            max_speed_frame = None
            for i in range(1, len(positions)):
                df = positions[i][0] - positions[i-1][0]
                if df <= 0:
                    continue
                dpos = positions[i][1] - positions[i-1][1]
                speed_i = float(np.linalg.norm(dpos / (df * dt)))
                if speed_i > max_speed_val:
                    max_speed_val = speed_i
                    max_speed_frame = positions[i][0]   # full-video frame index

            velocities[joint] = {
                "mean_speed_mps": round(float(np.mean(speeds)), 4),
                "max_speed_mps":  round(float(np.max(speeds)), 4),
                "max_speed_frame": max_speed_frame,     # full-video frame index
                "max_speed_time_s": round(max_speed_frame * dt, 2) if max_speed_frame else None,
                "unit": "m/s",
                "source_type": "sensor_gt",
                "reliability_tier": "GT_A",
                "confidence": 0.98,
            }

        # Acceleration: second-order differences
        if len(vels) >= 2:
            accels = []
            for i in range(1, len(vels)):
                a_vec = (vels[i] - vels[i-1]) / dt
                accels.append(float(np.linalg.norm(a_vec)))
            accelerations[joint] = {
                "mean_accel_mps2": round(float(np.mean(accels)), 4),
                "max_accel_mps2":  round(float(np.max(accels)), 4),
                "unit": "m/s²",
                "source_type": "computed_from_gt",
                "reliability_tier": "GT_B",
                "confidence": 0.94,
            }

    # Derived summary quantities
    derived: Dict[str, Any] = {}

    # Max wrist speed (relevant for throws, swings, etc.)
    wrist_joints = [j for j in velocities if "wrist" in j.lower()]
    if wrist_joints:
        best_wrist_j = max(wrist_joints, key=lambda j: velocities[j]["max_speed_mps"])
        max_wrist = velocities[best_wrist_j]["max_speed_mps"]
        max_frame = velocities[best_wrist_j].get("max_speed_frame")
        max_time_s = velocities[best_wrist_j].get("max_speed_time_s")
        dq = build_quantity_metadata(max_wrist, "m/s", "sensor_gt", "egoexo4d_ego_pose", "direct")
        dq["max_at_video_time_s"] = max_time_s   # allows LLM to verify action matches
        derived["max_wrist_speed_mps"] = dq

    # Max elbow speed
    elbow_joints = [j for j in velocities if "elbow" in j.lower()]
    if elbow_joints:
        best_elbow_j = max(elbow_joints, key=lambda j: velocities[j]["max_speed_mps"])
        max_elbow = velocities[best_elbow_j]["max_speed_mps"]
        max_time_s = velocities[best_elbow_j].get("max_speed_time_s")
        dq = build_quantity_metadata(max_elbow, "m/s", "sensor_gt", "egoexo4d_ego_pose", "direct")
        dq["max_at_video_time_s"] = max_time_s
        derived["max_elbow_speed_mps"] = dq

    # Stride length estimate from ankle positions
    ankle_joints = [j for j in all_joint_names if "ankle" in j.lower()]
    if len(ankle_joints) >= 2:
        left_ankle  = [joints_per_frame[str(f)]["joints"].get("left_ankle")  for f in sorted_frames]
        right_ankle = [joints_per_frame[str(f)]["joints"].get("right_ankle") for f in sorted_frames]
        left_pos  = [np.array(p[:3]) for p in left_ankle  if p and len(p) >= 3]
        right_pos = [np.array(p[:3]) for p in right_ankle if p and len(p) >= 3]
        if left_pos and right_pos:
            # Horizontal displacement of same ankle across frames ≈ stride length
            ankle_displacements = []
            for i in range(1, len(left_pos)):
                d = np.linalg.norm(left_pos[i][:2] - left_pos[i-1][:2])  # XY plane
                if d > 0.1:  # Filter noise: only count steps > 10cm
                    ankle_displacements.append(d)
            if ankle_displacements:
                stride = float(np.median(ankle_displacements))
                derived["stride_length_m"] = build_quantity_metadata(
                    stride, "m", "computed_from_gt", "egoexo4d_ego_pose", "computed_from_gt"
                )

    # Total duration and frame count
    derived["total_frames"] = len(sorted_frames)
    derived["duration_s"] = round((sorted_frames[-1] - sorted_frames[0]) * dt, 3) if sorted_frames else 0.0

    # Compute upright frame indices for measurement selection
    upright_frames = [
        f for f in sorted_frames
        if _is_upright_from_gt_joints(joints_per_frame.get(str(f), {}).get("joints", {}))
    ]

    return {
        "gt_source": "egoexo4d_ego_pose",
        "annotation_path": annotation_path,
        "video_fps": video_fps,
        "joint_count": len(all_joint_names),
        "frame_count": len(sorted_frames),
        "joints_per_frame_sample": joints_per_frame,
        "all_gt_frame_indices": sorted_frames,
        "measurement_frame_indices": upright_frames,
        "velocities": velocities,
        "accelerations": accelerations,
        "derived_quantities": derived,
    }


def inject_gt_physics(omni_info: Dict[str, Any], gt_physics: Dict[str, Any]) -> Dict[str, Any]:
    """
    Inject EgoExo4D GT physics data into the omni_info dict produced by Stage 1.
    Adds a 'gt_physics' key at the top level, and enriches each clip's
    observable_physical_quantities with GT-derived values.
    """
    if not gt_physics:
        return omni_info

    omni_info = dict(omni_info)  # shallow copy
    omni_info["gt_physics"] = gt_physics
    omni_info["dataset_class"] = "A"
    omni_info["gt_source"] = gt_physics.get("gt_source", "egoexo4d_ego_pose")

    derived = gt_physics.get("derived_quantities", {})

    # Enrich observable_physical_quantities in high_level
    hl = omni_info.setdefault("high_level", {})
    opq = hl.setdefault("observable_physical_quantities", {})

    for key, val in derived.items():
        if isinstance(val, dict) and "value" in val:
            opq[f"gt_{key}"] = val
        elif isinstance(val, (int, float)):
            opq[f"gt_{key}"] = val

    # Do NOT inject GT into each clip — GT quantities (e.g. max_wrist_speed) are
    # video-level metrics, not clip-specific. Injecting them into clips would be
    # misleading (each clip would show the same video-level max).

    return omni_info


def pick_best_exo_cam(take_dir: str) -> Optional[str]:
    """
    Select the best exo camera video from an EgoExo4D take directory.
    Strategy: largest frame_aligned_videos/cam*.mp4 file (usually highest quality).
    Falls back to any cam*.mp4 found recursively.
    """
    frame_aligned = os.path.join(take_dir, "frame_aligned_videos")
    if os.path.isdir(frame_aligned):
        cams = [f for f in os.listdir(frame_aligned) if f.startswith("cam") and f.endswith(".mp4")]
        if cams:
            return max(
                [os.path.join(frame_aligned, c) for c in cams],
                key=os.path.getsize
            )

    # Fallback: search recursively
    import glob as _glob
    all_cams = _glob.glob(os.path.join(take_dir, "**", "cam*.mp4"), recursive=True)
    if all_cams:
        return max(all_cams, key=os.path.getsize)

    return None


# ===========================================================================
# CLI
# ===========================================================================

if __name__ == "__main__":
    import sys
    logging.basicConfig(level=logging.INFO,
                        format="%(asctime)s [%(levelname)s] %(message)s",
                        datefmt="%H:%M:%S")

    parser = argparse.ArgumentParser(description="Test physics extractor on a single clip")
    parser.add_argument("--clip",     required=True)
    parser.add_argument("--fps",      type=float, default=30.0)
    parser.add_argument("--sport",    default="")
    parser.add_argument("--vlm_opq",  default=None, help="JSON string of VLM OPQ")
    args = parser.parse_args()

    vlm_opq = {}
    if args.vlm_opq:
        vlm_opq = json.loads(args.vlm_opq)

    result = augment_opq(args.clip, args.fps, vlm_opq, args.sport)
    print(json.dumps(result, indent=2, ensure_ascii=False))

