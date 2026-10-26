"""
Compliance logic and spatial association engine for Smart PPE Detection.
Supports detection and verification of Safety Helmet, Safety Vest, Safety Glasses, and Gloves.
Clearly identifies which classes the trained model actually supports vs unsupported.
"""
from typing import List, Dict, Any, Tuple, Optional
import config

def calculate_intersection_ratio(boxA: List[float], boxB: List[float]) -> float:
    """
    Calculates the proportion of boxA that lies inside boxB.
    box format: [x1, y1, x2, y2]
    """
    xA = max(boxA[0], boxB[0])
    yA = max(boxA[1], boxB[1])
    xB = min(boxA[2], boxB[2])
    yB = min(boxA[3], boxB[3])

    inter_width = max(0.0, xB - xA)
    inter_height = max(0.0, yB - yA)
    inter_area = inter_width * inter_height

    boxA_area = max(1.0, (boxA[2] - boxA[0]) * (boxA[3] - boxA[1]))
    return inter_area / boxA_area

def box_center(box: List[float]) -> Tuple[float, float]:
    """Returns the (cx, cy) center coordinate of a bounding box."""
    return ((box[0] + box[2]) / 2.0, (box[1] + box[3]) / 2.0)

def is_point_inside(pt: Tuple[float, float], box: List[float]) -> bool:
    """Checks whether a 2D coordinate is inside a bounding box."""
    return box[0] <= pt[0] <= box[2] and box[1] <= pt[1] <= box[3]

def normalize_class_name(raw_name: str) -> str:
    """Normalizes class names across various YOLO and Roboflow dataset formats."""
    cleaned = raw_name.lower().replace("-", "_").replace(" ", "_").strip()
    for standard_name, aliases in config.TARGET_CLASSES.items():
        if cleaned == standard_name or cleaned in aliases:
            return standard_name
    return cleaned

class ComplianceAnalyzer:
    """
    Associates detected PPE items (helmets, vests, glasses, gloves)
    with detected persons using spatial containment and anatomical proportion mapping.
    Distinguishes between model-supported classes and unsupported weights.
    """

    def __init__(
        self,
        head_ratio: float = config.HEAD_REGION_RATIO,
        torso_top_ratio: float = config.TORSO_REGION_TOP_RATIO,
        torso_bottom_ratio: float = config.TORSO_REGION_BOTTOM_RATIO,
        min_overlap: float = config.MIN_OVERLAP_RATIO
    ):
        self.head_ratio = head_ratio
        self.torso_top_ratio = torso_top_ratio
        self.torso_bottom_ratio = torso_bottom_ratio
        self.min_overlap = min_overlap

    def get_anatomical_regions(self, person_box: List[float]) -> Dict[str, List[float]]:
        """
        Derives head, eye band, upper torso, and hand/arm extremity regions from a person's bounding box [x1, y1, x2, y2].
        """
        px1, py1, px2, py2 = person_box
        pw = px2 - px1
        ph = py2 - py1

        margin_x = pw * 0.05
        rx1 = px1 - margin_x
        rx2 = px2 + margin_x

        head_box = [
            rx1,
            py1 - (ph * 0.05),
            rx2,
            py1 + (ph * self.head_ratio)
        ]

        # Eye region for safety glasses/goggles
        eye_box = [
            rx1 + (pw * 0.15),
            py1 + (ph * config.EYE_REGION_TOP_RATIO),
            rx2 - (pw * 0.15),
            py1 + (ph * config.EYE_REGION_BOTTOM_RATIO)
        ]

        torso_box = [
            rx1,
            py1 + (ph * self.torso_top_ratio),
            rx2,
            py1 + (ph * self.torso_bottom_ratio)
        ]

        # Hand/Arm zones for gloves
        hands_box = [
            px1 - (pw * 0.15),
            py1 + (ph * config.HAND_REGION_TOP_RATIO),
            px2 + (pw * 0.15),
            py1 + (ph * config.HAND_REGION_BOTTOM_RATIO)
        ]

        return {
            "head": head_box,
            "eyes": eye_box,
            "torso": torso_box,
            "hands": hands_box
        }

    def inspect_model_capabilities(self, model_classes: Optional[List[str]]) -> Dict[str, Any]:
        """
        Determines which of the 4 PPE categories (helmet, vest, glasses, gloves)
        are genuinely supported by the active model weights.
        """
        if not model_classes:
            # Default fallback assumption for standard PPE models
            return {
                "supported": ["helmet", "vest"],
                "unsupported": ["glasses", "gloves"],
                "has_helmet": True,
                "has_vest": True,
                "has_glasses": False,
                "has_gloves": False,
                "notes": "Model classes not introspected. Defaulting to standard Helmet & Vest weights."
            }

        norm_names = [normalize_class_name(c) for c in model_classes]

        has_helmet = any(c in ("helmet", "no_helmet") for c in norm_names)
        has_vest = any(c in ("vest", "no_vest") for c in norm_names)
        has_glasses = any(c in ("glasses", "no_glasses") for c in norm_names)
        has_gloves = any(c in ("gloves", "no_gloves") for c in norm_names)

        supported = []
        unsupported = []

        if has_helmet: supported.append("helmet")
        else: unsupported.append("helmet")

        if has_vest: supported.append("vest")
        else: unsupported.append("vest")

        if has_glasses: supported.append("glasses")
        else: unsupported.append("glasses")

        if has_gloves: supported.append("gloves")
        else: unsupported.append("gloves")

        return {
            "supported": supported,
            "unsupported": unsupported,
            "has_helmet": has_helmet,
            "has_vest": has_vest,
            "has_glasses": has_glasses,
            "has_gloves": has_gloves,
            "raw_classes": model_classes,
            "notes": (
                f"Model genuinely supports: {', '.join(supported).title()}. "
                f"Unsupported by weights: {', '.join(unsupported).title() if unsupported else 'None'}."
            )
        }

    def evaluate_detections(
        self,
        detections: List[Dict[str, Any]],
        frame_dimensions: Tuple[int, int] = (640, 480),
        model_classes: Optional[List[str]] = None
    ) -> Dict[str, Any]:
        """
        Evaluates a list of raw detections against persons in scene.
        Properly reports supported vs unsupported classes for glasses and gloves.
        """
        capabilities = self.inspect_model_capabilities(model_classes)

        # 1. Separate persons from PPE components
        persons = []
        ppe_items = []

        for d in detections:
            norm_name = normalize_class_name(d["class_name"])
            item = {
                "box": d["box"],
                "class_name": norm_name,
                "raw_name": d["class_name"],
                "confidence": float(d.get("confidence", 0.0))
            }
            if norm_name == "person":
                persons.append(item)
            else:
                ppe_items.append(item)

        # Inferred person if isolated PPE items exist
        if not persons and ppe_items:
            min_x = min(item["box"][0] for item in ppe_items)
            min_y = min(item["box"][1] for item in ppe_items)
            max_x = max(item["box"][2] for item in ppe_items)
            max_y = max(item["box"][3] for item in ppe_items)
            persons.append({
                "box": [min_x, min_y, max_x, max_y],
                "class_name": "person",
                "raw_name": "person_inferred",
                "confidence": 0.85
            })

        worker_evaluations = []

        # 2. Evaluate each person
        for idx, person in enumerate(persons):
            p_box = person["box"]
            p_conf = person["confidence"]
            p_w = p_box[2] - p_box[0]
            p_h = p_box[3] - p_box[1]

            regions = self.get_anatomical_regions(p_box)
            head_region = regions["head"]
            eyes_region = regions["eyes"]
            torso_region = regions["torso"]
            hands_region = regions["hands"]

            head_truncated = p_box[1] <= 4 and p_h < 100
            person_too_small = p_h < 45 or p_w < 25

            matched_helmet = None
            matched_no_helmet = None
            matched_vest = None
            matched_no_vest = None
            matched_glasses = None
            matched_no_glasses = None
            matched_gloves = None
            matched_no_gloves = None

            for item in ppe_items:
                item_box = item["box"]
                center = box_center(item_box)
                c_name = item["class_name"]

                # Head: Helmet
                if c_name in ("helmet", "no_helmet"):
                    overlap = calculate_intersection_ratio(item_box, head_region)
                    center_in = is_point_inside(center, head_region)
                    if overlap >= self.min_overlap or center_in:
                        if c_name == "helmet":
                            if matched_helmet is None or item["confidence"] > matched_helmet["confidence"]:
                                matched_helmet = item
                        elif c_name == "no_helmet":
                            if matched_no_helmet is None or item["confidence"] > matched_no_helmet["confidence"]:
                                matched_no_helmet = item

                # Torso: Vest
                elif c_name in ("vest", "no_vest"):
                    overlap = calculate_intersection_ratio(item_box, torso_region)
                    center_in = is_point_inside(center, torso_region)
                    if overlap >= self.min_overlap or center_in:
                        if c_name == "vest":
                            if matched_vest is None or item["confidence"] > matched_vest["confidence"]:
                                matched_vest = item
                        elif c_name == "no_vest":
                            if matched_no_vest is None or item["confidence"] > matched_no_vest["confidence"]:
                                matched_no_vest = item

                # Eyes: Safety Glasses
                elif c_name in ("glasses", "no_glasses"):
                    overlap = calculate_intersection_ratio(item_box, eyes_region)
                    center_in = is_point_inside(center, eyes_region)
                    if overlap >= self.min_overlap or center_in:
                        if c_name == "glasses":
                            if matched_glasses is None or item["confidence"] > matched_glasses["confidence"]:
                                matched_glasses = item
                        elif c_name == "no_glasses":
                            if matched_no_glasses is None or item["confidence"] > matched_no_glasses["confidence"]:
                                matched_no_glasses = item

                # Hands: Gloves
                elif c_name in ("gloves", "no_gloves"):
                    overlap = calculate_intersection_ratio(item_box, hands_region)
                    center_in = is_point_inside(center, hands_region)
                    if overlap >= self.min_overlap or center_in:
                        if c_name == "gloves":
                            if matched_gloves is None or item["confidence"] > matched_gloves["confidence"]:
                                matched_gloves = item
                        elif c_name == "no_gloves":
                            if matched_no_gloves is None or item["confidence"] > matched_no_gloves["confidence"]:
                                matched_no_gloves = item

            # Mandatory items: Helmet & Vest
            has_helmet = (matched_helmet is not None)
            explicit_no_helmet = (matched_no_helmet is not None)
            has_vest = (matched_vest is not None)
            explicit_no_vest = (matched_no_vest is not None)

            # Extended items: Glasses & Gloves (only enforced if model supports them)
            has_glasses = (matched_glasses is not None)
            has_gloves = (matched_gloves is not None)

            missing_list = []
            status = "NEEDS_REVIEW"
            status_reason = ""

            if person_too_small:
                status = "NEEDS_REVIEW"
                status_reason = "Worker too far or small for reliable detection"
            elif head_truncated:
                status = "NEEDS_REVIEW"
                status_reason = "Head partially out of camera frame"
            else:
                helmet_ok = has_helmet and not (explicit_no_helmet and matched_no_helmet["confidence"] > matched_helmet["confidence"])
                helmet_missing = explicit_no_helmet or (not has_helmet)

                vest_ok = has_vest and not (explicit_no_vest and matched_no_vest["confidence"] > matched_vest["confidence"])
                vest_missing = explicit_no_vest or (not has_vest)

                if helmet_missing and vest_missing:
                    status = "MULTIPLE_VIOLATIONS"
                    missing_list = ["Safety Helmet", "Safety Vest"]
                    status_reason = "Both safety helmet and reflective vest are missing"
                elif helmet_missing:
                    status = "HELMET_VIOLATION"
                    missing_list = ["Safety Helmet"]
                    status_reason = "Safety helmet is missing on worker head"
                elif vest_missing:
                    status = "SAFETY_VEST_VIOLATION"
                    missing_list = ["Safety Vest"]
                    status_reason = "Reflective safety vest is missing on worker torso"
                elif helmet_ok and vest_ok:
                    status = "COMPLIANT"
                    status_reason = "All mandatory PPE (Helmet + Vest) detected and compliant"
                else:
                    status = "NEEDS_REVIEW"
                    status_reason = "Uncertain PPE alignment or partial obstruction"

            # Formulate honest per-equipment status dictionary
            ppe_status_map = {
                "helmet": {
                    "supported_by_model": capabilities["has_helmet"],
                    "detected": has_helmet,
                    "status": "COMPLIANT" if has_helmet else "VIOLATION"
                },
                "vest": {
                    "supported_by_model": capabilities["has_vest"],
                    "detected": has_vest,
                    "status": "COMPLIANT" if has_vest else "VIOLATION"
                },
                "glasses": {
                    "supported_by_model": capabilities["has_glasses"],
                    "detected": has_glasses if capabilities["has_glasses"] else None,
                    "status": (
                        "COMPLIANT" if has_glasses else "VIOLATION"
                    ) if capabilities["has_glasses"] else "UNSUPPORTED_IN_MODEL_WEIGHTS"
                },
                "gloves": {
                    "supported_by_model": capabilities["has_gloves"],
                    "detected": has_gloves if capabilities["has_gloves"] else None,
                    "status": (
                        "COMPLIANT" if has_gloves else "VIOLATION"
                    ) if capabilities["has_gloves"] else "UNSUPPORTED_IN_MODEL_WEIGHTS"
                }
            }

            worker_evaluations.append({
                "worker_id": idx + 1,
                "box": p_box,
                "confidence": p_conf,
                "status": status,
                "status_reason": status_reason,
                "missing_ppe": missing_list,
                "helmet_detected": has_helmet,
                "vest_detected": has_vest,
                "glasses_detected": has_glasses if capabilities["has_glasses"] else None,
                "gloves_detected": has_gloves if capabilities["has_gloves"] else None,
                "ppe_gear_status": ppe_status_map,
                "matched_helmet": matched_helmet,
                "matched_vest": matched_vest,
                "matched_glasses": matched_glasses,
                "matched_gloves": matched_gloves,
                "head_region": head_region,
                "eyes_region": eyes_region,
                "torso_region": torso_region,
                "hands_region": hands_region
            })

        total_workers = len(worker_evaluations)
        compliant_count = sum(1 for w in worker_evaluations if w["status"] == "COMPLIANT")
        helmet_violations = sum(1 for w in worker_evaluations if w["status"] == "HELMET_VIOLATION")
        vest_violations = sum(1 for w in worker_evaluations if w["status"] == "SAFETY_VEST_VIOLATION")
        multiple_violations = sum(1 for w in worker_evaluations if w["status"] == "MULTIPLE_VIOLATIONS")
        needs_review_count = sum(1 for w in worker_evaluations if w["status"] == "NEEDS_REVIEW")

        frame_status = "COMPLIANT"
        if multiple_violations > 0:
            frame_status = "MULTIPLE_VIOLATIONS"
        elif helmet_violations > 0:
            frame_status = "HELMET_VIOLATION"
        elif vest_violations > 0:
            frame_status = "SAFETY_VEST_VIOLATION"
        elif needs_review_count > 0 and compliant_count == 0:
            frame_status = "NEEDS_REVIEW"
        elif total_workers == 0:
            frame_status = "CLEAR"

        return {
            "total_workers": total_workers,
            "compliant_count": compliant_count,
            "helmet_violations": helmet_violations,
            "vest_violations": vest_violations,
            "multiple_violations": multiple_violations,
            "needs_review_count": needs_review_count,
            "frame_status": frame_status,
            "model_capabilities": capabilities,
            "workers": worker_evaluations,
            "raw_detections": detections
        }
