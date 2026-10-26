"""
Evidence capture and cooldown management for Smart PPE Detection System.
"""
import time
import uuid
import datetime
from pathlib import Path
from typing import Optional, Dict, Any, Tuple
import config

try:
    import cv2
    CV2_AVAILABLE = True
except ImportError:
    CV2_AVAILABLE = False
    cv2 = None

try:
    import numpy as np
except ImportError:
    np = None

class EvidenceManager:
    """
    Manages snapshot capture for safety violations with a configurable cooldown
    to prevent flooding the filesystem during continuous violations.
    """

    def __init__(self, cooldown_seconds: float = config.VIOLATION_COOLDOWN_SECONDS):
        self.cooldown_seconds = cooldown_seconds
        self.last_violation_times: Dict[str, float] = {}

    def is_in_cooldown(self, violation_type: str) -> bool:
        """Checks if a violation type is currently within its cooldown window."""
        now = time.time()
        last_time = self.last_violation_times.get(violation_type, 0.0)
        return (now - last_time) < self.cooldown_seconds

    def record_violation_timestamp(self, violation_type: str):
        """Updates the timestamp when a violation occurred."""
        self.last_violation_times[violation_type] = time.time()

    def capture_evidence(
        self,
        frame: Any,
        violation_type: str,
        missing_ppe: str,
        worker_info: Optional[Dict[str, Any]] = None,
        force: bool = False
    ) -> Optional[Tuple[str, str]]:
        """
        Captures and saves an annotated evidence frame to disk.
        Returns (relative_path, absolute_path) or None if in cooldown.
        """
        if not force and self.is_in_cooldown(violation_type):
            return None

        # Update cooldown timestamp
        self.record_violation_timestamp(violation_type)

        if not CV2_AVAILABLE or cv2 is None or frame is None:
            filename = f"violation_{violation_type.lower()}_{uuid.uuid4().hex[:6]}.jpg"
            return (f"violations/{filename}", str(config.VIOLATIONS_DIR / filename))

        # Clone frame to avoid mutating the original
        annotated = frame.copy()
        h, w = annotated.shape[:2]

        now = datetime.datetime.now()
        timestamp_str = now.strftime("%Y-%m-%d %H:%M:%S")
        date_file_str = now.strftime("%Y%m%d_%H%M%S")
        short_id = uuid.uuid4().hex[:6]

        # Draw top industrial watermark badge
        header_h = 50
        overlay = annotated.copy()
        cv2.rectangle(overlay, (0, 0), (w, header_h), (25, 25, 30), -1)
        # Red violation indicator bar
        cv2.rectangle(overlay, (0, 0), (12, header_h), (0, 0, 220), -1)
        cv2.addWeighted(overlay, 0.85, annotated, 0.15, 0, annotated)

        # Text banner
        title_text = f"SAFETY VIOLATION DETECTED: {violation_type.replace('_', ' ')}"
        sub_text = f"TIME: {timestamp_str} | MISSING: {missing_ppe or 'None'} | ID: #{short_id}"

        cv2.putText(
            annotated,
            title_text,
            (24, 22),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.55,
            (255, 255, 255),
            2,
            cv2.LINE_AA
        )
        cv2.putText(
            annotated,
            sub_text,
            (24, 42),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.45,
            (200, 200, 200),
            1,
            cv2.LINE_AA
        )

        # Generate unique filename
        filename = f"violation_{violation_type.lower()}_{date_file_str}_{short_id}.jpg"
        abs_path = config.VIOLATIONS_DIR / filename
        rel_path = f"violations/{filename}"

        # Write image using OpenCV
        cv2.imwrite(str(abs_path), annotated)

        return (rel_path, str(abs_path))
