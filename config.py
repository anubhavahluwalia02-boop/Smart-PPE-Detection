"""
Configuration settings for Smart PPE Compliance Detection System.
"""
import os
from pathlib import Path

# Base paths
BASE_DIR = Path(__file__).resolve().parent
MODELS_DIR = BASE_DIR / "models"
UPLOADS_DIR = BASE_DIR / "uploads"
VIOLATIONS_DIR = BASE_DIR / "violations"
STATIC_DIR = BASE_DIR / "static"

# Ensure runtime directories exist
for folder in [MODELS_DIR, UPLOADS_DIR, VIOLATIONS_DIR, STATIC_DIR]:
    folder.mkdir(parents=True, exist_ok=True)

# Database
DATABASE_PATH = BASE_DIR / "ppe_violations.db"

# Flask
SECRET_KEY = os.getenv("SECRET_KEY", "smart_ppe_compliance_default_secret_key_2026")
FLASK_PORT = int(os.getenv("FLASK_PORT", 5000))
FLASK_HOST = os.getenv("FLASK_HOST", "127.0.0.1")
DEBUG = os.getenv("DEBUG", "False").lower() in ("true", "1", "yes")

# Model Configuration
LOCAL_MODEL_PATH = MODELS_DIR / "best.pt"
CONFIDENCE_THRESHOLD = float(os.getenv("CONFIDENCE_THRESHOLD", 0.45))
IOU_THRESHOLD = float(os.getenv("IOU_THRESHOLD", 0.45))

# Roboflow Optional Settings
ROBOFLOW_API_KEY = os.getenv("ROBOFLOW_API_KEY", "")
ROBOFLOW_MODEL_ID = os.getenv("ROBOFLOW_MODEL_ID", "")
ROBOFLOW_VERSION = os.getenv("ROBOFLOW_VERSION", "1")

# Standard Target PPE Classes (Supports all 4 primary safety gears)
TARGET_CLASSES = {
    "person": ["person", "worker", "human"],
    "helmet": ["helmet", "hard_hat", "hardhat", "safety_helmet"],
    "no_helmet": ["no_helmet", "no-helmet", "head", "without_helmet", "no_hardhat"],
    "vest": ["vest", "safety_vest", "jacket", "protective_suit", "reflective_jacket"],
    "no_vest": ["no_vest", "no-vest", "no_safety_vest", "without_vest"],
    "glasses": ["glasses", "safety_glasses", "goggles", "eye_protection", "protective_eyewear"],
    "no_glasses": ["no_glasses", "no-glasses", "without_glasses", "bare_eyes"],
    "gloves": ["gloves", "safety_gloves", "protective_gloves", "hand_protection"],
    "no_gloves": ["no_gloves", "no-gloves", "without_gloves", "bare_hands"]
}

# Supported Equipment Categories
CORE_PPE_CLASSES = ["helmet", "vest"]
EXTENDED_PPE_CLASSES = ["glasses", "gloves"]
ALL_PPE_CLASSES = ["helmet", "vest", "glasses", "gloves"]

# Spatial Association Thresholds
# Head region: Top portion of person bounding box
HEAD_REGION_RATIO = 0.38
# Eye region: Vertical band within head for safety glasses
EYE_REGION_TOP_RATIO = 0.12
EYE_REGION_BOTTOM_RATIO = 0.28
# Torso region: Upper-mid portion of person bounding box
TORSO_REGION_TOP_RATIO = 0.20
TORSO_REGION_BOTTOM_RATIO = 0.78
# Hand/arm regions for gloves: Outer lower-mid bounds of torso/hips
HAND_REGION_TOP_RATIO = 0.45
HAND_REGION_BOTTOM_RATIO = 0.85
# Minimum overlap ratio to assign PPE item to person
MIN_OVERLAP_RATIO = 0.30

# Violation Alerts & Cooldown
VIOLATION_COOLDOWN_SECONDS = float(os.getenv("VIOLATION_COOLDOWN", 5.0))
ENABLE_AUDIO_ALERT = os.getenv("ENABLE_AUDIO_ALERT", "True").lower() in ("true", "1", "yes")

# ESP32 Physical Hardware Integration
SERIAL_PORT = os.getenv("SERIAL_PORT", "COM3")
SERIAL_BAUD = int(os.getenv("SERIAL_BAUD", 115200))
ENABLE_SERIAL = os.getenv("ENABLE_SERIAL", "False").lower() in ("true", "1", "yes")

# ESP32 Pin Assignment (Matches physical board wiring)
ESP32_PIN_GREEN = int(os.getenv("ESP32_PIN_GREEN", 18))   # Green LED (Safe state)
ESP32_PIN_RED = int(os.getenv("ESP32_PIN_RED", 19))       # Red LED (Violation state)
ESP32_PIN_BUZZER = int(os.getenv("ESP32_PIN_BUZZER", 21)) # Piezoelectric Buzzer

# UI Palette for Bounding Boxes (BGR format for OpenCV)
COLORS = {
    "COMPLIANT": (46, 204, 113),       # Emerald Green
    "HELMET_VIOLATION": (41, 128, 185), # Orange / Red
    "SAFETY_VEST_VIOLATION": (52, 73, 94), # Dark Amber
    "MULTIPLE_VIOLATIONS": (0, 0, 220), # Bright Red
    "NEEDS_REVIEW": (241, 196, 15),     # Yellow / Amber
    "PERSON": (230, 230, 230),          # Soft White
    "HELMET": (46, 204, 113),           # Green
    "NO_HELMET": (0, 0, 230),           # Red
    "VEST": (46, 204, 113),             # Green
    "NO_VEST": (0, 0, 230),              # Red
    "GLASSES": (52, 152, 219),          # Sky Blue
    "NO_GLASSES": (0, 0, 230),          # Red
    "GLOVES": (155, 89, 182),           # Purple
    "NO_GLOVES": (0, 0, 230)            # Red
}

