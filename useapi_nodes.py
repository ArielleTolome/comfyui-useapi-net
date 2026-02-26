"""ComfyUI-UseapiNet: Custom nodes for Useapi.net API integration.

Provides image and video generation via:
  - Google Flow: Imagen 4, Gemini (Nano Banana), Veo 3.1
  - Runway: Gen-4, Gen-4 Turbo, Gen-3 Turbo, Frames
"""
import os
import io
import json
import time
import uuid
import base64
import hashlib
import tempfile
import urllib.request
import urllib.error
import urllib.parse
import numpy as np
import torch
from PIL import Image

try:
    import cv2
    _CV2_AVAILABLE = True
except ImportError:
    _CV2_AVAILABLE = False

# ── Constants ────────────────────────────────────────────────────────────────
LOG = "[Useapi.net]"
BASE_URL = "https://api.useapi.net/v1"
VIDEO_CACHE_DIR = os.path.join(tempfile.gettempdir(), "comfyui_useapi_videos")

# ── Shared utilities added in Task 2 ─────────────────────────────────────────
# ── Node classes added in Tasks 3-16 ─────────────────────────────────────────

# ── ComfyUI Registration ──────────────────────────────────────────────────────
NODE_CLASS_MAPPINGS = {}
NODE_DISPLAY_NAME_MAPPINGS = {}
