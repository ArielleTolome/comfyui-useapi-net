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

# ── Shared Utilities ─────────────────────────────────────────────────────────

def _get_token(api_token: str) -> str:
    """Return API token: use direct input, else USEAPI_TOKEN env var."""
    token = (api_token or "").strip()
    if not token:
        token = os.environ.get("USEAPI_TOKEN", "").strip()
    if not token:
        raise ValueError(
            f"{LOG} API token not provided. "
            "Set the USEAPI_TOKEN environment variable or wire the api_token input."
        )
    return token


def _auth_headers(token: str) -> dict:
    """Return JSON auth headers for Useapi.net requests."""
    return {
        "Authorization": f"Bearer {token}",
        "Content-Type": "application/json",
    }


def _make_request(url: str, method: str = "GET", headers: dict = None,
                  data: bytes = None, timeout: int = 600):
    """Make an HTTP request. Returns (status_code, response_body_bytes)."""
    req = urllib.request.Request(url, data=data, headers=headers or {}, method=method)
    try:
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            return resp.status, resp.read()
    except urllib.error.HTTPError as e:
        return e.code, e.read()
    except urllib.error.URLError as e:
        raise RuntimeError(f"{LOG} Network error reaching {url}: {e.reason}")
    except TimeoutError:
        raise RuntimeError(f"{LOG} Request timed out after {timeout}s: {url}")


def _check_status(status: int, body: bytes, url: str, context: str = "") -> dict:
    """Parse JSON response body; raise descriptive error if status != 200."""
    try:
        data = json.loads(body) if body else {}
    except json.JSONDecodeError:
        data = {}
    if status == 200:
        return data
    detail = data.get("error", body[:300].decode(errors="replace"))
    label = f"[{context}] " if context else ""
    if status == 429:
        raise RuntimeError(
            f"{LOG} {label}Rate limited (429). Wait 5-10s or add more Useapi.net accounts. "
            f"URL: {url}\nDetail: {detail}"
        )
    if status == 503:
        raise RuntimeError(
            f"{LOG} {label}Service unavailable (503). Retry in a moment. URL: {url}\nDetail: {detail}"
        )
    if status == 408:
        raise RuntimeError(
            f"{LOG} {label}Request timeout (408). Generation took too long. URL: {url}\nDetail: {detail}"
        )
    if status == 401:
        raise RuntimeError(
            f"{LOG} {label}Unauthorized (401). Check your Useapi.net token. URL: {url}"
        )
    raise RuntimeError(f"{LOG} {label}HTTP {status} from {url}.\nDetail: {detail}")


def _build_multipart(fields: dict, files: dict):
    """Build multipart/form-data without the requests library.

    Args:
        fields: {"name": "string_value"}
        files:  {"name": ("filename.ext", bytes_data, "mime/type")}
    Returns:
        (body_bytes, content_type_string_with_boundary)
    """
    boundary = "----ComfyUIBoundary" + uuid.uuid4().hex
    body = b""
    for name, value in fields.items():
        body += f"--{boundary}\r\n".encode()
        body += f'Content-Disposition: form-data; name="{name}"\r\n\r\n'.encode()
        body += str(value).encode()
        body += b"\r\n"
    for name, (filename, data, ctype) in files.items():
        body += f"--{boundary}\r\n".encode()
        body += (
            f'Content-Disposition: form-data; name="{name}"; filename="{filename}"\r\n'
            f"Content-Type: {ctype}\r\n\r\n"
        ).encode()
        body += data
        body += b"\r\n"
    body += f"--{boundary}--\r\n".encode()
    return body, f"multipart/form-data; boundary={boundary}"


def _tensor_to_png_bytes(tensor: torch.Tensor) -> bytes:
    """Convert ComfyUI IMAGE tensor (1, H, W, 3) float32 → PNG bytes."""
    arr = tensor[0].cpu().detach().numpy()
    arr = (arr * 255).clip(0, 255).astype(np.uint8)
    pil = Image.fromarray(arr, "RGB")
    buf = io.BytesIO()
    pil.save(buf, format="PNG")
    return buf.getvalue()


def _bytes_to_tensor(img_bytes: bytes) -> torch.Tensor:
    """Convert raw image bytes → ComfyUI IMAGE tensor (1, H, W, 3) float32."""
    pil = Image.open(io.BytesIO(img_bytes)).convert("RGB")
    arr = np.array(pil).astype(np.float32) / 255.0
    return torch.from_numpy(arr).unsqueeze(0)


def _download_file(url: str, ext: str = ".mp4") -> str:
    """Download URL to cache dir with MD5-hash filename. Returns local path."""
    os.makedirs(VIDEO_CACHE_DIR, exist_ok=True)
    fname = hashlib.md5(url.encode()).hexdigest() + ext
    dest = os.path.join(VIDEO_CACHE_DIR, fname)
    if os.path.exists(dest):
        print(f"{LOG} Cache hit: {dest}")
        return dest
    print(f"{LOG} Downloading to {dest} ...")
    with urllib.request.urlopen(url, timeout=120) as resp:
        data = resp.read()
    with open(dest, "wb") as f:
        f.write(data)
    print(f"{LOG} Downloaded {len(data):,} bytes → {dest}")
    return dest


def _runway_poll(task_id: str, token: str,
                 poll_interval: int = 10, max_wait: int = 600) -> list:
    """Poll Runway task until SUCCEEDED. Returns list of artifacts dicts."""
    poll_url = (
        f"{BASE_URL}/runwayml/tasks/?taskId={urllib.parse.quote(task_id, safe='')}"
    )
    headers = _auth_headers(token)
    deadline = time.time() + max_wait
    while time.time() < deadline:
        time.sleep(poll_interval)
        status, raw = _make_request(poll_url, "GET", headers, None, 30)
        data = _check_status(status, raw, poll_url, f"Runway poll {task_id[:30]}")
        task_status = data.get("status", "")
        print(f"{LOG} Runway task {task_id[:30]}... → {task_status}")
        if task_status == "SUCCEEDED":
            artifacts = data.get("artifacts", [])
            if not artifacts:
                raise RuntimeError(
                    f"{LOG} Runway task SUCCEEDED but no artifacts in response: {data}"
                )
            return artifacts
        if task_status in ("FAILED", "CANCELLED", "THROTTLED"):
            raise RuntimeError(
                f"{LOG} Runway task ended with status '{task_status}'. task_id={task_id}"
            )
    raise RuntimeError(
        f"{LOG} Runway task timed out after {max_wait}s. task_id={task_id}"
    )


def _runway_upload_image(token: str, image_tensor: torch.Tensor,
                         email: str = "") -> str:
    """Upload a ComfyUI IMAGE tensor to Runway as an image asset. Returns assetId."""
    url = f"{BASE_URL}/runwayml/assets"
    png_bytes = _tensor_to_png_bytes(image_tensor)
    fields = {"email": email.strip()} if email.strip() else {}
    files = {"file": ("comfyui_upload.png", png_bytes, "image/png")}
    body, ct = _build_multipart(fields, files)
    headers = {"Authorization": f"Bearer {token}", "Content-Type": ct}
    print(f"{LOG} Runway: uploading image asset...")
    status, raw = _make_request(url, "POST", headers, body, timeout=60)
    data = _check_status(status, raw, url, "Runway upload asset")
    asset_id = data.get("assetId", "")
    if not asset_id:
        raise RuntimeError(f"{LOG} Runway upload: no assetId in response: {data}")
    print(f"{LOG} Runway asset uploaded: {asset_id[:50]}...")
    return asset_id

# ── Node classes added in Tasks 3-16 ─────────────────────────────────────────

# ── ComfyUI Registration ──────────────────────────────────────────────────────
NODE_CLASS_MAPPINGS = {}
NODE_DISPLAY_NAME_MAPPINGS = {}
