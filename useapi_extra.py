"""Extra UseAPI.net nodes — PixVerse images + MiniMax file upload.

Kept separate from useapi_nodes.py to avoid merge thrash on the large core file.
"""
from __future__ import annotations

import io
import json
import time
import urllib.parse

import numpy as np
import torch
from PIL import Image

try:
    from . import useapi_nodes as core
except ImportError:
    import useapi_nodes as core


LOG = core.LOG
logger = core.logger
BASE_URL = core.BASE_URL


def _tensor_to_png_bytes(image: torch.Tensor) -> bytes:
    """Convert first frame of IMAGE tensor to PNG bytes."""
    if image is None:
        raise ValueError("image is required")
    t = image
    if t.ndim == 4:
        t = t[0]
    arr = (t.detach().cpu().numpy() * 255.0).clip(0, 255).astype(np.uint8)
    img = Image.fromarray(arr, mode="RGB")
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    return buf.getvalue()


class UseapiMinimaxUploadFile(core._BaseNode):
    """Upload an image to MiniMax (Hailuo) via UseAPI for use as fileID in video jobs."""

    CATEGORY = "Useapi.net/MiniMax"
    FUNCTION = "execute"
    RETURN_TYPES = ("STRING", "STRING")
    RETURN_NAMES = ("file_id", "oss_path")

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "image": ("IMAGE",),
            },
            "optional": {
                "api_token": ("STRING", {"default": ""}),
                "account": ("STRING", {"default": ""}),
            },
        }

    def execute(self, image, api_token: str = "", account: str = ""):
        token = core._get_token(api_token)
        png = _tensor_to_png_bytes(image)
        qs = ""
        if account.strip():
            qs = "?" + urllib.parse.urlencode({"account": account.strip()})
        url = f"{BASE_URL}/minimax/files/{qs}" if qs else f"{BASE_URL}/minimax/files/"
        headers = {
            "Authorization": f"Bearer {token}",
            "Content-Type": "image/png",
        }
        status, body = core._make_request(url, "POST", headers, png, timeout=120)
        data = core._check_status(status, body, url, "MiniMax file upload", token)
        file_id = data.get("fileID") or data.get("fileId") or ""
        oss = data.get("ossPath") or ""
        if not file_id:
            raise RuntimeError(f"{LOG} MiniMax upload: no fileID in response: {data}")
        logger.info(f"{LOG} MiniMax upload ok fileID={file_id[:60]}...")
        return (file_id, oss)


class UseapiPixverseGenerateImage(core._BaseNode):
    """Generate images via UseAPI PixVerse v2 (Seedream 5, Nano Banana, Kling, GPT Image, ...)."""

    CATEGORY = "Useapi.net/PixVerse"
    FUNCTION = "execute"
    OUTPUT_NODE = True
    RETURN_TYPES = ("IMAGE", "STRING", "STRING")
    RETURN_NAMES = ("image", "image_url", "image_id")

    @classmethod
    def INPUT_TYPES(cls):
        models = [
            "nano-banana-2-lite",
            "nano-banana-2",
            "nano-banana-pro",
            "nano-banana",
            "seedream-5.0-pro",
            "seedream-5.0-lite",
            "seedream-4.5",
            "seedream-4.0",
            "qwen-image",
            "kling-3.0",
            "kling-o3",
            "gpt-image-2.0",
        ]
        qualities = ["512p", "720p", "1080p", "1440p", "1800p", "2160p"]
        ars = ["auto", "1:1", "16:9", "9:16", "4:3", "3:4", "5:4", "4:5", "3:2", "2:3", "21:9", "2:1", "1:2"]
        return {
            "required": {
                "prompt": ("STRING", {"multiline": True, "default": ""}),
                "model": (models, {"default": "nano-banana-2-lite"}),
                "quality": (qualities, {"default": "1080p"}),
                "aspect_ratio": (ars, {"default": "auto"}),
            },
            "optional": {
                "api_token": ("STRING", {"default": ""}),
                "email": ("STRING", {"default": ""}),
                "create_count": ("INT", {"default": 1, "min": 1, "max": 4}),
                "seed": ("INT", {"default": 0, "min": 0, "max": 2147483647}),
                "detail_level": (["medium", "low", "high"], {"default": "medium"}),
                "timeout": ("INT", {"default": 300, "min": 30, "max": 3600}),
            },
        }

    def execute(
        self,
        prompt: str,
        model: str,
        quality: str,
        aspect_ratio: str,
        api_token: str = "",
        email: str = "",
        create_count: int = 1,
        seed: int = 0,
        detail_level: str = "medium",
        timeout: int = 300,
    ):
        token = core._get_token(api_token)
        # PixVerse is on API v2
        url = "https://api.useapi.net/v2/pixverse/images/create"
        body = {
            "prompt": prompt,
            "model": model,
            "quality": quality,
            "aspect_ratio": aspect_ratio,
            "create_count": int(create_count),
        }
        if email.strip():
            body["email"] = email.strip()
        if seed:
            body["seed"] = int(seed)
        if model == "gpt-image-2.0":
            body["detail_level"] = detail_level

        logger.info(f"{LOG} PixVerse Image: model={model} quality={quality} prompt='{prompt[:50]}...'")
        status, raw = core._make_request(
            url, "POST", core._auth_headers(token),
            json.dumps(body).encode("utf-8"), timeout=min(timeout, 120),
        )
        data = core._check_status(status, raw, url, "PixVerse create", token)

        image_ids = []
        if data.get("success_ids"):
            image_ids = list(data["success_ids"])
        elif data.get("image_id"):
            image_ids = [data["image_id"]]
        if not image_ids:
            raise RuntimeError(f"{LOG} PixVerse create: no image_id in response: {data}")

        image_id = image_ids[0]
        # Poll GET /images/{id}
        poll_url = f"https://api.useapi.net/v2/pixverse/images/{urllib.parse.quote(image_id, safe='')}"
        deadline = time.time() + timeout
        image_url = ""
        while time.time() < deadline:
            st, body2 = core._make_request(
                poll_url, "GET", {"Authorization": f"Bearer {token}"}, None, timeout=60
            )
            pdata = core._check_status(st, body2, poll_url, "PixVerse poll", token)
            status_name = str(
                pdata.get("image_status_name")
                or pdata.get("status")
                or pdata.get("image_status")
                or ""
            ).upper()
            if "FAIL" in status_name or "ERROR" in status_name:
                raise RuntimeError(f"{LOG} PixVerse image failed: {pdata}")
            image_url = pdata.get("image_url") or pdata.get("url") or ""
            if image_url and ("COMPLETE" in status_name or status_name in ("", "SUCCESS", "DONE")):
                # Some responses return url before COMPLETE — prefer COMPLETE when available
                if "COMPLETE" in status_name or image_url:
                    if "COMPLETE" in status_name or pdata.get("image_url"):
                        break
            if image_url and "COMPLETE" in status_name:
                break
            # Accept URL once present and not clearly pending
            if image_url and status_name in ("COMPLETED", "SUCCESS", "DONE", "COMPLETE"):
                break
            time.sleep(3)

        # Final fetch if we exited with URL but loop condition was messy
        if not image_url:
            st, body2 = core._make_request(
                poll_url, "GET", {"Authorization": f"Bearer {token}"}, None, timeout=60
            )
            pdata = core._check_status(st, body2, poll_url, "PixVerse final", token)
            image_url = pdata.get("image_url") or pdata.get("url") or ""
        if not image_url:
            raise RuntimeError(f"{LOG} PixVerse timed out waiting for image_url (id={image_id})")

        s2, img_bytes = core._make_request(image_url, "GET", {}, None, 60)
        if s2 != 200:
            raise RuntimeError(f"{LOG} Failed to download PixVerse image HTTP {s2}")
        tensor = core._bytes_to_tensor(img_bytes)
        logger.info(f"{LOG} PixVerse Image complete id={image_id[:50]}...")
        return (tensor, image_url, image_id)


NODE_CLASS_MAPPINGS = {
    "UseapiMinimaxUploadFile": UseapiMinimaxUploadFile,
    "UseapiPixverseGenerateImage": UseapiPixverseGenerateImage,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "UseapiMinimaxUploadFile": "Useapi MiniMax Upload File",
    "UseapiPixverseGenerateImage": "Useapi PixVerse Generate Image",
}
