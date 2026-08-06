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


def _poll_until_video_url(
    *,
    poll_url: str,
    token: str,
    timeout: int,
    context: str,
    url_keys: tuple[str, ...] = ("video_url", "url", "result_url"),
    status_keys: tuple[str, ...] = ("status_name", "video_status_name", "status"),
    success_tokens: tuple[str, ...] = ("SUCCEED", "SUCCESS", "COMPLETED", "COMPLETE", "DONE"),
    fail_tokens: tuple[str, ...] = ("FAIL", "ERROR", "CANCEL"),
) -> tuple[str, dict]:
    deadline = time.time() + timeout
    last = {}
    while time.time() < deadline:
        st, body = core._make_request(
            poll_url, "GET", {"Authorization": f"Bearer {token}"}, None, timeout=60
        )
        data = core._check_status(st, body, poll_url, context, token)
        last = data
        status_name = ""
        for k in status_keys:
            if data.get(k) is not None:
                status_name = str(data.get(k)).upper()
                break
        # Kling nests under task
        task = data.get("task") if isinstance(data.get("task"), dict) else {}
        if not status_name and task:
            status_name = str(task.get("status_name") or task.get("status") or "").upper()
        if any(tok in status_name for tok in fail_tokens):
            raise RuntimeError(f"{LOG} {context} failed: {data}")
        # works array for Kling
        video_url = ""
        works = data.get("works") or task.get("works") or []
        if isinstance(works, list):
            for w in works:
                if not isinstance(w, dict):
                    continue
                res = w.get("resource") or w.get("resourceInfo") or {}
                if isinstance(res, dict):
                    video_url = res.get("resource") or res.get("url") or ""
                video_url = video_url or w.get("url") or w.get("video_url") or ""
                if video_url:
                    break
        if not video_url:
            for k in url_keys:
                if data.get(k):
                    video_url = data.get(k)
                    break
        final = bool(data.get("status_final") or task.get("status_final"))
        if video_url and (final or any(tok in status_name for tok in success_tokens) or status_name == ""):
            if final or any(tok in status_name for tok in success_tokens) or video_url:
                if final or any(tok in status_name for tok in success_tokens):
                    return video_url, data
                # if only URL present and status processing, keep waiting unless final
        if video_url and any(tok in status_name for tok in success_tokens):
            return video_url, data
        time.sleep(4)
    # last chance
    if last:
        works = last.get("works") or []
        for w in works if isinstance(works, list) else []:
            if isinstance(w, dict):
                res = w.get("resource") or {}
                u = (res.get("resource") if isinstance(res, dict) else None) or w.get("url")
                if u:
                    return u, last
        for k in url_keys:
            if last.get(k):
                return last[k], last
    raise RuntimeError(f"{LOG} {context} timed out after {timeout}s")


class UseapiKlingText2Video(core._BaseNode):
    """Generate video via UseAPI Kling v1 text2video (v3 / turbo / 2.6 / ...)."""

    CATEGORY = "Useapi.net/Kling"
    FUNCTION = "execute"
    OUTPUT_NODE = True
    RETURN_TYPES = ("STRING", "STRING", "STRING")
    RETURN_NAMES = ("video_url", "video_path", "task_id")

    @classmethod
    def INPUT_TYPES(cls):
        models = [
            "kling-v3-0",
            "kling-v3-0-turbo",
            "kling-v2-6",
            "kling-v2-5",
            "kling-v2-1-master",
            "kling-v1-6",
            "kling-v1-5",
        ]
        return {
            "required": {
                "prompt": ("STRING", {"multiline": True, "default": ""}),
                "model_name": (models, {"default": "kling-v3-0-turbo"}),
                "aspect_ratio": (["9:16", "16:9", "1:1"], {"default": "9:16"}),
                "mode": (["std", "pro", "4k"], {"default": "pro"}),
                "duration": ("INT", {"default": 5, "min": 3, "max": 15}),
            },
            "optional": {
                "api_token": ("STRING", {"default": ""}),
                "email": ("STRING", {"default": ""}),
                "enable_audio": ("BOOLEAN", {"default": True}),
                "timeout": ("INT", {"default": 900, "min": 60, "max": 7200}),
            },
        }

    def execute(
        self,
        prompt: str,
        model_name: str,
        aspect_ratio: str,
        mode: str,
        duration: int = 5,
        api_token: str = "",
        email: str = "",
        enable_audio: bool = True,
        timeout: int = 900,
    ):
        token = core._get_token(api_token)
        url = f"{BASE_URL}/kling/videos/text2video"
        body = {
            "prompt": prompt,
            "model_name": model_name,
            "aspect_ratio": aspect_ratio,
            "mode": mode,
            "duration": str(int(duration)),
        }
        if email.strip():
            body["email"] = email.strip()
        # turbo always-on audio — skip flag
        if model_name not in ("kling-v3-0-turbo",):
            body["enable_audio"] = bool(enable_audio)

        logger.info(f"{LOG} Kling T2V model={model_name} mode={mode} dur={duration}")
        status, raw = core._make_request(
            url, "POST", core._auth_headers(token),
            json.dumps(body).encode("utf-8"), timeout=min(timeout, 120),
        )
        data = core._check_status(status, raw, url, "Kling text2video", token)
        task = data.get("task") if isinstance(data.get("task"), dict) else {}
        task_id = str(task.get("id") or data.get("task_id") or data.get("id") or "")
        if not task_id:
            raise RuntimeError(f"{LOG} Kling create: no task id in {data}")

        poll_url = f"{BASE_URL}/kling/tasks/{urllib.parse.quote(str(task_id), safe='')}"
        video_url, _pdata = _poll_until_video_url(
            poll_url=poll_url, token=token, timeout=timeout, context="Kling poll"
        )
        video_path = core._download_file(video_url, ".mp4")
        logger.info(f"{LOG} Kling T2V complete task={task_id}")
        return (video_url, video_path, str(task_id))


class UseapiPixverseGenerateVideo(core._BaseNode):
    """Generate video via UseAPI PixVerse v2 (native + third-party models)."""

    CATEGORY = "Useapi.net/PixVerse"
    FUNCTION = "execute"
    OUTPUT_NODE = True
    RETURN_TYPES = ("STRING", "STRING", "STRING")
    RETURN_NAMES = ("video_url", "video_path", "video_id")

    @classmethod
    def INPUT_TYPES(cls):
        models = [
            "v6",
            "v5.6",
            "v5.5",
            "seedance-2.0",
            "seedance-2.0-fast",
            "seedance-2.0-mini",
            "kling-v3",
            "kling-o3",
            "veo-3.1-fast",
            "veo-3.1-standard",
            "veo-3.1-lite",
            "sora-2",
            "sora-2-pro",
            "happyhorse-1.0",
            "grok-imagine",
            "grok-imagine-1.5",
        ]
        return {
            "required": {
                "prompt": ("STRING", {"multiline": True, "default": ""}),
                "model": (models, {"default": "v6"}),
                "duration": ("INT", {"default": 5, "min": 1, "max": 15}),
                "quality": (["360p", "480p", "540p", "720p", "1080p", "2160p"], {"default": "720p"}),
                "aspect_ratio": (["16:9", "9:16", "1:1", "4:3", "3:4", "3:2", "2:3", "21:9"], {"default": "9:16"}),
            },
            "optional": {
                "api_token": ("STRING", {"default": ""}),
                "email": ("STRING", {"default": ""}),
                "first_frame_path": ("STRING", {"default": "", "tooltip": "PixVerse upload path from POST files"}),
                "audio": ("BOOLEAN", {"default": True}),
                "timeout": ("INT", {"default": 900, "min": 60, "max": 7200}),
            },
        }

    def execute(
        self,
        prompt: str,
        model: str,
        duration: int = 5,
        quality: str = "720p",
        aspect_ratio: str = "9:16",
        api_token: str = "",
        email: str = "",
        first_frame_path: str = "",
        audio: bool = True,
        timeout: int = 900,
    ):
        token = core._get_token(api_token)
        url = "https://api.useapi.net/v2/pixverse/videos/create"
        body = {
            "prompt": prompt,
            "model": model,
            "duration": int(duration),
            "quality": quality,
        }
        if email.strip():
            body["email"] = email.strip()
        if first_frame_path.strip():
            body["first_frame_path"] = first_frame_path.strip()
        else:
            body["aspect_ratio"] = aspect_ratio
        # audio toggle where supported
        if model not in ("grok-imagine", "grok-imagine-1.5", "veo-3.1-lite", "sora-2", "sora-2-pro"):
            body["audio"] = bool(audio)

        logger.info(f"{LOG} PixVerse Video model={model} quality={quality} dur={duration}")
        status, raw = core._make_request(
            url, "POST", core._auth_headers(token),
            json.dumps(body).encode("utf-8"), timeout=min(timeout, 120),
        )
        data = core._check_status(status, raw, url, "PixVerse video create", token)
        video_id = data.get("video_id") or data.get("image_id") or ""
        if not video_id:
            raise RuntimeError(f"{LOG} PixVerse video create: no video_id in {data}")

        poll_url = f"https://api.useapi.net/v2/pixverse/videos/{urllib.parse.quote(str(video_id), safe='')}"
        # image templates may return image endpoint — try videos first
        try:
            video_url, _ = _poll_until_video_url(
                poll_url=poll_url,
                token=token,
                timeout=timeout,
                context="PixVerse video poll",
                url_keys=("video_url", "url", "image_url"),
                status_keys=("video_status_name", "status_name", "status"),
            )
        except RuntimeError:
            # fallback image id path
            poll_url = f"https://api.useapi.net/v2/pixverse/images/{urllib.parse.quote(str(video_id), safe='')}"
            video_url, _ = _poll_until_video_url(
                poll_url=poll_url,
                token=token,
                timeout=timeout,
                context="PixVerse image-template poll",
                url_keys=("image_url", "video_url", "url"),
                status_keys=("image_status_name", "status_name", "status"),
            )

        video_path = core._download_file(video_url, ".mp4" if "video" in video_url or video_url.endswith(".mp4") else ".mp4")
        logger.info(f"{LOG} PixVerse Video complete id={str(video_id)[:50]}")
        return (video_url, video_path, str(video_id))


NODE_CLASS_MAPPINGS = {
    "UseapiMinimaxUploadFile": UseapiMinimaxUploadFile,
    "UseapiPixverseGenerateImage": UseapiPixverseGenerateImage,
    "UseapiKlingText2Video": UseapiKlingText2Video,
    "UseapiPixverseGenerateVideo": UseapiPixverseGenerateVideo,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "UseapiMinimaxUploadFile": "Useapi MiniMax Upload File",
    "UseapiPixverseGenerateImage": "Useapi PixVerse Generate Image",
    "UseapiKlingText2Video": "Useapi Kling Text-to-Video",
    "UseapiPixverseGenerateVideo": "Useapi PixVerse Generate Video",
}
