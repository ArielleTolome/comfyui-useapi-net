"""Extra UseAPI.net nodes — PixVerse, MiniMax, Kling I2V/lipsync/motion.

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



def _kling_upload_asset(token: str, data: bytes, content_type: str, email: str = "") -> str:
    """Upload bytes to Kling assets; return usable URL (url/resourceUrl) or fileName."""
    qs = ""
    if email.strip():
        qs = "?" + urllib.parse.urlencode({"email": email.strip()})
    url = f"{BASE_URL}/kling/assets/{qs}" if qs else f"{BASE_URL}/kling/assets/"
    headers = {"Authorization": f"Bearer {token}", "Content-Type": content_type}
    status, body = core._make_request(url, "POST", headers, data, timeout=180)
    resp = core._check_status(status, body, url, "Kling asset upload", token)
    asset_url = (
        resp.get("url")
        or resp.get("resourceUrl")
        or resp.get("resource_url")
        or ""
    )
    file_name = resp.get("fileName") or resp.get("file_name") or ""
    if not asset_url and file_name:
        # resolve via GET assets/uploaded
        get_url = f"{BASE_URL}/kling/assets/uploaded/?fileName={urllib.parse.quote(file_name)}"
        if email.strip():
            get_url += f"&email={urllib.parse.quote(email.strip())}"
        st2, b2 = core._make_request(get_url, "GET", {"Authorization": f"Bearer {token}"}, None, 60)
        det = core._check_status(st2, b2, get_url, "Kling asset resolve", token)
        asset_url = (
            det.get("url")
            or det.get("resourceUrl")
            or det.get("resource_url")
            or (det.get("asset") or {}).get("url")
            or ""
        )
        if not asset_url and isinstance(det.get("items"), list) and det["items"]:
            item = det["items"][0]
            if isinstance(item, dict):
                asset_url = item.get("url") or item.get("resourceUrl") or ""
    if not asset_url:
        # last resort: some flows accept fileName
        if file_name:
            logger.warning(f"{LOG} Kling upload: no URL in response; returning fileName={file_name}")
            return file_name
        raise RuntimeError(f"{LOG} Kling upload: no url/resourceUrl/fileName in {resp}")
    return asset_url


def _kling_create_and_poll(token: str, path: str, body: dict, timeout: int, context: str):
    url = f"{BASE_URL}{path}"
    status, raw = core._make_request(
        url, "POST", core._auth_headers(token),
        json.dumps(body).encode("utf-8"), timeout=min(timeout, 120),
    )
    data = core._check_status(status, raw, url, context, token)
    task = data.get("task") if isinstance(data.get("task"), dict) else {}
    task_id = str(task.get("id") or data.get("task_id") or data.get("id") or "")
    if not task_id:
        raise RuntimeError(f"{LOG} {context}: no task id in {data}")
    poll_url = f"{BASE_URL}/kling/tasks/{urllib.parse.quote(str(task_id), safe='')}"
    video_url, _pdata = _poll_until_video_url(
        poll_url=poll_url, token=token, timeout=timeout, context=f"{context} poll"
    )
    video_path = core._download_file(video_url, ".mp4")
    return video_url, video_path, str(task_id)

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



class UseapiKlingUploadAsset(core._BaseNode):
    """Upload IMAGE / local video path / audio path to Kling assets."""

    CATEGORY = "Useapi.net/Kling"
    FUNCTION = "execute"
    RETURN_TYPES = ("STRING", "STRING")
    RETURN_NAMES = ("asset_url", "file_name")

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {},
            "optional": {
                "image": ("IMAGE",),
                "video_path": ("STRING", {"default": "", "tooltip": "Local mp4 path or empty"}),
                "audio_path": ("STRING", {"default": "", "tooltip": "Local mp3/wav path or empty"}),
                "api_token": ("STRING", {"default": ""}),
                "email": ("STRING", {"default": ""}),
            },
        }

    def execute(self, image=None, video_path: str = "", audio_path: str = "",
                api_token: str = "", email: str = ""):
        token = core._get_token(api_token)
        if image is not None:
            data = _tensor_to_png_bytes(image)
            ctype = "image/png"
        elif video_path.strip():
            path = video_path.strip()
            if not core._is_safe_path(path):
                raise ValueError(f"{LOG} unsafe video_path rejected")
            with open(path, "rb") as f:
                data = f.read()
            ctype = "video/mp4"
        elif audio_path.strip():
            path = audio_path.strip()
            if not core._is_safe_path(path):
                raise ValueError(f"{LOG} unsafe audio_path rejected")
            with open(path, "rb") as f:
                data = f.read()
            lower = path.lower()
            if lower.endswith(".wav"):
                ctype = "audio/wav"
            elif lower.endswith(".mp3"):
                ctype = "audio/mpeg"
            else:
                ctype = "audio/mpeg"
        else:
            raise ValueError(f"{LOG} Provide image, video_path, or audio_path")
        url = _kling_upload_asset(token, data, ctype, email)
        # file_name best-effort from URL
        file_name = url.rsplit("/", 1)[-1] if "/" in url else url
        logger.info(f"{LOG} Kling asset uploaded: {url[:80]}")
        return (url, file_name)


class UseapiKlingImage2Video(core._BaseNode):
    """Kling image2video-frames (start/end frame). Uploads IMAGE tensors automatically."""

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
            "kling-v2-1",
            "kling-v2-1-master",
            "kling-v1-6",
            "kling-v1-5",
        ]
        return {
            "required": {
                "image": ("IMAGE",),
                "prompt": ("STRING", {"multiline": True, "default": ""}),
                "model_name": (models, {"default": "kling-v3-0-turbo"}),
                "mode": (["std", "pro", "4k"], {"default": "pro"}),
                "duration": ("INT", {"default": 5, "min": 3, "max": 15}),
            },
            "optional": {
                "image_tail": ("IMAGE",),
                "api_token": ("STRING", {"default": ""}),
                "email": ("STRING", {"default": ""}),
                "enable_audio": ("BOOLEAN", {"default": True}),
                "timeout": ("INT", {"default": 900, "min": 60, "max": 7200}),
            },
        }

    def execute(
        self,
        image,
        prompt: str,
        model_name: str,
        mode: str = "pro",
        duration: int = 5,
        image_tail=None,
        api_token: str = "",
        email: str = "",
        enable_audio: bool = True,
        timeout: int = 900,
    ):
        token = core._get_token(api_token)
        start_url = _kling_upload_asset(token, _tensor_to_png_bytes(image), "image/png", email)
        body = {
            "image": start_url,
            "prompt": prompt,
            "model_name": model_name,
            "mode": mode,
            "duration": str(int(duration)),
        }
        if email.strip():
            body["email"] = email.strip()
        if image_tail is not None and model_name not in ("kling-v3-0-turbo", "kling-v2-1-master"):
            body["image_tail"] = _kling_upload_asset(
                token, _tensor_to_png_bytes(image_tail), "image/png", email
            )
        if model_name not in ("kling-v3-0-turbo",):
            body["enable_audio"] = bool(enable_audio)
        logger.info(f"{LOG} Kling I2V model={model_name} mode={mode}")
        return _kling_create_and_poll(
            token, "/kling/videos/image2video-frames", body, timeout, "Kling I2V"
        )


class UseapiKlingLipsync(core._BaseNode):
    """Apply Kling lip-sync to a video using an audio track (URLs or local paths)."""

    CATEGORY = "Useapi.net/Kling"
    FUNCTION = "execute"
    OUTPUT_NODE = True
    RETURN_TYPES = ("STRING", "STRING", "STRING")
    RETURN_NAMES = ("video_url", "video_path", "task_id")

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "video": ("STRING", {"default": "", "tooltip": "Video URL or local path"}),
                "audio": ("STRING", {"default": "", "tooltip": "Audio URL or local path"}),
            },
            "optional": {
                "api_token": ("STRING", {"default": ""}),
                "email": ("STRING", {"default": ""}),
                "timeout": ("INT", {"default": 900, "min": 60, "max": 7200}),
            },
        }

    def execute(self, video: str, audio: str, api_token: str = "", email: str = "", timeout: int = 900):
        token = core._get_token(api_token)

        def _resolve(val: str, kind: str) -> str:
            v = (val or "").strip()
            if not v:
                raise ValueError(f"{LOG} {kind} is required")
            if v.startswith("http://") or v.startswith("https://"):
                return v
            if not core._is_safe_path(v):
                raise ValueError(f"{LOG} unsafe {kind} path")
            with open(v, "rb") as f:
                data = f.read()
            if kind == "video":
                ctype = "video/mp4"
            else:
                ctype = "audio/mpeg" if v.lower().endswith(".mp3") else "audio/wav"
            return _kling_upload_asset(token, data, ctype, email)

        body = {
            "video": _resolve(video, "video"),
            "audio": _resolve(audio, "audio"),
        }
        if email.strip():
            body["email"] = email.strip()
        logger.info(f"{LOG} Kling lipsync start")
        return _kling_create_and_poll(token, "/kling/videos/lipsync", body, timeout, "Kling lipsync")


class UseapiKlingMotionCreate(core._BaseNode):
    """Apply motion from a reference video onto a person image (Kling motion-control)."""

    CATEGORY = "Useapi.net/Kling"
    FUNCTION = "execute"
    OUTPUT_NODE = True
    RETURN_TYPES = ("STRING", "STRING", "STRING")
    RETURN_NAMES = ("video_url", "video_path", "task_id")

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "image": ("IMAGE",),
                "motion": ("STRING", {"default": "", "tooltip": "Motion video URL or local path"}),
                "model_name": (["kling-v3-0", "kling-v2-6"], {"default": "kling-v3-0"}),
                "mode": (["std", "pro"], {"default": "std"}),
            },
            "optional": {
                "prompt": ("STRING", {"multiline": True, "default": ""}),
                "keep_audio": ("BOOLEAN", {"default": False}),
                "motion_direction": (["motion_direction", "image_direction"], {"default": "motion_direction"}),
                "api_token": ("STRING", {"default": ""}),
                "email": ("STRING", {"default": ""}),
                "timeout": ("INT", {"default": 900, "min": 60, "max": 7200}),
            },
        }

    def execute(
        self,
        image,
        motion: str,
        model_name: str = "kling-v3-0",
        mode: str = "std",
        prompt: str = "",
        keep_audio: bool = False,
        motion_direction: str = "motion_direction",
        api_token: str = "",
        email: str = "",
        timeout: int = 900,
    ):
        token = core._get_token(api_token)
        image_url = _kling_upload_asset(token, _tensor_to_png_bytes(image), "image/png", email)
        m = (motion or "").strip()
        if not m:
            raise ValueError(f"{LOG} motion is required")
        if m.startswith("http://") or m.startswith("https://"):
            motion_url = m
        else:
            if not core._is_safe_path(m):
                raise ValueError(f"{LOG} unsafe motion path")
            with open(m, "rb") as f:
                data = f.read()
            motion_url = _kling_upload_asset(token, data, "video/mp4", email)
        body = {
            "model_name": model_name,
            "imageUrl": image_url,
            "motionUrl": motion_url,
            "mode": mode,
            "keepAudio": bool(keep_audio),
            "motionDirection": motion_direction,
        }
        if prompt.strip():
            body["prompt"] = prompt.strip()
        if email.strip():
            body["email"] = email.strip()
        logger.info(f"{LOG} Kling motion-create model={model_name}")
        return _kling_create_and_poll(
            token, "/kling/videos/motion-create", body, timeout, "Kling motion"
        )



class UseapiKlingTTS(core._BaseNode):
    """Free Kling TTS — text to speech (up to 5 minutes). Returns audio URL + local path."""

    CATEGORY = "Useapi.net/Kling"
    FUNCTION = "execute"
    OUTPUT_NODE = True
    RETURN_TYPES = ("STRING", "STRING", "AUDIO")
    RETURN_NAMES = ("audio_url", "audio_path", "audio")

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "text": ("STRING", {"multiline": True, "default": ""}),
                "speaker_id": ("STRING", {"default": "", "tooltip": "From GET /kling/tts/voices"}),
            },
            "optional": {
                "api_token": ("STRING", {"default": ""}),
                "email": ("STRING", {"default": ""}),
                "speed": ("FLOAT", {"default": 1.0, "min": 0.8, "max": 2.0, "step": 0.05}),
                "emotion": (
                    ["neutral", "happy", "angry", "sad", "fearful", "disgusted", "surprised"],
                    {"default": "neutral"},
                ),
            },
        }

    def execute(
        self,
        text: str,
        speaker_id: str,
        api_token: str = "",
        email: str = "",
        speed: float = 1.0,
        emotion: str = "neutral",
    ):
        token = core._get_token(api_token)
        if not (text or "").strip():
            raise ValueError(f"{LOG} text is required")
        if not (speaker_id or "").strip():
            raise ValueError(f"{LOG} speaker_id is required (GET /kling/tts/voices)")
        body = {
            "speakerId": speaker_id.strip(),
            "text": text.strip(),
            "speed": float(speed),
            "emotion": emotion,
        }
        if email.strip():
            body["email"] = email.strip()
        url = f"{BASE_URL}/kling/tts/create"
        status, raw = core._make_request(
            url, "POST", core._auth_headers(token),
            json.dumps(body).encode("utf-8"), timeout=180,
        )
        data = core._check_status(status, raw, url, "Kling TTS", token)
        audio_url = data.get("resource") or data.get("url") or data.get("audio_url") or ""
        if not audio_url:
            raise RuntimeError(f"{LOG} Kling TTS: no resource URL in {data}")
        audio_path = core._download_file(audio_url, ".mp3")
        # build AUDIO for ComfyUI
        with open(audio_path, "rb") as f:
            audio_bytes = f.read()
        # reuse runway-style path if available; simple torch audio via torchaudio optional
        try:
            import torchaudio
            waveform, sr = torchaudio.load(audio_path)
            if waveform.ndim == 1:
                waveform = waveform.unsqueeze(0)
            if waveform.ndim == 2:
                waveform = waveform.unsqueeze(0)
            audio = {"waveform": waveform, "sample_rate": int(sr)}
        except Exception:
            # fallback silent short buffer if torchaudio missing
            import torch as _torch
            audio = {"waveform": _torch.zeros((1, 1, 16000)), "sample_rate": 16000}
            logger.warning(f"{LOG} torchaudio unavailable; AUDIO tensor is placeholder. Use audio_path.")
        logger.info(f"{LOG} Kling TTS complete: {audio_url[:80]}")
        return (audio_url, audio_path, audio)


class UseapiKlingAvatarVideo(core._BaseNode):
    """Kling Avatars 2.0 — lip-sync talking head from image/avatar + audio or TTS text."""

    CATEGORY = "Useapi.net/Kling"
    FUNCTION = "execute"
    OUTPUT_NODE = True
    RETURN_TYPES = ("STRING", "STRING", "STRING")
    RETURN_NAMES = ("video_url", "video_path", "task_id")

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "mode": (["std", "pro"], {"default": "std"}),
            },
            "optional": {
                "image": ("IMAGE",),
                "image_url": ("STRING", {"default": ""}),
                "avatar_id": ("STRING", {"default": ""}),
                "audio_url": ("STRING", {"default": ""}),
                "audio_path": ("STRING", {"default": ""}),
                "text": ("STRING", {"multiline": True, "default": ""}),
                "speaker_id": ("STRING", {"default": ""}),
                "prompt": ("STRING", {"multiline": True, "default": "Natural speaking"}),
                "speed": ("FLOAT", {"default": 1.0, "min": 0.8, "max": 2.0, "step": 0.05}),
                "emotion": (
                    ["neutral", "happy", "angry", "sad", "fearful", "disgusted", "surprised"],
                    {"default": "neutral"},
                ),
                "api_token": ("STRING", {"default": ""}),
                "email": ("STRING", {"default": ""}),
                "timeout": ("INT", {"default": 900, "min": 60, "max": 7200}),
            },
        }

    def execute(
        self,
        mode: str = "std",
        image=None,
        image_url: str = "",
        avatar_id: str = "",
        audio_url: str = "",
        audio_path: str = "",
        text: str = "",
        speaker_id: str = "",
        prompt: str = "Natural speaking",
        speed: float = 1.0,
        emotion: str = "neutral",
        api_token: str = "",
        email: str = "",
        timeout: int = 900,
    ):
        token = core._get_token(api_token)
        body: dict = {"mode": mode}
        if email.strip():
            body["email"] = email.strip()
        if prompt.strip():
            body["prompt"] = prompt.strip()

        # avatar source
        av = (avatar_id or "").strip()
        iu = (image_url or "").strip()
        if av:
            body["avatarId"] = av
        elif iu:
            body["imageUrl"] = iu
        elif image is not None:
            body["imageUrl"] = _kling_upload_asset(
                token, _tensor_to_png_bytes(image), "image/png", email
            )
        else:
            raise ValueError(f"{LOG} Provide avatar_id, image_url, or image")

        # audio source
        au = (audio_url or "").strip()
        ap = (audio_path or "").strip()
        tx = (text or "").strip()
        if au:
            body["audioUrl"] = au
        elif ap:
            if not core._is_safe_path(ap):
                raise ValueError(f"{LOG} unsafe audio_path")
            with open(ap, "rb") as f:
                data = f.read()
            ctype = "audio/mpeg" if ap.lower().endswith(".mp3") else "audio/wav"
            body["audioUrl"] = _kling_upload_asset(token, data, ctype, email)
        elif tx:
            if not (speaker_id or "").strip():
                raise ValueError(f"{LOG} speaker_id required when using text TTS")
            body["text"] = tx
            body["speakerId"] = speaker_id.strip()
            body["speed"] = float(speed)
            body["emotion"] = emotion
        else:
            raise ValueError(f"{LOG} Provide audio_url, audio_path, or text+speaker_id")

        logger.info(f"{LOG} Kling avatar video mode={mode}")
        return _kling_create_and_poll(
            token, "/kling/avatars/video", body, timeout, "Kling avatar video"
        )



def _pixverse_upload(token: str, data: bytes, content_type: str, email: str = "") -> str:
    """Upload to PixVerse files; return path (preferred) or url."""
    qs = ""
    if email.strip():
        qs = "?" + urllib.parse.urlencode({"email": email.strip()})
    url = f"https://api.useapi.net/v2/pixverse/files/{qs}" if qs else "https://api.useapi.net/v2/pixverse/files/"
    headers = {"Authorization": f"Bearer {token}", "Content-Type": content_type}
    status, body = core._make_request(url, "POST", headers, data, timeout=300)
    resp = core._check_status(status, body, url, "PixVerse file upload", token)
    path = resp.get("path") or (resp.get("file") or {}).get("path") or ""
    if not path:
        path = resp.get("url") or resp.get("id") or ""
    if not path:
        raise RuntimeError(f"{LOG} PixVerse upload: no path in {resp}")
    return str(path)


def _pixverse_create_and_poll(token: str, path: str, body: dict, timeout: int, context: str):
    url = f"https://api.useapi.net/v2{path}"
    status, raw = core._make_request(
        url, "POST", core._auth_headers(token),
        json.dumps(body).encode("utf-8"), timeout=min(timeout, 120),
    )
    data = core._check_status(status, raw, url, context, token)
    video_id = data.get("video_id") or data.get("image_id") or ""
    if not video_id:
        raise RuntimeError(f"{LOG} {context}: no video_id in {data}")
    poll_url = f"https://api.useapi.net/v2/pixverse/videos/{urllib.parse.quote(str(video_id), safe='')}"
    video_url, _ = _poll_until_video_url(
        poll_url=poll_url,
        token=token,
        timeout=timeout,
        context=f"{context} poll",
        url_keys=("url", "video_url", "image_url"),
        status_keys=("video_status_name", "status_name", "status"),
    )
    video_path = core._download_file(video_url, ".mp4")
    return video_url, video_path, str(video_id)


class UseapiPixverseUploadFile(core._BaseNode):
    """Upload IMAGE / local video / audio to PixVerse files; returns path for other nodes."""

    CATEGORY = "Useapi.net/PixVerse"
    FUNCTION = "execute"
    RETURN_TYPES = ("STRING",)
    RETURN_NAMES = ("path",)

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {},
            "optional": {
                "image": ("IMAGE",),
                "video_path": ("STRING", {"default": ""}),
                "audio_path": ("STRING", {"default": ""}),
                "api_token": ("STRING", {"default": ""}),
                "email": ("STRING", {"default": ""}),
            },
        }

    def execute(self, image=None, video_path: str = "", audio_path: str = "",
                api_token: str = "", email: str = ""):
        token = core._get_token(api_token)
        if image is not None:
            data = _tensor_to_png_bytes(image)
            ctype = "image/png"
        elif video_path.strip():
            vp = video_path.strip()
            if not core._is_safe_path(vp):
                raise ValueError(f"{LOG} unsafe video_path")
            with open(vp, "rb") as f:
                data = f.read()
            ctype = "video/mp4"
        elif audio_path.strip():
            ap = audio_path.strip()
            if not core._is_safe_path(ap):
                raise ValueError(f"{LOG} unsafe audio_path")
            with open(ap, "rb") as f:
                data = f.read()
            ctype = "audio/mpeg" if ap.lower().endswith(".mp3") else "audio/wav"
        else:
            raise ValueError(f"{LOG} Provide image, video_path, or audio_path")
        path = _pixverse_upload(token, data, ctype, email)
        logger.info(f"{LOG} PixVerse file uploaded: {path}")
        return (path,)


class UseapiPixverseLipsync(core._BaseNode):
    """Lip-sync a PixVerse video with audio_path or TTS prompt."""

    CATEGORY = "Useapi.net/PixVerse"
    FUNCTION = "execute"
    OUTPUT_NODE = True
    RETURN_TYPES = ("STRING", "STRING", "STRING")
    RETURN_NAMES = ("video_url", "video_path", "video_id")

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {},
            "optional": {
                "video_id": ("STRING", {"default": ""}),
                "video_path": ("STRING", {"default": "", "tooltip": "PixVerse upload path OR local file"}),
                "audio_path": ("STRING", {"default": "", "tooltip": "PixVerse upload path OR local file"}),
                "prompt": ("STRING", {"default": "", "tooltip": "TTS text if no audio_path (max 200)"}),
                "speaker_id": ("INT", {"default": 0, "min": 0, "max": 10_000_000}),
                "original_sound_switch": ("BOOLEAN", {"default": False}),
                "api_token": ("STRING", {"default": ""}),
                "email": ("STRING", {"default": ""}),
                "timeout": ("INT", {"default": 900, "min": 60, "max": 7200}),
            },
        }

    def execute(
        self,
        video_id: str = "",
        video_path: str = "",
        audio_path: str = "",
        prompt: str = "",
        speaker_id: int = 0,
        original_sound_switch: bool = False,
        api_token: str = "",
        email: str = "",
        timeout: int = 900,
    ):
        token = core._get_token(api_token)
        body: dict = {"original_sound_switch": bool(original_sound_switch)}
        if email.strip():
            body["email"] = email.strip()
        vid = (video_id or "").strip()
        vp = (video_path or "").strip()
        if vid:
            body["video_id"] = vid
        elif vp:
            if core._is_safe_path(vp) and not vp.startswith("upload/"):
                with open(vp, "rb") as f:
                    data = f.read()
                body["video_path"] = _pixverse_upload(token, data, "video/mp4", email)
            else:
                body["video_path"] = vp
        else:
            raise ValueError(f"{LOG} Provide video_id or video_path")

        ap = (audio_path or "").strip()
        if ap:
            if core._is_safe_path(ap) and not ap.startswith("upload/"):
                with open(ap, "rb") as f:
                    data = f.read()
                ctype = "audio/mpeg" if ap.lower().endswith(".mp3") else "audio/wav"
                body["audio_path"] = _pixverse_upload(token, data, ctype, email)
            else:
                body["audio_path"] = ap
        elif prompt.strip():
            body["prompt"] = prompt.strip()[:200]
            if speaker_id:
                body["speaker_id"] = int(speaker_id)
        else:
            raise ValueError(f"{LOG} Provide audio_path or prompt(+speaker_id)")

        logger.info(f"{LOG} PixVerse lipsync")
        return _pixverse_create_and_poll(token, "/pixverse/videos/lipsync", body, timeout, "PixVerse lipsync")


class UseapiPixverseMotionControl(core._BaseNode):
    """Drive a character image with motion from a reference video."""

    CATEGORY = "Useapi.net/PixVerse"
    FUNCTION = "execute"
    OUTPUT_NODE = True
    RETURN_TYPES = ("STRING", "STRING", "STRING")
    RETURN_NAMES = ("video_url", "video_path", "video_id")

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "image": ("IMAGE",),
                "motion": ("STRING", {"default": "", "tooltip": "Local video path or PixVerse upload path"}),
                "quality": (["720p", "540p", "360p"], {"default": "720p"}),
            },
            "optional": {
                "api_token": ("STRING", {"default": ""}),
                "email": ("STRING", {"default": ""}),
                "timeout": ("INT", {"default": 900, "min": 60, "max": 7200}),
            },
        }

    def execute(self, image, motion: str, quality: str = "720p",
                api_token: str = "", email: str = "", timeout: int = 900):
        token = core._get_token(api_token)
        frame_path = _pixverse_upload(token, _tensor_to_png_bytes(image), "image/png", email)
        m = (motion or "").strip()
        if not m:
            raise ValueError(f"{LOG} motion is required")
        if core._is_safe_path(m) and not m.startswith("upload/"):
            with open(m, "rb") as f:
                data = f.read()
            video_path = _pixverse_upload(token, data, "video/mp4", email)
        else:
            video_path = m
        body = {
            "frame_1_path": frame_path,
            "video_1_path": video_path,
            "quality": quality,
            "model": "v5.6",
        }
        if email.strip():
            body["email"] = email.strip()
        logger.info(f"{LOG} PixVerse motion-control")
        return _pixverse_create_and_poll(
            token, "/pixverse/videos/motion-control", body, timeout, "PixVerse motion"
        )


class UseapiPixverseExtend(core._BaseNode):
    """Extend a PixVerse (or Grok Imagine) video."""

    CATEGORY = "Useapi.net/PixVerse"
    FUNCTION = "execute"
    OUTPUT_NODE = True
    RETURN_TYPES = ("STRING", "STRING", "STRING")
    RETURN_NAMES = ("video_url", "video_path", "video_id")

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "prompt": ("STRING", {"multiline": True, "default": ""}),
                "model": (["v6", "grok-imagine"], {"default": "v6"}),
                "duration": ("INT", {"default": 5, "min": 1, "max": 15}),
                "quality": (["1080p", "720p", "540p", "480p", "360p"], {"default": "720p"}),
            },
            "optional": {
                "video_id": ("STRING", {"default": ""}),
                "video_path": ("STRING", {"default": ""}),
                "audio": ("BOOLEAN", {"default": False}),
                "api_token": ("STRING", {"default": ""}),
                "email": ("STRING", {"default": ""}),
                "timeout": ("INT", {"default": 900, "min": 60, "max": 7200}),
            },
        }

    def execute(
        self,
        prompt: str,
        model: str = "v6",
        duration: int = 5,
        quality: str = "720p",
        video_id: str = "",
        video_path: str = "",
        audio: bool = False,
        api_token: str = "",
        email: str = "",
        timeout: int = 900,
    ):
        token = core._get_token(api_token)
        body = {
            "prompt": prompt,
            "model": model,
            "duration": int(duration),
            "quality": quality,
        }
        if email.strip():
            body["email"] = email.strip()
        if model == "v6":
            body["audio"] = bool(audio)
        vid = (video_id or "").strip()
        vp = (video_path or "").strip()
        if vid:
            body["video_id"] = vid
        elif vp:
            if core._is_safe_path(vp) and not vp.startswith("upload/"):
                with open(vp, "rb") as f:
                    data = f.read()
                body["video_path"] = _pixverse_upload(token, data, "video/mp4", email)
            else:
                body["video_path"] = vp
        else:
            raise ValueError(f"{LOG} Provide video_id or video_path to extend")
        logger.info(f"{LOG} PixVerse extend model={model}")
        return _pixverse_create_and_poll(token, "/pixverse/videos/extend", body, timeout, "PixVerse extend")


NODE_CLASS_MAPPINGS = {
    "UseapiMinimaxUploadFile": UseapiMinimaxUploadFile,
    "UseapiPixverseGenerateImage": UseapiPixverseGenerateImage,
    "UseapiKlingText2Video": UseapiKlingText2Video,
    "UseapiPixverseGenerateVideo": UseapiPixverseGenerateVideo,
    "UseapiKlingUploadAsset": UseapiKlingUploadAsset,
    "UseapiKlingImage2Video": UseapiKlingImage2Video,
    "UseapiKlingLipsync": UseapiKlingLipsync,
    "UseapiKlingMotionCreate": UseapiKlingMotionCreate,
    "UseapiKlingTTS": UseapiKlingTTS,
    "UseapiKlingAvatarVideo": UseapiKlingAvatarVideo,
    "UseapiPixverseUploadFile": UseapiPixverseUploadFile,
    "UseapiPixverseLipsync": UseapiPixverseLipsync,
    "UseapiPixverseMotionControl": UseapiPixverseMotionControl,
    "UseapiPixverseExtend": UseapiPixverseExtend,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "UseapiMinimaxUploadFile": "Useapi MiniMax Upload File",
    "UseapiPixverseGenerateImage": "Useapi PixVerse Generate Image",
    "UseapiKlingText2Video": "Useapi Kling Text-to-Video",
    "UseapiPixverseGenerateVideo": "Useapi PixVerse Generate Video",
    "UseapiKlingUploadAsset": "Useapi Kling Upload Asset",
    "UseapiKlingImage2Video": "Useapi Kling Image-to-Video",
    "UseapiKlingLipsync": "Useapi Kling Lipsync",
    "UseapiKlingMotionCreate": "Useapi Kling Motion Control",
    "UseapiKlingTTS": "Useapi Kling TTS",
    "UseapiKlingAvatarVideo": "Useapi Kling Avatar Video",
    "UseapiPixverseUploadFile": "Useapi PixVerse Upload File",
    "UseapiPixverseLipsync": "Useapi PixVerse Lipsync",
    "UseapiPixverseMotionControl": "Useapi PixVerse Motion Control",
    "UseapiPixverseExtend": "Useapi PixVerse Extend Video",
}
