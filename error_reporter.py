"""
error_reporter.py - Drop-in mixin + decorator for ComfyUI custom nodes.
Catches exceptions in execute() (or any FUNCTION) and reports them as GitHub issues.

Usage:
    from error_reporter import ErrorReporterMixin

    class MyNode(ErrorReporterMixin):
        CATEGORY = "mypack"
        RETURN_TYPES = ("IMAGE",)
        FUNCTION = "execute"

        def execute(self, image, prompt):
            ...  # exceptions auto-reported to GitHub

Or use the decorator:
    from error_reporter import report_errors

    class MyNode:
        @report_errors
        def execute(self, image, prompt):
            ...
"""
import os
import sys
import json
import traceback
import hashlib
import subprocess
import functools
from datetime import datetime, timezone

try:
    import requests
    _HAS_REQUESTS = True
except ImportError:
    _HAS_REQUESTS = False

GITHUB_ERROR_TOKEN = os.environ.get("GITHUB_ERROR_TOKEN", "")
ENV_LABEL = os.environ.get("WATCHER_ENV", "production")
HTTP_TIMEOUT = 8


def _detect_github_repo():
    """Auto-detect GitHub repo (owner/repo) from git remote origin URL."""
    try:
        r = subprocess.run(
            ["git", "remote", "get-url", "origin"],
            capture_output=True, text=True, timeout=3,
            cwd=os.path.dirname(os.path.abspath(__file__))
        )
        url = r.stdout.strip()
        if not url:
            return ""
        # SSH: git@github.com:owner/repo.git
        if url.startswith("git@"):
            path = url.split(":", 1)[-1]
        else:
            # HTTPS: https://github.com/owner/repo.git
            from urllib.parse import urlparse
            path = urlparse(url).path.lstrip("/")
        if path.endswith(".git"):
            path = path[:-4]
        return path  # "owner/repo"
    except Exception:
        return ""


GITHUB_REPO = _detect_github_repo()


def _git_hash():
    try:
        r = subprocess.run(
            ["git", "rev-parse", "--short", "HEAD"],
            capture_output=True, text=True, timeout=3,
            cwd=os.path.dirname(os.path.abspath(__file__))
        )
        return r.stdout.strip() or "unknown"
    except Exception:
        return "unknown"


_GIT_HASH = _git_hash()


def _fingerprint(tb):
    import re
    normalized = re.sub(r", line \d+", ", line N", tb)
    return hashlib.md5(normalized.encode()).hexdigest()[:8]


def _create_github_issue(payload):
    """Create a GitHub issue for the error."""
    if not GITHUB_ERROR_TOKEN or GITHUB_ERROR_TOKEN == "disabled-for-test":
        print(
            f"[error_reporter] GITHUB_ERROR_TOKEN not set or disabled — skipping issue creation",
            file=sys.stderr,
        )
        return
    if not GITHUB_REPO:
        print("[error_reporter] Could not detect GITHUB_REPO from git remote", file=sys.stderr)
        return
    if not _HAS_REQUESTS:
        print("[error_reporter] pip install requests to enable GitHub issue reporting", file=sys.stderr)
        return

    title = (
        f"[{ENV_LABEL}] {payload['error_type']}: "
        f"{str(payload['error_message'])[:80]} "
        f"[{payload['error_fingerprint']}]"
    )
    body_lines = [
        f"**Node:** `{payload['node_class']}`",
        f"**Environment:** `{payload['environment']}`",
        f"**Git hash:** `{payload['git_hash']}`",
        f"**Fingerprint:** `{payload['error_fingerprint']}`",
        f"**Timestamp:** {payload['timestamp']}",
        f"**Error type:** `{payload['error_type']}`",
        f"**Error message:** {payload['error_message']}",
        "",
        "**Inputs snapshot:**",
        "```json",
        json.dumps(payload.get("inputs_snapshot", {}), indent=2),
        "```",
        "",
        "**Traceback:**",
        "```",
        payload["traceback"],
        "```",
    ]
    issue_body = "\n".join(body_lines)

    url = f"https://api.github.com/repos/{GITHUB_REPO}/issues"
    headers = {
        "Authorization": f"Bearer {GITHUB_ERROR_TOKEN}",
        "Accept": "application/vnd.github+json",
        "Content-Type": "application/json",
        "X-GitHub-Api-Version": "2022-11-28",
    }
    issue_data = {
        "title": title,
        "body": issue_body,
        "labels": ["bug", "auto-reported"],
    }
    try:
        r = requests.post(url, json=issue_data, headers=headers, timeout=HTTP_TIMEOUT)
        r.raise_for_status()
        issue = r.json()
        print(f"[error_reporter] GitHub issue created: {issue.get('html_url', '?')}", file=sys.stderr)
    except Exception as e:
        print(f"[error_reporter] Failed to create GitHub issue: {e}", file=sys.stderr)


def _build_payload(exc, node_class=None, node_file=None, inputs=None):
    tb_text = traceback.format_exc()
    safe_inputs = {}
    for k, v in (inputs or {}).items():
        if isinstance(v, (str, int, float, bool)) or v is None:
            safe_inputs[k] = v
        else:
            safe_inputs[k] = f"<{type(v).__name__}>"
    return {
        "source": "comfyui_node_execute",
        "environment": ENV_LABEL,
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "git_hash": _GIT_HASH,
        "error_fingerprint": _fingerprint(tb_text),
        "error_type": type(exc).__name__,
        "error_message": str(exc),
        "node_class": node_class,
        "node_file": node_file,
        "traceback": tb_text,
        "inputs_snapshot": safe_inputs,
    }


class ErrorReporterMixin:
    def __init_subclass__(cls, **kwargs):
        super().__init_subclass__(**kwargs)
        fn_name = getattr(cls, "FUNCTION", "execute")
        original = getattr(cls, fn_name, None)
        if original and callable(original):
            @functools.wraps(original)
            def wrapped(self_inner, *args, **kwargs):
                try:
                    return original(self_inner, *args, **kwargs)
                except Exception as exc:
                    inputs = {}
                    try:
                        it = cls.INPUT_TYPES()
                        keys = (
                            list(it.get("required", {}).keys())
                            + list(it.get("optional", {}).keys())
                        )
                        inputs = dict(zip(keys, args))
                        inputs.update(kwargs)
                    except Exception:
                        pass
                    payload = _build_payload(
                        exc,
                        node_class=cls.__name__,
                        node_file=getattr(
                            sys.modules.get(cls.__module__), "__file__", None
                        ),
                        inputs=inputs,
                    )
                    _create_github_issue(payload)
                    raise
            setattr(cls, fn_name, wrapped)


def report_errors(fn):
    @functools.wraps(fn)
    def wrapper(self, *args, **kwargs):
        try:
            return fn(self, *args, **kwargs)
        except Exception as exc:
            payload = _build_payload(
                exc,
                node_class=type(self).__name__,
                node_file=getattr(
                    sys.modules.get(type(self).__module__), "__file__", None
                ),
            )
            _create_github_issue(payload)
            raise
    return wrapper
