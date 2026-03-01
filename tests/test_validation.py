"""Validation tests for Useapi nodes to ensure error handling works as expected.
"""
import unittest
import sys
import os
from unittest import mock

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

# Mock heavy deps if not in ComfyUI env
try:
    import torch
    import numpy as np
    from PIL import Image
except ImportError:
    from unittest.mock import MagicMock
    torch = MagicMock()
    np = MagicMock()
    sys.modules["torch"] = torch
    sys.modules["numpy"] = np

try:
    import cv2
except ImportError:
    from unittest.mock import MagicMock
    sys.modules["cv2"] = MagicMock()

from useapi_nodes import UseapiVeoConcatenate

class TestVeoConcatenateValidation(unittest.TestCase):
    def test_insufficient_media_args(self):
        node = UseapiVeoConcatenate()

        # Case 1: All empty
        with self.assertRaisesRegex(ValueError, "at least 2 mediaGenerationIds required"):
            node.execute(
                media_1=" ",
                media_2=" ",
                api_token="dummy"
            )

        # Case 2: Only one valid
        with self.assertRaisesRegex(ValueError, "at least 2 mediaGenerationIds required"):
            node.execute(
                media_1="id1",
                media_2=" ",
                api_token="dummy"
            )

    @mock.patch('useapi_nodes._make_request')
    def test_valid_media_args(self, mock_make_request):
        # Configure mock to simulate network call attempt
        mock_make_request.side_effect = RuntimeError("Network call attempted")

        node = UseapiVeoConcatenate()

        # Should pass validation and hit the network mock
        with self.assertRaisesRegex(RuntimeError, "Network call attempted"):
            node.execute(
                media_1="id1",
                media_2="id2",
                api_token="dummy"
            )

class TestCheckStatusValidation(unittest.TestCase):
    def test_redact_token(self):
        from useapi_nodes import _check_status
        token = "secret_token_123"
        url = f"https://api.useapi.net/v1/runwayml/gen4/upscale?token={token}"
        body = b'{"error": {"message": "Invalid auth token: secret_token_123"}}'

        with self.assertRaisesRegex(RuntimeError, r"Unauthorized \(401\).*URL: https://api.useapi.net/v1/runwayml/gen4/upscale") as context:
            _check_status(401, body, url, "Test context", token)

        # Verify the actual exception message does not contain the token
        exc_str = str(context.exception)
        self.assertNotIn(token, exc_str)

    def test_redact_token_detail(self):
        from useapi_nodes import _check_status
        token = "secret_token_123"
        url = f"https://api.useapi.net/v1/runwayml/gen4/upscale?token={token}"
        body = b'{"error": "Invalid auth token: secret_token_123"}'

        # Test a generic status code (like 409) to ensure detail is printed
        with self.assertRaisesRegex(RuntimeError, r"HTTP 409 from https://api.useapi.net/v1/runwayml/gen4/upscale\.\nDetail: Invalid auth token: \*\*\*") as context:
            _check_status(409, body, url, "Test context", token)

        # Verify the actual exception message does not contain the token
        exc_str = str(context.exception)
        self.assertNotIn(token, exc_str)
        self.assertIn("***", exc_str)

    def test_safe_url(self):
        from useapi_nodes import _check_status
        token = "dummy_token"
        url = "https://api.useapi.net/v1/endpoint?query=sensitive_param#fragment"
        body = b'{"error": "Not Found"}'

        with self.assertRaisesRegex(RuntimeError, r"Not Found \(404\).*URL: https://api.useapi.net/v1/endpoint") as context:
            _check_status(404, body, url, "Test context", token)

        exc_str = str(context.exception)
        self.assertNotIn("sensitive_param", exc_str)
        self.assertNotIn("fragment", exc_str)

if __name__ == "__main__":
    unittest.main(verbosity=2)
