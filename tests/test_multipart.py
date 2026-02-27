"""Unit tests for the _build_multipart helper function in useapi_nodes.py.

This function manually constructs multipart/form-data bodies to avoid
heavy dependencies like `requests`. These tests ensure the binary structure,
boundaries, and headers are compliant with RFC 7578.
"""
import sys
import os
import unittest
import uuid

# Ensure we can import the module under test
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

# Mock dependencies to allow import in restricted environments
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

from useapi_nodes import _build_multipart


class TestMultipartBuilder(unittest.TestCase):
    def test_empty(self):
        """Test with no fields and no files."""
        # It should produce just the closing boundary.
        body, content_type = _build_multipart({}, {})

        self.assertIn("multipart/form-data; boundary=", content_type)
        boundary = content_type.split("boundary=")[1]

        # Expectation: --boundary--\r\n
        expected_body = f"--{boundary}--\r\n".encode()
        self.assertEqual(body, expected_body)

    def test_fields_only(self):
        """Test with only simple text fields."""
        fields = {"key1": "value1", "key2": "value2"}
        body, content_type = _build_multipart(fields, {})

        boundary = content_type.split("boundary=")[1]
        boundary_bytes = boundary.encode()

        # Split by boundary marker
        parts = body.split(b"--" + boundary_bytes)

        # Expected parts:
        # [0]: empty (preamble)
        # [1]: \r\n...key1...\r\n
        # [2]: \r\n...key2...\r\n
        # [3]: --\r\n (closing suffix)

        self.assertEqual(len(parts), 4)

        # Check part 1 headers and content
        self.assertIn(b'Content-Disposition: form-data; name="key1"', parts[1])
        self.assertIn(b'\r\n\r\nvalue1\r\n', parts[1])

        # Check part 2 headers and content
        self.assertIn(b'Content-Disposition: form-data; name="key2"', parts[2])
        self.assertIn(b'\r\n\r\nvalue2\r\n', parts[2])

    def test_files_only(self):
        """Test with only file uploads."""
        # files format: {"name": ("filename.ext", bytes_data, "mime/type")}
        file_content = b"\x89PNG\r\n\x1a\n"
        files = {
            "image": ("image.png", file_content, "image/png")
        }

        body, content_type = _build_multipart({}, files)
        boundary = content_type.split("boundary=")[1].encode()

        # Should contain the boundary
        self.assertIn(boundary, body)

        # Should contain headers
        expected_headers = (
            b'Content-Disposition: form-data; name="image"; filename="image.png"\r\n'
            b'Content-Type: image/png\r\n\r\n'
        )
        self.assertIn(expected_headers, body)

        # Should contain the file content
        self.assertIn(file_content, body)

        # Should end correctly
        self.assertTrue(body.endswith(f"--".encode() + boundary + b"--\r\n"))

    def test_mixed_content(self):
        """Test mixing fields and files."""
        fields = {"description": "A test file"}
        files = {"document": ("test.txt", b"Hello World", "text/plain")}

        body, content_type = _build_multipart(fields, files)
        boundary = content_type.split("boundary=")[1].encode()

        # Split by boundary to verify structure
        parts = body.split(b"--" + boundary)

        # Expected parts:
        # [0]: empty (preamble)
        # [1]: field part (\r\nHeaders\r\n\r\nValue\r\n)
        # [2]: file part (\r\nHeaders\r\n\r\nValue\r\n)
        # [3]: closing boundary suffix (--\r\n)

        self.assertEqual(len(parts), 4)

        # Check field part
        self.assertIn(b'name="description"', parts[1])
        self.assertIn(b'\r\n\r\nA test file\r\n', parts[1])

        # Check file part
        self.assertIn(b'name="document"', parts[2])
        self.assertIn(b'filename="test.txt"', parts[2])
        self.assertIn(b'Content-Type: text/plain', parts[2])
        self.assertIn(b'\r\n\r\nHello World\r\n', parts[2])

    def test_binary_integrity(self):
        """Ensure binary data isn't corrupted or encoded."""
        # Arbitrary binary data that is not valid UTF-8
        binary_data = b"\xff\x00\xfe\x01\xfd\x02"
        files = {"bin": ("data.bin", binary_data, "application/octet-stream")}

        body, _ = _build_multipart({}, files)

        # The exact byte sequence must be present
        self.assertIn(binary_data, body)

        # Ensure it's not accidentally string-encoded (e.g. via str(value).encode())
        # If it were stringified, b'\xff' would become something like b'\\xff' or error out
        # depending on encoding. The helper directly concatenates bytes for files.

    def test_boundary_consistency(self):
        """Verify the boundary in Content-Type matches the one used in the body."""
        body, content_type = _build_multipart({"a": "b"}, {})
        boundary = content_type.split("boundary=")[1]

        boundary_marker = f"--{boundary}".encode()
        closing_marker = f"--{boundary}--\r\n".encode()

        # Body starts with boundary
        self.assertTrue(body.startswith(boundary_marker))

        # Body ends with closing boundary
        self.assertTrue(body.endswith(closing_marker))


if __name__ == "__main__":
    unittest.main()
