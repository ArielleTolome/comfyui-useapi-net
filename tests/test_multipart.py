"""Unit tests for _build_multipart helper function.
Runs without network access or heavy dependencies.
"""
import sys
import os
import unittest
from unittest.mock import MagicMock

# Add repo root to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

# Mock heavy deps if not in ComfyUI env
for mod in ["torch", "numpy", "PIL", "cv2"]:
    if mod not in sys.modules:
        sys.modules[mod] = MagicMock()

# Import target function
# We need to import after mocking
from useapi_nodes import _build_multipart


class TestMultipartBuilder(unittest.TestCase):
    def test_fields_only(self):
        fields = {"name": "test_value", "key": 123}
        files = {}
        body, content_type = _build_multipart(fields, files)

        self.assertIn("multipart/form-data; boundary=", content_type)
        boundary = content_type.split("boundary=")[1]

        body_str = body.decode()
        self.assertIn(f"--{boundary}", body_str)
        self.assertIn('Content-Disposition: form-data; name="name"', body_str)
        self.assertIn('test_value', body_str)
        self.assertIn('Content-Disposition: form-data; name="key"', body_str)
        self.assertIn('123', body_str)
        self.assertTrue(body_str.endswith(f"--{boundary}--\r\n"))

    def test_files_only(self):
        fields = {}
        files = {
            "file_field": ("test.txt", b"file_content", "text/plain")
        }
        body, content_type = _build_multipart(fields, files)

        self.assertIn("multipart/form-data; boundary=", content_type)
        boundary = content_type.split("boundary=")[1]

        body_str = body.decode()
        self.assertIn(f"--{boundary}", body_str)
        self.assertIn('Content-Disposition: form-data; name="file_field"; filename="test.txt"', body_str)
        self.assertIn('Content-Type: text/plain', body_str)
        self.assertIn('file_content', body_str)
        self.assertTrue(body_str.endswith(f"--{boundary}--\r\n"))

    def test_mixed_content(self):
        fields = {"description": "A test file"}
        files = {
            "document": ("doc.pdf", b"%PDF-1.4...", "application/pdf")
        }
        body, content_type = _build_multipart(fields, files)

        boundary = content_type.split("boundary=")[1]
        body_str = body.decode()

        # Check field
        self.assertIn(f'name="description"', body_str)
        self.assertIn("A test file", body_str)

        # Check file
        self.assertIn(f'name="document"; filename="doc.pdf"', body_str)
        self.assertIn("application/pdf", body_str)
        self.assertIn("%PDF-1.4...", body_str)

        # Check structure
        self.assertEqual(body_str.count(f"--{boundary}"), 3) # 2 parts + 1 end

    def test_empty(self):
        fields = {}
        files = {}
        body, content_type = _build_multipart(fields, files)

        self.assertIn("multipart/form-data; boundary=", content_type)
        boundary = content_type.split("boundary=")[1]

        # Should just be the closing boundary
        self.assertEqual(body, f"--{boundary}--\r\n".encode())

    def test_boundary_format(self):
        fields = {"a": 1}
        files = {}
        body, content_type = _build_multipart(fields, files)

        self.assertTrue(content_type.startswith("multipart/form-data; boundary="))
        boundary = content_type.split("boundary=")[1]
        self.assertTrue(boundary.startswith("----ComfyUIBoundary"))
        # UUID hex is 32 chars
        self.assertTrue(len(boundary) > 20)

if __name__ == "__main__":
    unittest.main()
