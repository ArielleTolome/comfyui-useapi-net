"""Tests for UseapiVideoToFrames node."""
import sys
import os
import unittest
from unittest.mock import MagicMock, patch

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

# Mock heavy deps if not present
try:
    import torch
except ImportError:
    torch = MagicMock()
    sys.modules["torch"] = torch

try:
    import numpy as np
except ImportError:
    np = MagicMock()
    sys.modules["numpy"] = np

try:
    import cv2
except ImportError:
    cv2 = MagicMock()
    sys.modules["cv2"] = cv2

from useapi_nodes import UseapiVideoToFrames


class TestUseapiVideoToFramesContract(unittest.TestCase):
    """Verify the node satisfies the ComfyUI node contract."""

    def test_has_required_class_attributes(self):
        for attr in ("CATEGORY", "FUNCTION", "RETURN_TYPES", "RETURN_NAMES", "OUTPUT_NODE"):
            self.assertTrue(hasattr(UseapiVideoToFrames, attr), f"Missing {attr}")

    def test_output_node_is_true(self):
        self.assertTrue(UseapiVideoToFrames.OUTPUT_NODE)

    def test_return_types(self):
        self.assertEqual(UseapiVideoToFrames.RETURN_TYPES, ("IMAGE", "INT", "FLOAT"))

    def test_return_names(self):
        self.assertEqual(UseapiVideoToFrames.RETURN_NAMES, ("frames", "frame_count", "fps"))

    def test_return_types_names_same_length(self):
        self.assertEqual(len(UseapiVideoToFrames.RETURN_TYPES), len(UseapiVideoToFrames.RETURN_NAMES))

    def test_category(self):
        self.assertEqual(UseapiVideoToFrames.CATEGORY, "Useapi.net/Utils")

    def test_input_types_has_video_path(self):
        inputs = UseapiVideoToFrames.INPUT_TYPES()
        self.assertIn("video_path", inputs.get("required", {}))

    def test_input_types_has_optional_frame_controls(self):
        inputs = UseapiVideoToFrames.INPUT_TYPES()
        optional = inputs.get("optional", {})
        self.assertIn("max_frames", optional)
        self.assertIn("start_frame", optional)
        self.assertIn("frame_step", optional)


class TestUseapiVideoToFramesExecute(unittest.TestCase):
    """Test execute() with mocked cv2 and filesystem."""

    @patch("useapi_nodes._CV2_AVAILABLE", False)
    def test_raises_when_cv2_not_available(self):
        node = UseapiVideoToFrames()
        with self.assertRaises(RuntimeError) as ctx:
            node.execute(video_path="/fake/video.mp4")
        self.assertIn("opencv-python", str(ctx.exception))

    @patch("useapi_nodes._CV2_AVAILABLE", True)
    @patch("useapi_nodes._is_safe_path", return_value=False)
    def test_raises_on_unsafe_path(self, _):
        node = UseapiVideoToFrames()
        with self.assertRaises(ValueError) as ctx:
            node.execute(video_path="http://evil.com/video.mp4")
        self.assertIn("Security error", str(ctx.exception))

    @patch("useapi_nodes._CV2_AVAILABLE", True)
    @patch("useapi_nodes._is_safe_path", return_value=True)
    @patch("useapi_nodes.cv2")
    def test_raises_when_video_cannot_open(self, mock_cv2, _):
        cap = MagicMock()
        cap.isOpened.return_value = False
        mock_cv2.VideoCapture.return_value = cap
        node = UseapiVideoToFrames()
        with self.assertRaises(RuntimeError) as ctx:
            node.execute(video_path="/fake/video.mp4")
        self.assertIn("Cannot open", str(ctx.exception))

    @patch("useapi_nodes._CV2_AVAILABLE", True)
    @patch("useapi_nodes._is_safe_path", return_value=True)
    @patch("useapi_nodes.cv2")
    def test_raises_when_no_frames_extracted(self, mock_cv2, _):
        cap = MagicMock()
        cap.isOpened.return_value = True
        cap.get.return_value = 24.0
        cap.read.return_value = (False, None)  # no frames at all
        mock_cv2.VideoCapture.return_value = cap
        mock_cv2.CAP_PROP_FPS = 5
        mock_cv2.CAP_PROP_FRAME_COUNT = 7
        node = UseapiVideoToFrames()
        with self.assertRaises(RuntimeError):
            node.execute(video_path="/fake/video.mp4", start_frame=99999)


if __name__ == "__main__":
    unittest.main(verbosity=2)
