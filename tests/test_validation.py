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

from useapi_nodes import UseapiVeoConcatenate, UseapiVeoGenerate

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

class TestVeoGenerateValidation(unittest.TestCase):
    @mock.patch('useapi_nodes._make_request')
    def test_veo_generate_media_fallback(self, mock_make_request):
        import json

        # Test modern media array parsing
        mock_make_request.return_value = (200, json.dumps({
            "media": [{
                "mediaMetadata": {
                    "mediaStatus": {
                        "mediaGenerationStatus": "SUCCESS"
                    }
                },
                "videoUrl": "http://example.com/video1.mp4",
                "mediaGenerationId": "gen_id_1"
            }]
        }).encode())

        node = UseapiVeoGenerate()
        # Should not raise exception
        # We mock _download_file to avoid actual downloads
        with mock.patch('useapi_nodes._download_file', return_value="local_path.mp4"):
            video_url, video_path, media_gen_id = node.execute(
                prompt="test", model="veo-3.1-fast", aspect_ratio="landscape", api_token="dummy"
            )
            self.assertEqual(video_url, "http://example.com/video1.mp4")
            self.assertEqual(media_gen_id, "gen_id_1")

        # Test legacy operations array parsing fallback
        mock_make_request.return_value = (200, json.dumps({
            "operations": [{
                "status": "SUCCESS",
                "operation": {
                    "metadata": {
                        "video": {
                            "fifeUrl": "http://example.com/video2.mp4",
                            "mediaGenerationId": "gen_id_2"
                        }
                    }
                }
            }]
        }).encode())

        with mock.patch('useapi_nodes._download_file', return_value="local_path.mp4"):
            video_url, video_path, media_gen_id = node.execute(
                prompt="test", model="veo-3.1-fast", aspect_ratio="landscape", api_token="dummy"
            )
            self.assertEqual(video_url, "http://example.com/video2.mp4")
            self.assertEqual(media_gen_id, "gen_id_2")

if __name__ == "__main__":
    unittest.main(verbosity=2)
