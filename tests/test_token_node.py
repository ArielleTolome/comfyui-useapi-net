import sys
import os
import unittest
from unittest.mock import patch, MagicMock

# Set up module path to import from parent directory
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

# Mock heavy deps if not in ComfyUI env
try:
    import torch
    import numpy as np
    from PIL import Image
except ImportError:
    torch = MagicMock()
    np = MagicMock()
    sys.modules["torch"] = torch
    sys.modules["numpy"] = np
    # PIL is used in useapi_nodes, so we might need to mock it if not present
    sys.modules["PIL"] = MagicMock()
    sys.modules["PIL.Image"] = MagicMock()

try:
    import cv2
except ImportError:
    sys.modules["cv2"] = MagicMock()

# Import the node class to be tested
from useapi_nodes import UseapiTokenFromEnv


class TestUseapiTokenFromEnv(unittest.TestCase):
    def setUp(self):
        self.node = UseapiTokenFromEnv()
        self.env_var_name = "TEST_USEAPI_TOKEN"

    def test_execute_with_valid_token(self):
        """Test that execute returns the token when the environment variable is set."""
        expected_token = "valid_token_123"
        with patch.dict(os.environ, {self.env_var_name: expected_token}):
            result = self.node.execute(self.env_var_name)
            self.assertEqual(result, (expected_token,))

    def test_execute_strips_whitespace(self):
        """Test that execute strips leading/trailing whitespace from the token."""
        raw_token = "  valid_token_123  "
        expected_token = "valid_token_123"
        with patch.dict(os.environ, {self.env_var_name: raw_token}):
            result = self.node.execute(self.env_var_name)
            self.assertEqual(result, (expected_token,))

    def test_execute_raises_when_env_var_missing(self):
        """Test that execute raises ValueError when the environment variable is missing."""
        # Ensure the variable is NOT in the environment
        with patch.dict(os.environ, clear=True):
            # We might need to keep PATH or other basics, but clear=True removes all.
            # To be safer, we just ensure our specific var is not there.
            if self.env_var_name in os.environ:
                del os.environ[self.env_var_name]

            with self.assertRaises(ValueError) as cm:
                self.node.execute(self.env_var_name)
            self.assertIn(f"Environment variable '{self.env_var_name}' is not set", str(cm.exception))

    def test_execute_raises_when_env_var_empty(self):
        """Test that execute raises ValueError when the environment variable is empty."""
        with patch.dict(os.environ, {self.env_var_name: ""}):
            with self.assertRaises(ValueError) as cm:
                self.node.execute(self.env_var_name)
            self.assertIn(f"Environment variable '{self.env_var_name}' is not set or empty", str(cm.exception))

    def test_execute_raises_when_env_var_whitespace_only(self):
        """Test that execute raises ValueError when the environment variable is only whitespace."""
        with patch.dict(os.environ, {self.env_var_name: "   "}):
            with self.assertRaises(ValueError) as cm:
                self.node.execute(self.env_var_name)
            self.assertIn(f"Environment variable '{self.env_var_name}' is not set or empty", str(cm.exception))

if __name__ == "__main__":
    unittest.main()
