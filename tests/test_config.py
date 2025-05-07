"""
Tests for configuration module.
"""
import os
import tempfile
import unittest
from unittest.mock import patch

import json

from src.config import ConfigManager


class TestConfigManager(unittest.TestCase):
    """Test the ConfigManager class."""

    def setUp(self):
        """Set up test environment."""
        # Create a temporary config file
        self.temp_config = tempfile.NamedTemporaryFile(delete=False, suffix=".json")
        self.temp_config.write(b'{"test": {"key": "value"}}')
        self.temp_config.close()

    def tearDown(self):
        """Clean up test environment."""
        os.unlink(self.temp_config.name)
        # Reset the singleton instance
        ConfigManager._instance = None

    def test_singleton(self):
        """Test that ConfigManager is a singleton."""
        config1 = ConfigManager()
        config2 = ConfigManager()
        self.assertIs(config1, config2)

    def test_get_config(self):
        """Test getting configuration values."""
        config = ConfigManager()
        
        # Test getting a section
        app_config = config.get("app")
        self.assertIsInstance(app_config, dict)
        self.assertIn("name", app_config)
        
        # Test getting a specific key
        app_name = config.get("app", "name")
        self.assertEqual(app_name, "app-reviews-ai")
        
        # Test getting the entire config
        full_config = config.get()
        self.assertIsInstance(full_config, dict)
        self.assertIn("app", full_config)

    def test_load_from_file(self):
        """Test loading configuration from a file."""
        with patch.object(ConfigManager, "_load_from_env"):
            config = ConfigManager()
            config._load_from_file(self.temp_config.name)
            
            # Check if values from the file were loaded
            self.assertEqual(config.get("test", "key"), "value")

    @patch.dict(os.environ, {"APP_REVIEWS_TEST__KEY": "env_value"})
    def test_load_from_env(self):
        """Test loading configuration from environment variables."""
        with patch.object(ConfigManager, "_load_from_file"):
            config = ConfigManager()
            config._config = {"test": {"key": "default"}}
            config._load_from_env()
            
            # Check if values from environment were loaded
            # Note: Env variable name changed from INDIGO_REVIEWS_TEST__KEY to APP_REVIEWS_TEST__KEY
            self.assertEqual(config.get("test", "key"), "env_value")

    def test_parse_env_value(self):
        """Test parsing environment variable values."""
        # Test parsing various types
        self.assertEqual(ConfigManager._parse_env_value("true"), True)
        self.assertEqual(ConfigManager._parse_env_value("false"), False)
        self.assertEqual(ConfigManager._parse_env_value("none"), None)
        self.assertEqual(ConfigManager._parse_env_value("42"), 42)
        self.assertEqual(ConfigManager._parse_env_value("3.14"), 3.14)
        self.assertEqual(ConfigManager._parse_env_value("hello"), "hello")
        
        # Test parsing JSON
        self.assertEqual(
            ConfigManager._parse_env_value('{"key": "value"}'),
            {"key": "value"}
        )

    def test_update_nested_dict(self):
        """Test updating nested dictionaries."""
        base = {"level1": {"level2": "old_value", "untouched": "value"}}
        update = {"level1": {"level2": "new_value"}}
        
        ConfigManager._update_nested_dict(base, update)
        
        self.assertEqual(base["level1"]["level2"], "new_value")
        self.assertEqual(base["level1"]["untouched"], "value")


if __name__ == "__main__":
    unittest.main()