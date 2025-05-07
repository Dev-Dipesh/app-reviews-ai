"""
Configuration management for the application.
"""
import json
import os
from typing import Any, Dict, Optional

import dotenv


class ConfigManager:
    """
    Configuration manager to handle all application settings.
    """
    
    _instance = None
    
    def __new__(cls):
        """Implement singleton pattern."""
        if cls._instance is None:
            cls._instance = super(ConfigManager, cls).__new__(cls)
            cls._instance._initialized = False
        return cls._instance
    
    def __init__(self):
        """Initialize configuration manager."""
        if self._initialized:
            return
            
        # Load environment variables
        dotenv.load_dotenv()
        
        # Default configuration
        self._config = {
            "app": {
                "name": "app-reviews-ai",
                "version": "0.1.0",
            },
            "acquisition": {
                "app_id": os.environ.get("APP_ID", ""),  # App ID from environment
                "time_frame": {
                    "start_date": os.environ.get("START_DATE", "1 year ago"),  # Reviews from last year
                    "end_date": os.environ.get("END_DATE", "now"),
                },
                "max_reviews": int(os.environ.get("MAX_REVIEWS", "1000")),  # Maximum number of reviews to fetch
            },
            "storage": {
                "db_type": "file",  # Options: file, sqlite, mongodb
                "connection_string": None,
                "file_path": "data/reviews.csv",
            },
            "preprocessing": {
                "language": "en",
                "remove_stopwords": True,
                "lemmatize": True,
            },
            "analytics": {
                "sentiment_analyzer": "nltk_vader",  # Options: nltk_vader, textblob
                "topic_modeling": {
                    "method": "lda",  # Options: lda, nmf
                    "num_topics": 10,
                },
            },
            "vector_db": {
                "engine": "chroma",  # Options: chroma, qdrant
                "embedding_model": "openai",  # Options: openai, local, sentence_transformer
                "collection_name": "app_reviews",
            },
            "llm": {
                "provider": "openai",  # Options: openai, cohere
                "model": "gpt-4o",
                "temperature": 0.3,
            },
            "visualization": {
                "theme": "dark",
                "export_formats": ["png", "html"],
            },
        }
        
        # Override with config file if exists
        self._load_from_file()
        
        # Override with environment variables
        self._load_from_env()
        
        self._initialized = True
    
    def _load_from_file(self, file_path: str = "config/config.json") -> None:
        """
        Load configuration from a JSON file.
        
        Args:
            file_path: Path to the config file
        """
        if os.path.exists(file_path):
            try:
                with open(file_path, 'r') as f:
                    file_config = json.load(f)
                    self._update_nested_dict(self._config, file_config)
            except (json.JSONDecodeError, IOError) as e:
                print(f"Error loading config file: {e}")
    
    def _load_from_env(self) -> None:
        """Load configuration from environment variables."""
        # Check for direct APP_ID environment variable - highest priority
        if "APP_ID" in os.environ:
            app_id = os.environ["APP_ID"]
            print(f"DEBUG: Applying APP_ID from environment: '{app_id}'")
            self._config["acquisition"]["app_id"] = app_id
        
        # Process other environment variables with prefix APP_REVIEWS_
        prefix = "APP_REVIEWS_"
        for key, value in os.environ.items():
            if key.startswith(prefix):
                # Convert APP_REVIEWS_SECTION__KEY to config['section']['key']
                parts = key[len(prefix):].lower().split('__')
                
                if len(parts) == 1:
                    # Top-level config
                    self._config[parts[0]] = self._parse_env_value(value)
                elif len(parts) == 2:
                    # Nested config
                    section, key = parts
                    if section in self._config:
                        self._config[section][key] = self._parse_env_value(value)
    
    @staticmethod
    def _parse_env_value(value: str) -> Any:
        """
        Parse environment variable values to appropriate types.
        
        Args:
            value: String value from environment variable
            
        Returns:
            Parsed value with appropriate type
        """
        # Try to parse as JSON first (for complex types)
        try:
            return json.loads(value)
        except json.JSONDecodeError:
            # Handle booleans
            if value.lower() in ('true', 'yes', '1'):
                return True
            if value.lower() in ('false', 'no', '0'):
                return False
            
            # Handle None
            if value.lower() in ('none', 'null'):
                return None
            
            # Try to parse as number
            try:
                if '.' in value:
                    return float(value)
                else:
                    return int(value)
            except ValueError:
                # Return as string
                return value
    
    @staticmethod
    def _update_nested_dict(base_dict: Dict, update_dict: Dict) -> None:
        """
        Update a nested dictionary without completely overwriting nested structures.
        
        Args:
            base_dict: Base dictionary to update
            update_dict: Dictionary with updates
        """
        for key, value in update_dict.items():
            if (
                key in base_dict and 
                isinstance(base_dict[key], dict) and 
                isinstance(value, dict)
            ):
                ConfigManager._update_nested_dict(base_dict[key], value)
            else:
                base_dict[key] = value
    
    def get(self, section: Optional[str] = None, key: Optional[str] = None) -> Any:
        """
        Get configuration value.
        
        Args:
            section: Configuration section
            key: Configuration key within section
            
        Returns:
            Configuration value
            
        Raises:
            KeyError: If section or key not found
        """
        if section is None:
            return self._config
        
        if section not in self._config:
            raise KeyError(f"Config section '{section}' not found")
        
        if key is None:
            return self._config[section]
        
        if key not in self._config[section]:
            raise KeyError(f"Config key '{key}' not found in section '{section}'")
        
        return self._config[section][key]
    
    def save(self, file_path: str = "config/config.json") -> None:
        """
        Save current configuration to a file.
        
        Args:
            file_path: Path to save the config file
        """
        # Create directory if it doesn't exist
        os.makedirs(os.path.dirname(file_path), exist_ok=True)
        
        try:
            with open(file_path, 'w') as f:
                json.dump(self._config, f, indent=2)
        except IOError as e:
            print(f"Error saving config file: {e}")


# Create a singleton instance
config = ConfigManager()