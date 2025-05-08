"""
File-based storage implementation for review data.
"""
import os
from typing import Any, Dict, List, Optional, Union

import pandas as pd

from src.config import config
from src.modules.storage.interface import StorageInterface


class FileStorage(StorageInterface):
    """
    Implementation of storage interface using CSV or Parquet files.
    """
    
    def __init__(self, config_override: Optional[Dict[str, Any]] = None):
        """
        Initialize file storage module.
        
        Args:
            config_override: Override for default configuration
        """
        # Initialize attributes with default values
        self._file_path = None
        self._file_format = "csv"
        
        # Call parent constructor
        super().__init__(config_override)
        
        # Set attributes from config after validation
        self._file_path = self.config.get("file_path", None)
        self._file_format = self.config.get("file_format", "csv").lower()
        
        # Ensure file format is valid
        if self._file_format not in ["csv", "parquet"]:
            self._file_format = "csv"  # Default to CSV if invalid
    
    def _validate_config(self) -> None:
        """
        Validate storage configuration.
        
        Raises:
            ValueError: If required configuration is missing
        """
        if not self._file_path:
            # If not in module config, check global config
            try:
                self._file_path = config.get("storage", "file_path")
            except KeyError:
                pass
            
            # Still no file_path, use default
            if not self._file_path:
                # Use absolute path for default
                project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "../.."))
                self._file_path = os.path.join(project_root, "data", "reviews.csv")  # Default path
                print(f"Using default file path: {self._file_path}")
        
        # Ensure file_path has a value and is absolute
        if not self._file_path:
            # Use absolute path for default
            project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "../.."))
            self._file_path = os.path.join(project_root, "data", "reviews.csv")
        
        # Ensure directory exists
        os.makedirs(os.path.dirname(self._file_path), exist_ok=True)
    
    def initialize(self) -> None:
        """
        Initialize storage module.
        
        Creates necessary directories and verifies file access.
        
        Raises:
            RuntimeError: If storage initialization fails
        """
        try:
            # Ensure file_path has a value and is absolute (this is a second check as a safeguard)
            if not self._file_path:
                # Use absolute path for default
                project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "../.."))
                self._file_path = os.path.join(project_root, "data", "reviews.csv")
                print(f"Initialize: Using default file path: {self._file_path}")
            
            # Create directory if it doesn't exist
            dir_path = os.path.dirname(self._file_path)
            if dir_path:  # Only if there's a directory component
                os.makedirs(dir_path, exist_ok=True)
            
            # Check if file exists and is writable
            if os.path.exists(self._file_path):
                # Attempt to open file for reading
                if self._file_format == "csv":
                    pd.read_csv(self._file_path, nrows=1)
                else:  # parquet
                    pd.read_parquet(self._file_path, engine="pyarrow")
            
            self.is_initialized = True
        except Exception as e:
            raise RuntimeError(f"Failed to initialize file storage: {e}")
    
    def store_data(self, data: pd.DataFrame, append: bool = False, **kwargs) -> bool:
        """
        Store data in file.
        
        Args:
            data: DataFrame containing data to store
            append: Whether to append to existing data
            **kwargs: Additional parameters passed to to_csv or to_parquet
            
        Returns:
            True if successful, False otherwise
        """
        try:
            if append and os.path.exists(self._file_path):
                if self._file_format == "csv":
                    # Read existing data
                    existing_data = pd.read_csv(self._file_path)
                    # Combine with new data
                    combined_data = pd.concat([existing_data, data], ignore_index=True)
                    # Write combined data
                    combined_data.to_csv(self._file_path, index=False, **kwargs)
                else:  # parquet
                    # Parquet doesn't support appending directly, so we need to read and combine
                    existing_data = pd.read_parquet(self._file_path, engine="pyarrow")
                    combined_data = pd.concat([existing_data, data], ignore_index=True)
                    combined_data.to_parquet(self._file_path, engine="pyarrow", **kwargs)
            else:
                # Write new data
                if self._file_format == "csv":
                    data.to_csv(self._file_path, index=False, **kwargs)
                else:  # parquet
                    data.to_parquet(self._file_path, engine="pyarrow", **kwargs)
            
            return True
        except Exception as e:
            print(f"Error storing data: {e}")
            return False
    
    def retrieve_data(self, filters: Optional[Dict[str, Any]] = None, **kwargs) -> pd.DataFrame:
        """
        Retrieve data from file.
        
        Args:
            filters: Dictionary of column-value pairs to filter data
            **kwargs: Additional parameters passed to read_csv or read_parquet
            
        Returns:
            DataFrame containing retrieved data
        """
        try:
            if not os.path.exists(self._file_path):
                # Return empty DataFrame if file doesn't exist
                return pd.DataFrame()
            
            # Read data
            if self._file_format == "csv":
                data = pd.read_csv(self._file_path, **kwargs)
            else:  # parquet
                data = pd.read_parquet(self._file_path, engine="pyarrow", **kwargs)
            
            # Apply filters if provided
            if filters:
                for column, value in filters.items():
                    if column in data.columns:
                        data = data[data[column] == value]
            
            return data
        except Exception as e:
            print(f"Error retrieving data: {e}")
            return pd.DataFrame()
    
    def update_data(self, data: pd.DataFrame, key_column: str = "review_id", **kwargs) -> bool:
        """
        Update existing data in file.
        
        Args:
            data: DataFrame containing updated data
            key_column: Column to use as unique identifier
            **kwargs: Additional parameters
            
        Returns:
            True if successful, False otherwise
        """
        try:
            if not os.path.exists(self._file_path):
                # If file doesn't exist, just store the data directly
                return self.store_data(data, **kwargs)
            
            # Read existing data
            if self._file_format == "csv":
                existing_data = pd.read_csv(self._file_path)
            else:  # parquet
                existing_data = pd.read_parquet(self._file_path, engine="pyarrow")
            
            # Ensure key column exists in both DataFrames
            if key_column not in existing_data.columns or key_column not in data.columns:
                print(f"Key column '{key_column}' not found in data")
                return False
            
            # Create a combined DataFrame
            # First, remove rows from existing_data that will be updated
            existing_data = existing_data[~existing_data[key_column].isin(data[key_column])]
            
            # Then concatenate with new data
            combined_data = pd.concat([existing_data, data], ignore_index=True)
            
            # Write combined data
            if self._file_format == "csv":
                combined_data.to_csv(self._file_path, index=False)
            else:  # parquet
                combined_data.to_parquet(self._file_path, engine="pyarrow")
            
            return True
        except Exception as e:
            print(f"Error updating data: {e}")
            return False
    
    def delete_data(self, filters: Optional[Dict[str, Any]] = None, **kwargs) -> bool:
        """
        Delete data from file.
        
        Args:
            filters: Dictionary of column-value pairs to filter data for deletion
            **kwargs: Additional parameters
            
        Returns:
            True if successful, False otherwise
        """
        try:
            if not os.path.exists(self._file_path):
                # Nothing to delete if file doesn't exist
                return True
            
            # Read existing data
            if self._file_format == "csv":
                data = pd.read_csv(self._file_path)
            else:  # parquet
                data = pd.read_parquet(self._file_path, engine="pyarrow")
            
            if filters:
                # Create a mask for rows to keep
                mask = pd.Series(True, index=data.index)
                
                for column, value in filters.items():
                    if column in data.columns:
                        mask = mask & (data[column] != value)
                
                # Filter the data
                filtered_data = data[mask]
                
                # Write filtered data back to file
                if self._file_format == "csv":
                    filtered_data.to_csv(self._file_path, index=False)
                else:  # parquet
                    filtered_data.to_parquet(self._file_path, engine="pyarrow")
            else:
                # If no filters provided, delete the entire file
                os.remove(self._file_path)
            
            return True
        except Exception as e:
            print(f"Error deleting data: {e}")
            return False