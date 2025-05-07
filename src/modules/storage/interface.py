"""
Interface for data storage modules.
"""
from abc import abstractmethod
from typing import Any, Dict, List, Optional, Union

import pandas as pd

from src.modules.base import DataConsumer, DataProvider


class StorageInterface(DataConsumer, DataProvider):
    """
    Interface for storing and retrieving review data.
    """
    
    @abstractmethod
    def _validate_config(self) -> None:
        """
        Validate storage configuration.
        
        Should check for:
        - Storage type
        - Connection details
        - Authentication information if applicable
        
        Raises:
            ValueError: If configuration is invalid
        """
        pass
    
    @abstractmethod
    def store_data(self, data: pd.DataFrame, **kwargs) -> bool:
        """
        Store data in the storage system.
        
        Args:
            data: DataFrame containing data to store
            **kwargs: Additional parameters specific to the storage system
            
        Returns:
            True if successful, False otherwise
        """
        pass
    
    @abstractmethod
    def retrieve_data(self, **kwargs) -> pd.DataFrame:
        """
        Retrieve data from the storage system.
        
        Args:
            **kwargs: Parameters for data retrieval (e.g., filters)
            
        Returns:
            DataFrame containing retrieved data
        """
        pass
    
    @abstractmethod
    def update_data(self, data: pd.DataFrame, **kwargs) -> bool:
        """
        Update existing data in the storage system.
        
        Args:
            data: DataFrame containing updated data
            **kwargs: Additional parameters specific to the storage system
            
        Returns:
            True if successful, False otherwise
        """
        pass
    
    @abstractmethod
    def delete_data(self, **kwargs) -> bool:
        """
        Delete data from the storage system.
        
        Args:
            **kwargs: Parameters for data deletion (e.g., filters)
            
        Returns:
            True if successful, False otherwise
        """
        pass
    
    def process_data(self, data: pd.DataFrame, **kwargs) -> bool:
        """
        Implementation of DataConsumer interface method.
        
        Args:
            data: DataFrame to store
            
        Returns:
            True if successful, False otherwise
        """
        return self.store_data(data, **kwargs)
    
    def get_data(self, **kwargs) -> pd.DataFrame:
        """
        Implementation of DataProvider interface method.
        
        Returns:
            DataFrame containing retrieved data
        """
        return self.retrieve_data(**kwargs)