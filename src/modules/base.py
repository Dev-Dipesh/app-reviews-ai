"""
Base module definitions that provide interfaces for all modules.
"""
from abc import ABC, abstractmethod
from typing import Any, Dict, List, Optional


class BaseModule(ABC):
    """
    Base class for all modules that enforces a consistent interface pattern.
    """
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """
        Initialize a module with optional configuration.
        
        Args:
            config: Configuration dictionary
        """
        self.config = config or {}
        self._validate_config()
        self.is_initialized = False
        
    @abstractmethod
    def _validate_config(self) -> None:
        """
        Validate the configuration provided.
        
        Raises:
            ValueError: If the configuration is invalid
        """
        pass
    
    @abstractmethod
    def initialize(self) -> None:
        """
        Initialize the module with the provided configuration.
        
        Raises:
            RuntimeError: If initialization fails
        """
        pass
    
    def is_ready(self) -> bool:
        """
        Check if the module is initialized and ready to use.
        
        Returns:
            bool: True if the module is ready, False otherwise
        """
        return self.is_initialized


class DataProvider(BaseModule):
    """
    Interface for modules that provide data.
    """
    
    @abstractmethod
    def get_data(self, **kwargs) -> Any:
        """
        Get data from the provider.
        
        Returns:
            Any: The data retrieved
        """
        pass


class DataConsumer(BaseModule):
    """
    Interface for modules that consume data.
    """
    
    @abstractmethod
    def process_data(self, data: Any, **kwargs) -> Any:
        """
        Process the provided data.
        
        Args:
            data: The data to process
            
        Returns:
            Any: The processed data
        """
        pass


class DataTransformer(DataProvider, DataConsumer):
    """
    Interface for modules that both consume and provide data.
    """
    pass