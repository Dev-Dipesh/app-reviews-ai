"""
Interface for data acquisition modules.
"""
from abc import abstractmethod
from datetime import datetime
from typing import Any, Dict, List, Optional, Union

import pandas as pd

from src.modules.base import DataProvider


class ReviewAcquisitionInterface(DataProvider):
    """
    Interface for acquiring app reviews from various sources.
    """
    
    @abstractmethod
    def _validate_config(self) -> None:
        """
        Validate acquisition configuration.
        
        Should check for:
        - App ID/package name
        - Time frame settings
        - API keys if applicable
        
        Raises:
            ValueError: If configuration is invalid
        """
        pass
    
    @abstractmethod
    def fetch_reviews(
        self,
        app_id: Optional[str] = None,
        start_date: Optional[Union[str, datetime]] = None,
        end_date: Optional[Union[str, datetime]] = None,
        max_reviews: Optional[int] = None,
        **kwargs
    ) -> pd.DataFrame:
        """
        Fetch reviews from the data source.
        
        Args:
            app_id: Application ID or package name
            start_date: Start date for review collection
            end_date: End date for review collection
            max_reviews: Maximum number of reviews to fetch
            **kwargs: Additional parameters specific to the data source
            
        Returns:
            DataFrame containing reviews with at least the following columns:
            - review_id: Unique identifier for the review
            - author: Author name or identifier
            - date: Date of the review
            - rating: Numerical rating (e.g., 1-5)
            - text: Review text content
            - version: App version (if available)
        """
        pass
    
    @abstractmethod
    def get_app_info(self, app_id: Optional[str] = None) -> Dict[str, Any]:
        """
        Get information about the app.
        
        Args:
            app_id: Application ID or package name
            
        Returns:
            Dictionary containing app information such as:
            - name: App name
            - developer: Developer name
            - category: App category
            - current_version: Current app version
            - total_reviews: Total number of reviews
            - average_rating: Average rating
        """
        pass
    
    def get_data(self, **kwargs) -> pd.DataFrame:
        """
        Implementation of DataProvider interface method.
        
        Returns:
            DataFrame containing reviews
        """
        return self.fetch_reviews(**kwargs)