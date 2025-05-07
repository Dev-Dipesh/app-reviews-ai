"""
Interface for analytics modules.
"""
from abc import abstractmethod
from typing import Any, Dict, List, Optional, Tuple, Union

import pandas as pd

from src.modules.base import DataTransformer


class AnalyticsInterface(DataTransformer):
    """
    Interface for analyzing review data.
    """
    
    @abstractmethod
    def _validate_config(self) -> None:
        """
        Validate analytics configuration.
        
        Should check for:
        - Analyzer settings
        - Model parameters
        - Output formats
        
        Raises:
            ValueError: If configuration is invalid
        """
        pass
    
    @abstractmethod
    def analyze_sentiment(
        self, 
        data: pd.DataFrame, 
        text_column: str = "text",
        **kwargs
    ) -> pd.DataFrame:
        """
        Perform sentiment analysis on review text.
        
        Args:
            data: DataFrame containing reviews
            text_column: Column containing text to analyze
            
        Returns:
            DataFrame with added sentiment analysis columns
        """
        pass
    
    @abstractmethod
    def extract_topics(
        self, 
        data: pd.DataFrame, 
        text_column: str = "text",
        n_topics: int = 10,
        **kwargs
    ) -> Tuple[pd.DataFrame, Dict[int, List[str]]]:
        """
        Extract topics from review text.
        
        Args:
            data: DataFrame containing reviews
            text_column: Column containing text to analyze
            n_topics: Number of topics to extract
            
        Returns:
            Tuple containing:
              - DataFrame with topic assignments
              - Dictionary mapping topic IDs to representative words
        """
        pass
    
    @abstractmethod
    def cluster_reviews(
        self, 
        data: pd.DataFrame, 
        text_column: str = "text",
        n_clusters: int = 5,
        **kwargs
    ) -> pd.DataFrame:
        """
        Cluster reviews based on content similarity.
        
        Args:
            data: DataFrame containing reviews
            text_column: Column containing text to analyze
            n_clusters: Number of clusters to create
            
        Returns:
            DataFrame with cluster assignments
        """
        pass
    
    @abstractmethod
    def identify_trends(
        self, 
        data: pd.DataFrame, 
        date_column: str = "date",
        value_column: str = "rating", 
        freq: str = "M",
        **kwargs
    ) -> pd.DataFrame:
        """
        Identify trends in review data over time.
        
        Args:
            data: DataFrame containing reviews
            date_column: Column containing dates
            value_column: Column containing values to track
            freq: Frequency for resampling (D=daily, W=weekly, M=monthly)
            
        Returns:
            DataFrame with trend analysis
        """
        pass
    
    @abstractmethod
    def extract_keywords(
        self, 
        data: pd.DataFrame, 
        text_column: str = "text",
        n_keywords: int = 20,
        **kwargs
    ) -> pd.DataFrame:
        """
        Extract important keywords from review text.
        
        Args:
            data: DataFrame containing reviews
            text_column: Column containing text to analyze
            n_keywords: Number of keywords to extract
            
        Returns:
            DataFrame with keyword information
        """
        pass
    
    def process_data(
        self, 
        data: pd.DataFrame, 
        analysis_types: Optional[List[str]] = None,
        **kwargs
    ) -> Dict[str, Any]:
        """
        Process data with specified analysis types.
        
        Args:
            data: DataFrame containing reviews
            analysis_types: List of analysis types to perform
                            (sentiment, topics, clusters, trends, keywords)
            
        Returns:
            Dictionary containing analysis results for each requested type
        """
        if analysis_types is None:
            analysis_types = ["sentiment"]
        
        results = {}
        
        # Perform each requested analysis
        if "sentiment" in analysis_types:
            results["sentiment"] = self.analyze_sentiment(data, **kwargs)
        
        if "topics" in analysis_types:
            results["topics"] = self.extract_topics(data, **kwargs)
        
        if "clusters" in analysis_types:
            results["clusters"] = self.cluster_reviews(data, **kwargs)
        
        if "trends" in analysis_types:
            results["trends"] = self.identify_trends(data, **kwargs)
        
        if "keywords" in analysis_types:
            results["keywords"] = self.extract_keywords(data, **kwargs)
        
        return results