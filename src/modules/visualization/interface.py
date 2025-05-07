"""
Interface for visualization modules.
"""
from abc import abstractmethod
from typing import Any, Dict, List, Optional, Union

import pandas as pd

from src.modules.base import DataConsumer


class VisualizationInterface(DataConsumer):
    """
    Interface for visualizing review data.
    """
    
    @abstractmethod
    def _validate_config(self) -> None:
        """
        Validate visualization configuration.
        
        Should check for:
        - Output directory
        - Theme settings
        - Export formats
        
        Raises:
            ValueError: If configuration is invalid
        """
        pass
    
    @abstractmethod
    def plot_rating_distribution(
        self,
        data: pd.DataFrame,
        rating_column: str = "rating",
        title: Optional[str] = None,
        **kwargs
    ) -> Dict[str, Any]:
        """
        Create a rating distribution visualization.
        
        Args:
            data: DataFrame containing reviews
            rating_column: Column containing ratings
            title: Plot title
            
        Returns:
            Dictionary with visualization information
        """
        pass
    
    @abstractmethod
    def plot_rating_trend(
        self,
        data: pd.DataFrame,
        rating_column: str = "rating",
        date_column: str = "date",
        freq: str = "M",
        title: Optional[str] = None,
        **kwargs
    ) -> Dict[str, Any]:
        """
        Create a rating trend visualization.
        
        Args:
            data: DataFrame containing reviews
            rating_column: Column containing ratings
            date_column: Column containing dates
            freq: Frequency for resampling (D=daily, W=weekly, M=monthly)
            title: Plot title
            
        Returns:
            Dictionary with visualization information
        """
        pass
    
    @abstractmethod
    def plot_sentiment_distribution(
        self,
        data: pd.DataFrame,
        sentiment_column: str = "sentiment",
        title: Optional[str] = None,
        **kwargs
    ) -> Dict[str, Any]:
        """
        Create a sentiment distribution visualization.
        
        Args:
            data: DataFrame containing reviews
            sentiment_column: Column containing sentiment values
            title: Plot title
            
        Returns:
            Dictionary with visualization information
        """
        pass
    
    @abstractmethod
    def plot_word_cloud(
        self,
        data: pd.DataFrame,
        text_column: str = "text",
        title: Optional[str] = None,
        **kwargs
    ) -> Dict[str, Any]:
        """
        Create a word cloud visualization.
        
        Args:
            data: DataFrame containing reviews
            text_column: Column containing text
            title: Plot title
            
        Returns:
            Dictionary with visualization information
        """
        pass
    
    @abstractmethod
    def plot_topic_distribution(
        self,
        data: pd.DataFrame,
        topic_column: str = "primary_topic",
        topic_words: Optional[Dict[int, List[str]]] = None,
        title: Optional[str] = None,
        **kwargs
    ) -> Dict[str, Any]:
        """
        Create a topic distribution visualization.
        
        Args:
            data: DataFrame containing reviews
            topic_column: Column containing topic assignments
            topic_words: Dictionary mapping topic IDs to representative words
            title: Plot title
            
        Returns:
            Dictionary with visualization information
        """
        pass
    
    @abstractmethod
    def create_dashboard(
        self,
        data: pd.DataFrame,
        output_path: Optional[str] = None,
        include_plots: Optional[List[str]] = None,
        title: str = "Review Analysis Dashboard",
        **kwargs
    ) -> Dict[str, Any]:
        """
        Create a dashboard with multiple visualizations.
        
        Args:
            data: DataFrame containing reviews
            output_path: Path to save the dashboard
            include_plots: List of plots to include
            title: Dashboard title
            
        Returns:
            Dictionary with dashboard information
        """
        pass
    
    def process_data(
        self, 
        data: pd.DataFrame,
        visualization_type: str = "dashboard",
        **kwargs
    ) -> Dict[str, Any]:
        """
        Implementation of DataConsumer interface method.
        
        Args:
            data: DataFrame to visualize
            visualization_type: Type of visualization to create
            
        Returns:
            Dictionary with visualization results
        """
        if visualization_type == "dashboard":
            return self.create_dashboard(data, **kwargs)
        elif visualization_type == "rating_distribution":
            return self.plot_rating_distribution(data, **kwargs)
        elif visualization_type == "rating_trend":
            return self.plot_rating_trend(data, **kwargs)
        elif visualization_type == "sentiment_distribution":
            return self.plot_sentiment_distribution(data, **kwargs)
        elif visualization_type == "word_cloud":
            return self.plot_word_cloud(data, **kwargs)
        elif visualization_type == "topic_distribution":
            return self.plot_topic_distribution(data, **kwargs)
        else:
            raise ValueError(f"Unsupported visualization type: {visualization_type}")