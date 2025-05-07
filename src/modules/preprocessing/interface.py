"""
Interface for data preprocessing modules.
"""
from abc import abstractmethod
from typing import Any, Dict, List, Optional, Union

import pandas as pd

from src.modules.base import DataTransformer


class PreprocessingInterface(DataTransformer):
    """
    Interface for preprocessing review data.
    """
    
    @abstractmethod
    def _validate_config(self) -> None:
        """
        Validate preprocessing configuration.
        
        Should check for:
        - Language settings
        - Text cleaning options
        - Tokenization settings
        
        Raises:
            ValueError: If configuration is invalid
        """
        pass
    
    @abstractmethod
    def clean_text(self, text: Union[str, List[str]]) -> Union[str, List[str]]:
        """
        Clean text by removing noise, special characters, etc.
        
        Args:
            text: Text or list of texts to clean
            
        Returns:
            Cleaned text or list of cleaned texts
        """
        pass
    
    @abstractmethod
    def normalize_text(self, text: Union[str, List[str]]) -> Union[str, List[str]]:
        """
        Normalize text through steps like lowercase, stemming, lemmatization.
        
        Args:
            text: Text or list of texts to normalize
            
        Returns:
            Normalized text or list of normalized texts
        """
        pass
    
    @abstractmethod
    def tokenize_text(self, text: Union[str, List[str]]) -> Union[List[str], List[List[str]]]:
        """
        Tokenize text into individual words or tokens.
        
        Args:
            text: Text or list of texts to tokenize
            
        Returns:
            Tokens or list of tokens for each text
        """
        pass
    
    @abstractmethod
    def remove_stopwords(self, tokens: Union[List[str], List[List[str]]]) -> Union[List[str], List[List[str]]]:
        """
        Remove stopwords from tokenized text.
        
        Args:
            tokens: Tokens or list of tokens to process
            
        Returns:
            Tokens with stopwords removed
        """
        pass
    
    @abstractmethod
    def process_data(self, data: pd.DataFrame, text_column: str = "text", **kwargs) -> pd.DataFrame:
        """
        Process the dataframe to clean and prepare text data.
        
        Args:
            data: DataFrame containing reviews
            text_column: Column containing text to process
            
        Returns:
            Processed DataFrame with additional columns for processed text
        """
        pass