"""
Implementation of NLP-based preprocessing module.
"""
import re
import string
from typing import Any, Dict, List, Optional, Union

import nltk
import pandas as pd
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer, WordNetLemmatizer
from nltk.tokenize import word_tokenize

from src.config import config
from src.modules.preprocessing.interface import PreprocessingInterface


class NLPPreprocessor(PreprocessingInterface):
    """
    Implementation of preprocessing module using NLTK.
    """
    
    def __init__(self, config_override: Optional[Dict[str, Any]] = None):
        """
        Initialize the NLP preprocessor.
        
        Args:
            config_override: Override for default configuration
        """
        # Initialize attributes with defaults
        self._language = "english"
        self._remove_stopwords_flag = True
        self._lemmatize_flag = True
        self._stem_flag = False
        self._min_word_length = 3
        self._stopwords = None
        self._lemmatizer = None
        self._stemmer = None
        
        # Call parent constructor
        super().__init__(config_override)
        
        # Set attributes from config after validation
        self._language = self.config.get("language", "english")
        self._remove_stopwords_flag = self.config.get("remove_stopwords", True)
        self._lemmatize_flag = self.config.get("lemmatize", True)
        self._stem_flag = self.config.get("stem", False)
        self._min_word_length = self.config.get("min_word_length", 3)
    
    def _validate_config(self) -> None:
        """
        Validate preprocessing configuration.
        
        Raises:
            ValueError: If required configuration is missing
        """
        # Convert language code to NLTK format if needed
        if self._language == "en":
            self._language = "english"
        
        # Check if language is supported
        supported_languages = [
            "danish", "dutch", "english", "finnish", "french", "german",
            "hungarian", "italian", "norwegian", "portuguese", "russian",
            "spanish", "swedish", "turkish"
        ]
        
        if self._language not in supported_languages:
            raise ValueError(f"Language '{self._language}' is not supported")
    
    def initialize(self) -> None:
        """
        Initialize the preprocessing module.
        
        Initializes NLTK components without downloading resources again (they're downloaded at startup).
        
        Raises:
            RuntimeError: If initialization fails
        """
        try:
            # Initialize components using the already-downloaded resources
            # The resources are downloaded at application startup in initialize_resources.py
            
            # Initialize stopwords
            try:
                self._stopwords = set(stopwords.words(self._language))
            except LookupError as e:
                print(f"Warning: Could not load stopwords for {self._language}: {e}")
                self._stopwords = set()  # Use empty set as fallback
            
            # Initialize lemmatizer if needed
            if self._lemmatize_flag:
                self._lemmatizer = WordNetLemmatizer()
            
            # Initialize stemmer if needed
            if self._stem_flag:
                self._stemmer = PorterStemmer()
            
            self.is_initialized = True
        except Exception as e:
            raise RuntimeError(f"Failed to initialize NLP preprocessor: {e}")
    
    def clean_text(self, text: Union[str, List[str]]) -> Union[str, List[str]]:
        """
        Clean text by removing special characters, HTML tags, etc.
        
        Args:
            text: Text or list of texts to clean
            
        Returns:
            Cleaned text or list of cleaned texts
        """
        if isinstance(text, list):
            return [self.clean_text(t) for t in text]
        
        if not isinstance(text, str):
            return ""
        
        # Convert to lowercase
        text = text.lower()
        
        # Remove HTML tags
        text = re.sub(r"<.*?>", "", text)
        
        # Remove URLs
        text = re.sub(r"http\S+|www\S+|https\S+", "", text)
        
        # Remove email addresses
        text = re.sub(r"\S+@\S+", "", text)
        
        # Remove special characters and numbers
        text = re.sub(r"[^\w\s]", "", text)
        text = re.sub(r"\d+", "", text)
        
        # Remove extra whitespace
        text = re.sub(r"\s+", " ", text).strip()
        
        return text
    
    def normalize_text(self, text: Union[str, List[str]]) -> Union[str, List[str]]:
        """
        Normalize text through steps like lowercase, stemming, lemmatization.
        
        Args:
            text: Text or list of texts to normalize
            
        Returns:
            Normalized text or list of normalized texts
        """
        if isinstance(text, list):
            return [self.normalize_text(t) for t in text]
        
        if not isinstance(text, str):
            return ""
        
        # Tokenize
        tokens = word_tokenize(text)
        
        # Remove stopwords if enabled
        if self._remove_stopwords_flag:
            tokens = [token for token in tokens if token not in self._stopwords]
        
        # Apply lemmatization if enabled
        if self._lemmatize_flag and self._lemmatizer:
            tokens = [self._lemmatizer.lemmatize(token) for token in tokens]
        
        # Apply stemming if enabled
        if self._stem_flag and self._stemmer:
            tokens = [self._stemmer.stem(token) for token in tokens]
        
        # Filter short words
        tokens = [token for token in tokens if len(token) >= self._min_word_length]
        
        # Join tokens back into text
        return " ".join(tokens)
    
    def tokenize_text(self, text: Union[str, List[str]]) -> Union[List[str], List[List[str]]]:
        """
        Tokenize text into individual words or tokens.
        
        Args:
            text: Text or list of texts to tokenize
            
        Returns:
            Tokens or list of tokens for each text
        """
        if isinstance(text, list):
            return [self.tokenize_text(t) for t in text]
        
        if not isinstance(text, str):
            return []
        
        # Tokenize text
        return word_tokenize(text)
    
    def remove_stopwords(self, tokens: Union[List[str], List[List[str]]]) -> Union[List[str], List[List[str]]]:
        """
        Remove stopwords from tokenized text.
        
        Args:
            tokens: Tokens or list of tokens to process
            
        Returns:
            Tokens with stopwords removed
        """
        if not tokens:
            return []
        
        if isinstance(tokens[0], list):
            return [self.remove_stopwords(t) for t in tokens]
        
        # Remove stopwords
        return [token for token in tokens if token.lower() not in self._stopwords]
    
    def process_data(self, data: pd.DataFrame, text_column: str = "text", **kwargs) -> pd.DataFrame:
        """
        Process the dataframe to clean and prepare text data.
        
        Args:
            data: DataFrame containing reviews
            text_column: Column containing text to process
            
        Returns:
            Processed DataFrame with additional columns for processed text
        """
        if text_column not in data.columns:
            raise ValueError(f"Text column '{text_column}' not found in data")
        
        # Create a copy to avoid modifying the original
        result = data.copy()
        
        # Apply preprocessing steps
        result["cleaned_text"] = result[text_column].apply(self.clean_text)
        result["normalized_text"] = result["cleaned_text"].apply(self.normalize_text)
        
        # Optionally add tokenized columns
        if kwargs.get("include_tokens", False):
            result["tokens"] = result["normalized_text"].apply(self.tokenize_text)
        
        return result
    
    def get_data(self, **kwargs) -> pd.DataFrame:
        """
        Implementation of DataProvider interface method.
        
        Not applicable for this module, as it requires input data.
        
        Raises:
            NotImplementedError: This method is not supported
        """
        raise NotImplementedError("PreprocessingModule does not support get_data() without input")