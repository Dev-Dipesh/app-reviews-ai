"""
Interface for LLM modules.
"""
from abc import abstractmethod
from typing import Any, Dict, List, Optional, Union

from src.modules.base import BaseModule


class LLMInterface(BaseModule):
    """
    Interface for large language model operations.
    """
    
    @abstractmethod
    def _validate_config(self) -> None:
        """
        Validate LLM configuration.
        
        Should check for:
        - API keys and authentication
        - Model configuration
        - Generation parameters
        
        Raises:
            ValueError: If configuration is invalid
        """
        pass
    
    @abstractmethod
    def generate_text(
        self,
        prompt: str,
        system_prompt: Optional[str] = None,
        max_tokens: Optional[int] = None,
        temperature: Optional[float] = None,
        **kwargs
    ) -> str:
        """
        Generate text using the LLM.
        
        Args:
            prompt: Text prompt for the LLM
            system_prompt: System prompt to guide the model
            max_tokens: Maximum number of tokens to generate
            temperature: Temperature for generation (0.0 to 1.0)
            
        Returns:
            Generated text
        """
        pass
    
    @abstractmethod
    def analyze_reviews(
        self,
        reviews: Union[List[Dict[str, Any]], List[str]],
        analysis_type: str = "general",
        context: Optional[str] = None,
        **kwargs
    ) -> Dict[str, Any]:
        """
        Analyze reviews using the LLM.
        
        Args:
            reviews: List of reviews to analyze
            analysis_type: Type of analysis to perform
                (general, sentiment, issues, suggestions, etc.)
            context: Additional context for the analysis
            
        Returns:
            Analysis results
        """
        pass
    
    @abstractmethod
    def summarize_text(
        self,
        text: str,
        max_length: Optional[int] = None,
        format_type: str = "paragraph",
        **kwargs
    ) -> str:
        """
        Summarize text using the LLM.
        
        Args:
            text: Text to summarize
            max_length: Maximum length of summary
            format_type: Format of summary (paragraph, bullets, etc.)
            
        Returns:
            Summarized text
        """
        pass
    
    @abstractmethod
    def extract_structured_data(
        self,
        text: str,
        schema: Dict[str, Any],
        **kwargs
    ) -> Dict[str, Any]:
        """
        Extract structured data from text using the LLM.
        
        Args:
            text: Text to extract data from
            schema: Schema defining the structure of the data to extract
            
        Returns:
            Extracted structured data
        """
        pass