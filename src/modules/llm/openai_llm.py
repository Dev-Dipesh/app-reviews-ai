"""
Implementation of LLM module using OpenAI API.
"""
import json
import os
from typing import Any, Dict, List, Optional, Union

import pandas as pd
from openai import OpenAI

from src.config import config
from src.modules.llm.interface import LLMInterface


class OpenAILLM(LLMInterface):
    """
    Implementation of LLM interface using OpenAI API.
    """
    
    def __init__(self, config_override: Optional[Dict[str, Any]] = None):
        """
        Initialize OpenAI LLM module.
        
        Args:
            config_override: Override for default configuration
        """
        # Initialize attributes with defaults
        self._api_key = None
        self._model = "gpt-4o"
        self._temperature = 0.3
        self._max_tokens = 1000
        self._client = None
        
        # Call parent constructor
        super().__init__(config_override)
        
        # Set attributes from config after validation
        self._api_key = self.config.get("api_key", None)
        self._model = self.config.get("model", "gpt-4o")
        self._temperature = self.config.get("temperature", 0.3)
        self._max_tokens = self.config.get("max_tokens", 1000)
    
    def _validate_config(self) -> None:
        """
        Validate OpenAI LLM configuration.
        
        Raises:
            ValueError: If required configuration is missing
        """
        # Check if API key is provided in config
        if not self._api_key:
            # Try to get from environment variable
            self._api_key = os.environ.get("OPENAI_API_KEY")
            
            # If still not available, try global config
            if not self._api_key:
                try:
                    self._api_key = config.get("llm", "api_key")
                except KeyError:
                    raise ValueError("OpenAI API key is required for the OpenAI LLM module")
        
        # Validate model
        if not self._model:
            try:
                self._model = config.get("llm", "model")
            except KeyError:
                self._model = "gpt-4o"  # Default model
    
    def initialize(self) -> None:
        """
        Initialize OpenAI client.
        
        Raises:
            RuntimeError: If initialization fails
        """
        try:
            # Check if we're in mock data mode
            use_mock_data = os.environ.get("USE_MOCK_DATA", "").lower() in ("true", "1", "yes", "y")
            
            if use_mock_data:
                # Skip API connection in mock mode to avoid unnecessary API calls
                print("Using mock mode for LLM - skipping OpenAI API initialization")
                self.is_initialized = True
                return
                
            # Create the client with API key
            self._client = OpenAI(api_key=self._api_key)
            
            # Test the connection with a simple completion
            _ = self._client.chat.completions.create(
                model=self._model,
                messages=[{"role": "user", "content": "Hello"}],
                max_tokens=10
            )
            
            self.is_initialized = True
        except Exception as e:
            raise RuntimeError(f"Failed to initialize OpenAI LLM module: {e}")
    
    def generate_text(
        self,
        prompt: str,
        system_prompt: Optional[str] = None,
        max_tokens: Optional[int] = None,
        temperature: Optional[float] = None,
        **kwargs
    ) -> str:
        """
        Generate text using OpenAI API.
        
        Args:
            prompt: Text prompt for the LLM
            system_prompt: System prompt to guide the model
            max_tokens: Maximum number of tokens to generate
            temperature: Temperature for generation (0.0 to 1.0)
            
        Returns:
            Generated text
        """
        if not self.is_initialized:
            raise RuntimeError("OpenAI LLM module not initialized")
            
        # Check if we're in mock data mode
        use_mock_data = os.environ.get("USE_MOCK_DATA", "").lower() in ("true", "1", "yes", "y")
        
        if use_mock_data:
            # Return mock data instead of making API call
            print("Using mock LLM response for prompt:", prompt[:50] + "..." if len(prompt) > 50 else prompt)
            
            # Generate a simple mock response based on the analysis type
            if "analyze" in prompt.lower() or "analyze_reviews" in kwargs.get("caller", ""):
                # Get app name from environment or use generic name
                app_name = os.environ.get("APP_NAME", "Mobile App")
                return f"This is a mock analysis of the {app_name} reviews. Users generally like the app's interface but have reported some issues with performance and crashes. The sentiment is mostly positive with some areas for improvement."
            elif "summarize" in prompt.lower():
                return "This is a mock summary of the provided text."
            else:
                return "This is a mock response from the LLM model. In production mode, real completions from OpenAI would be used."
        
        # Build messages array
        messages = []
        
        # Add system prompt if provided
        if system_prompt:
            messages.append({"role": "system", "content": system_prompt})
        
        # Add user prompt
        messages.append({"role": "user", "content": prompt})
        
        try:
            # Set generation parameters
            response = self._client.chat.completions.create(
                model=kwargs.get("model", self._model),
                messages=messages,
                max_tokens=max_tokens or self._max_tokens,
                temperature=temperature if temperature is not None else self._temperature,
                **{k: v for k, v in kwargs.items() if k not in ["model", "caller"]}
            )
            
            # Extract and return the generated text
            if response.choices and response.choices[0].message:
                return response.choices[0].message.content
            
            return ""
        except Exception as e:
            print(f"Error generating text with OpenAI: {e}")
            return f"Error: {str(e)}"
    
    def analyze_reviews(
        self,
        reviews: Union[List[Dict[str, Any]], List[str]],
        analysis_type: str = "general",
        context: Optional[str] = None,
        **kwargs
    ) -> Dict[str, Any]:
        """
        Analyze reviews using OpenAI.
        
        Args:
            reviews: List of reviews to analyze
            analysis_type: Type of analysis to perform
                (general, sentiment, issues, suggestions, etc.)
            context: Additional context for the analysis
            
        Returns:
            Analysis results
        """
        if not self.is_initialized:
            raise RuntimeError("OpenAI LLM module not initialized")
        
        # Format reviews for prompt
        formatted_reviews = []
        
        for i, review in enumerate(reviews):
            if isinstance(review, dict):
                # Extract relevant fields
                text = review.get("text", "")
                rating = review.get("rating", None)
                date = review.get("date", None)
                
                review_str = f"Review {i+1}:"
                if rating is not None:
                    review_str += f" Rating: {rating}/5"
                if date is not None:
                    review_str += f" Date: {date}"
                review_str += f"\n{text}\n"
                
                formatted_reviews.append(review_str)
            else:
                # If review is already a string
                formatted_reviews.append(f"Review {i+1}:\n{review}\n")
        
        # Create prompt based on analysis type
        system_prompts = {
            "general": "You are an expert analyst who specializes in analyzing customer reviews. Provide a comprehensive analysis of the reviews.",
            "sentiment": "You are a sentiment analysis expert. Focus on the emotional tone and overall sentiment in the reviews.",
            "issues": "You are a problem identification expert. Focus on identifying issues, bugs, and problems mentioned in the reviews.",
            "suggestions": "You are a product improvement expert. Focus on extracting suggestions and feature requests from the reviews.",
            "themes": "You are a thematic analysis expert. Identify recurring themes and topics across the reviews.",
            "trends": "You are a trend analysis expert. Identify patterns and changes in sentiment or topics over time in the reviews."
        }
        
        prompt_instructions = {
            "general": "Please analyze the following app reviews and provide insights on:\n- Overall sentiment\n- Key themes and patterns\n- Common issues users face\n- Feature requests or suggestions\n- Any other notable observations",
            "sentiment": "Please analyze the sentiment of the following app reviews:\n- Classify each review as positive, negative, or neutral\n- Identify the emotional tone and intensity\n- Summarize the overall sentiment across all reviews\n- Note any sentiment patterns or shifts",
            "issues": "Please identify all issues mentioned in the following app reviews:\n- Technical problems or bugs\n- Usability issues\n- Performance concerns\n- Feature limitations\n- Rank issues by frequency and severity",
            "suggestions": "Please extract all suggestions and feature requests from the following app reviews:\n- New features users want\n- Improvements to existing features\n- Usability enhancements\n- Rank suggestions by popularity and feasibility",
            "themes": "Please identify the main themes in the following app reviews:\n- Categorize reviews into 3-7 key themes\n- List key terms associated with each theme\n- Rank themes by frequency and importance",
            "trends": "Please analyze trends in the following app reviews:\n- Changes in sentiment over time\n- Emerging or fading topics\n- Shifts in user priorities or concerns"
        }
        
        # Get appropriate system prompt and instruction
        system_prompt = system_prompts.get(analysis_type, system_prompts["general"])
        instruction = prompt_instructions.get(analysis_type, prompt_instructions["general"])
        
        # Add context if provided
        if context:
            instruction = f"{context}\n\n{instruction}"
        
        # Format output style
        output_format = kwargs.get("output_format", "default")
        if output_format == "json":
            instruction += "\n\nProvide your analysis in JSON format with appropriate structure."
        elif output_format == "bullets":
            instruction += "\n\nFormat your analysis as concise bullet points."
        elif output_format == "markdown":
            instruction += "\n\nFormat your analysis in Markdown with appropriate headers and structure."
        
        # Build final prompt
        prompt = f"{instruction}\n\n{''.join(formatted_reviews)}\n\nPlease provide your detailed analysis."
        
        try:
            # Remove output_format if present to avoid OpenAI API errors
            kwargs_copy = kwargs.copy()
            kwargs_copy.pop('output_format', None)
            
            # Generate the analysis
            response = self.generate_text(
                prompt=prompt,
                system_prompt=system_prompt,
                caller="analyze_reviews",
                **kwargs_copy
            )
            
            # Parse JSON response if requested
            if output_format == "json":
                try:
                    # Find JSON in the response
                    json_start = response.find("{")
                    json_end = response.rfind("}") + 1
                    
                    if json_start >= 0 and json_end > json_start:
                        json_str = response[json_start:json_end]
                        return json.loads(json_str)
                    else:
                        return {"analysis": response, "format_error": "No JSON found in response"}
                except json.JSONDecodeError:
                    return {"analysis": response, "format_error": "Invalid JSON in response"}
            
            # Return plain text response for other formats
            return {"analysis": response}
        except Exception as e:
            print(f"Error analyzing reviews with OpenAI: {e}")
            return {"error": str(e)}
    
    def summarize_text(
        self,
        text: str,
        max_length: Optional[int] = None,
        format_type: str = "paragraph",
        **kwargs
    ) -> str:
        """
        Summarize text using OpenAI.
        
        Args:
            text: Text to summarize
            max_length: Maximum length of summary
            format_type: Format of summary (paragraph, bullets, etc.)
            
        Returns:
            Summarized text
        """
        if not self.is_initialized:
            raise RuntimeError("OpenAI LLM module not initialized")
        
        # Create system prompt
        system_prompt = "You are an expert summarizer who can distill complex information into clear, concise summaries."
        
        # Create prompt based on format type
        format_instructions = {
            "paragraph": "Provide a concise paragraph summary.",
            "bullets": "Provide a bulleted list of key points.",
            "tldr": "Provide a one-sentence TL;DR summary.",
            "detailed": "Provide a detailed summary covering all important aspects.",
            "executive": "Provide an executive summary with highlights and key takeaways."
        }
        
        format_instruction = format_instructions.get(format_type, format_instructions["paragraph"])
        
        # Add length constraint if provided
        if max_length:
            format_instruction += f" Keep the summary under {max_length} characters."
        
        # Build final prompt
        prompt = f"Please summarize the following text:\n\n{text}\n\n{format_instruction}"
        
        try:
            # Generate the summary
            return self.generate_text(
                prompt=prompt,
                system_prompt=system_prompt,
                **kwargs
            )
        except Exception as e:
            print(f"Error summarizing text with OpenAI: {e}")
            return f"Error: {str(e)}"
    
    def extract_structured_data(
        self,
        text: str,
        schema: Dict[str, Any],
        **kwargs
    ) -> Dict[str, Any]:
        """
        Extract structured data from text using OpenAI.
        
        Args:
            text: Text to extract data from
            schema: Schema defining the structure of the data to extract
            
        Returns:
            Extracted structured data
        """
        if not self.is_initialized:
            raise RuntimeError("OpenAI LLM module not initialized")
        
        # Create system prompt
        system_prompt = "You are an expert at extracting structured information from text."
        
        # Format schema as instructions
        schema_str = json.dumps(schema, indent=2)
        
        # Build final prompt
        prompt = f"""Please extract information from the following text according to this schema:

```json
{schema_str}
```

Text:
{text}

Extract the information and return it in valid JSON format that matches the schema exactly."""
        
        try:
            # Generate the extraction
            response = self.generate_text(
                prompt=prompt,
                system_prompt=system_prompt,
                **kwargs
            )
            
            # Parse JSON response
            try:
                # Find JSON in the response
                json_start = response.find("{")
                json_end = response.rfind("}") + 1
                
                if json_start >= 0 and json_end > json_start:
                    json_str = response[json_start:json_end]
                    return json.loads(json_str)
                else:
                    return {"error": "No JSON found in response", "raw_response": response}
            except json.JSONDecodeError:
                return {"error": "Invalid JSON in response", "raw_response": response}
        except Exception as e:
            print(f"Error extracting structured data with OpenAI: {e}")
            return {"error": str(e)}