"""
Initialize required resources for the application.
This script downloads required NLTK resources needed for text processing.
"""
import logging
import nltk
import os

def initialize_nltk_resources():
    """Download required NLTK resources."""
    logger = logging.getLogger("app_reviews")
    logger.info("Initializing NLTK resources...")
    
    # Create a data directory for NLTK if it doesn't exist
    nltk_data_dir = os.path.join(os.path.dirname(os.path.dirname(__file__)), "data", "nltk_data")
    os.makedirs(nltk_data_dir, exist_ok=True)
    
    # Set NLTK data path
    nltk.data.path.append(nltk_data_dir)
    
    # List of required resources
    required_resources = [
        "punkt", 
        "stopwords", 
        "wordnet", 
        "omw-1.4"
        # Note: punkt_tab is not needed and was causing unnecessary download attempts
    ]
    
    # Download each resource
    for resource in required_resources:
        try:
            logger.info(f"Downloading NLTK resource: {resource}")
            nltk.download(resource, download_dir=nltk_data_dir, quiet=True)
            logger.info(f"Successfully downloaded NLTK resource: {resource}")
        except Exception as e:
            logger.warning(f"Warning: Could not download NLTK resource {resource}: {e}")
    
    logger.info("NLTK resource initialization completed")

if __name__ == "__main__":
    # Configure basic logging when run directly
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    )
    initialize_nltk_resources()