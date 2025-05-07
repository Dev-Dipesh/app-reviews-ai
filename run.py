#!/usr/bin/env python
"""
Main entry point for App Reviews AI.
This script handles Python path issues and calls the main runner.
"""
import os
import sys
import logging

# Add the project root to the Python path
project_root = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, project_root)

# Set environment variables for third-party libraries
# Prevent huggingface tokenizers parallelism warning
os.environ["TOKENIZERS_PARALLELISM"] = os.environ.get("TOKENIZERS_PARALLELISM", "false")

# Configure basic logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)

# Initialize required resources
from src.initialize_resources import initialize_nltk_resources
initialize_nltk_resources()

# Import and run the main function
from src.runner import main

if __name__ == "__main__":
    # Pass command line arguments to the main function
    main()