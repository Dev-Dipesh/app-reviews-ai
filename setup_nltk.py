#!/usr/bin/env python
"""
Setup script to download required NLTK resources.
Run this script once after installing the package to ensure all required NLTK resources are available.
"""
import nltk
import os
import sys

def setup_nltk():
    """Download all required NLTK resources."""
    print("=== Setting up NLTK resources ===")
    
    # Create data directory for NLTK
    nltk_data_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "data", "nltk_data")
    os.makedirs(nltk_data_dir, exist_ok=True)
    
    # Add to NLTK's data path
    nltk.data.path.append(nltk_data_dir)
    
    # Resources to download
    resources = [
        'punkt',
        'stopwords',
        'wordnet',
        'omw-1.4'
    ]
    
    # Download each resource
    for resource in resources:
        print(f"Downloading {resource}...")
        try:
            nltk.download(resource, download_dir=nltk_data_dir)
            print(f"✓ Successfully downloaded {resource}")
        except Exception as e:
            print(f"✗ Failed to download {resource}: {e}")
    
    print("\nSetup complete!")
    print(f"NLTK resources downloaded to: {nltk_data_dir}")
    print("\nIf you encounter any NLTK resource errors when running the application, please run this script again.")

if __name__ == "__main__":
    setup_nltk()