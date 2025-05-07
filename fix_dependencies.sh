#!/bin/bash

# Exit script if any command fails
set -e

echo "Creating a clean virtual environment..."
# Remove existing venv if it exists
if [ -d "venv" ]; then
  rm -rf venv
fi

# Create new virtual environment
python -m venv venv

# Activate virtual environment
source venv/bin/activate

echo "Installing compatible versions of critical dependencies..."
# Install specific versions of critical dependencies first
# pip install openai==0.28.1
# pip install chromadb==0.4.18

echo "Installing remaining dependencies..."
# Install the rest with the constraint that openai and chromadb are already installed
pip install -r requirements.txt

echo "Setup complete. Now run the application with:"
echo "source venv/bin/activate"
echo "python run.py --max-reviews 50"