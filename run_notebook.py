#!/usr/bin/env python
import os
import sys
import subprocess

# Set environment variables
os.environ['FORCE_REFRESH'] = 'true'
os.environ['USE_MOCK_DATA'] = 'false'

# Change to project directory
os.chdir('/Users/dipesh/Local-Projects/indigo-reviews-ai')

# Run the notebook
print("Running data_preprocessing.ipynb...")
try:
    # Use IPython to execute the notebook
    command = [sys.executable, '-c', """
import sys
from IPython import get_ipython
from nbformat import read
from IPython.core.interactiveshell import InteractiveShell

# Load the notebook
with open('notebooks/data_preprocessing.ipynb') as f:
    nb = read(f, as_version=4)

# Create an interactive shell
shell = InteractiveShell.instance()

# Execute each cell
for cell in nb.cells:
    if cell.cell_type == 'code':
        print(f"Executing cell: {cell.source[:50]}...")
        shell.run_cell(cell.source)
"""]
    
    subprocess.run(command, check=True)
    print("Notebook execution completed successfully!")
except Exception as e:
    print(f"Error running notebook: {e}")