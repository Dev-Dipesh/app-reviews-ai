#!/usr/bin/env python
"""
This script fixes the issue with the developer response rate visualization
in the enhanced_dashboard.py file.
"""

import os
import re

# Path to the enhanced_dashboard.py file
file_path = os.path.join("src", "modules", "visualization", "enhanced_dashboard.py")

# Read the current content
with open(file_path, 'r') as f:
    content = f.read()

# Find the problematic code pattern
pattern = r'yaxis=dict\(range=\[\-5, max\(valid_response\[\s*[\'"]response_rate[\'"]\s*\]\s*\*\s*1\.1,\s*5\)\]\)'

# Create the fixed version with scalar max value
replacement = "# Calculate max response rate\n                max_response_rate = valid_response['response_rate'].max()\n                \n                # Update layout using scalar max\n                fig.update_layout(\n                    xaxis_title=\"App Version\",\n                    yaxis_title=\"Response Rate (%)\",\n                    yaxis=dict(range=[-5, max(max_response_rate * 1.1, 5)])  # Use scalar value"

# Replace the pattern if found
if re.search(pattern, content):
    modified_content = re.sub(pattern, "yaxis=dict(range=[-5, max(max_response_rate * 1.1, 5)])", content)
    
    # Now let's insert the calculation of max_response_rate before the update_layout call
    # Find the update_layout pattern
    layout_pattern = r'fig\.update_layout\('
    
    # Insert the calculation before update_layout
    calculation = "# Calculate max response rate as scalar first\n                max_response_rate = valid_response['response_rate'].max()\n                \n                "
    modified_content = re.sub(layout_pattern, calculation + "fig.update_layout(", modified_content)
    
    # Write the modified content back to the file
    with open(file_path, 'w') as f:
        f.write(modified_content)
    
    print(f"Successfully fixed the response rate visualization in {file_path}")
else:
    print(f"Could not find the pattern to fix in {file_path}")
    print("You may need to manually modify the file to fix the Series comparison issue.")