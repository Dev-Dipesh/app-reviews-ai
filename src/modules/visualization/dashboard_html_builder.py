"""
Simplified dashboard HTML generator module that ensures visualizations work properly.
"""
import os
import base64
import re
from datetime import datetime
from typing import Any, Dict, List, Optional

import pandas as pd
import plotly

def parse_ai_insights(text):
    """
    Parse AI insights with special formatting for our specific structure.
    
    Args:
        text: The insight text from LLM
        
    Returns:
        Formatted HTML
    """
    # Handle the specific format of the insights
    # 1. Detect sections with numbers
    sections = re.split(r'\n\s*(\d+\.)\s+(.*?):', text)
    
    if len(sections) < 3:  # Not the expected format, use default formatting
        return basic_markdown_to_html(text)
    
    # Build the HTML structure
    html = "<ol>"
    
    for i in range(1, len(sections), 3):
        if i+1 < len(sections):
            section_num = sections[i]  # e.g., "1."
            section_title = sections[i+1]  # e.g., "TOP 5 CRITICAL ISSUES TO FIX"
            
            # Get the content up to the next section number
            if i+3 < len(sections):
                section_content = sections[i+2].strip()
            else:
                section_content = sections[i+2].strip()
            
            # Handle bullet points by converting them to an inner HTML list
            if re.search(r'^\s*-\s+\*\*', section_content, re.MULTILINE):
                # This section has bullet points with bold headers
                bullet_items = re.split(r'\n\s*-\s+', section_content)
                list_html = "<ul>"
                
                for item in bullet_items:
                    if item.strip():
                        # Extract the bold part as the header if it exists
                        bold_match = re.match(r'\*\*(.*?)\*\*:(.*)', item)
                        if bold_match:
                            header = bold_match.group(1).strip()
                            content = bold_match.group(2).strip()
                            list_html += f"<li><strong>{header}</strong>: {content}</li>"
                        else:
                            # Try another pattern: **Bold Text** non-bold text
                            bold_match = re.match(r'\*\*(.*?)\*\*(.*)', item)
                            if bold_match:
                                header = bold_match.group(1).strip()
                                content = bold_match.group(2).strip()
                                list_html += f"<li><strong>{header}</strong>{content}</li>"
                            else:
                                # Just a regular bullet point
                                list_html += f"<li>{item}</li>"
                
                list_html += "</ul>"
                
                # Add this section to the main HTML
                html += f"<li><strong>{section_title}</strong>:{list_html}</li>"
            else:
                # No bullet points, just regular text
                content_html = basic_markdown_to_html(section_content)
                html += f"<li><strong>{section_title}</strong>:{content_html}</li>"
    
    html += "</ol>"
    return html

def basic_markdown_to_html(markdown_text):
    """
    Convert basic markdown to HTML
    
    Args:
        markdown_text: The markdown text to convert
        
    Returns:
        Converted HTML
    """
    if not markdown_text:
        return ""
        
    html = markdown_text
    
    # Convert bold (** or __)
    html = re.sub(r'\*\*(.*?)\*\*', r'<strong>\1</strong>', html)
    html = re.sub(r'__(.*?)__', r'<strong>\1</strong>', html)
    
    # Convert italic (* or _)
    html = re.sub(r'\*(.*?)\*', r'<em>\1</em>', html)
    html = re.sub(r'_(.*?)_', r'<em>\1</em>', html)
    
    # Convert bullet points lists
    # First identify blocks of bullet points
    bullet_blocks = re.findall(r'((?:^\s*-\s+.+\n?)+)', html, re.MULTILINE)
    for block in bullet_blocks:
        # Extract each bullet point
        bullets = re.findall(r'^\s*-\s+(.*?)$', block, re.MULTILINE)
        list_html = "<ul>" + "".join([f"<li>{bullet}</li>" for bullet in bullets]) + "</ul>"
        # Replace the block with the list HTML
        html = html.replace(block, list_html)
    
    # Convert paragraphs (empty lines between text)
    html = re.sub(r'(\n\s*\n)', r'</p><p>', html)
    
    # Wrap in <p> tags if not already done
    if not html.startswith(('<h1', '<h2', '<h3', '<p', '<ul', '<ol')):
        html = '<p>' + html + '</p>'
    
    return html

def create_simple_dashboard(
    output_dir: str,
    visualizations: Dict[str, Any],
    insights: Optional[Dict[str, Any]] = None,
    title: str = "Review Analysis Dashboard"
) -> str:
    """
    Create a simple dashboard with visualizations and insights.
    
    Args:
        output_dir: Directory to save the dashboard
        visualizations: Dictionary of pre-generated visualization HTML files
        insights: Dictionary of LLM-generated insights
        title: Dashboard title
        
    Returns:
        Path to the generated dashboard HTML file
    """
    # Create timestamp for filename
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)
    
    # Create main HTML file path
    output_path = os.path.join(output_dir, f"dashboard_{timestamp}.html")
    
    # Start building HTML
    html_content = f"""<!DOCTYPE html>
<html>
<head>
    <title>{title}</title>
    <meta charset="UTF-8">
    <style>
        body {{
            font-family: Arial, sans-serif;
            margin: 0;
            padding: 20px;
            background-color: #f5f5f5;
        }}
        .container {{
            max-width: 1200px;
            margin: 0 auto;
            background-color: white;
            padding: 20px;
            border-radius: 10px;
            box-shadow: 0 0 10px rgba(0,0,0,0.1);
        }}
        h1 {{
            color: #333;
            border-bottom: 2px solid #eee;
            padding-bottom: 10px;
        }}
        h2 {{
            color: #444;
            margin-top: 30px;
        }}
        h3 {{
            color: #555;
            margin-top: 20px;
        }}
        .plot-container {{
            margin-bottom: 30px;
            border: 1px solid #eee;
            border-radius: 5px;
            padding: 15px;
        }}
        .insights-container {{
            margin-top: 40px;
            padding: 20px;
            border: 1px solid #ddd;
            border-radius: 10px;
            background-color: #f9f9f9;
        }}
        .insight-section {{
            margin-bottom: 25px;
        }}
        .insight-content {{
            padding: 15px;
            background-color: white;
            border-left: 4px solid #007bff;
            margin-top: 10px;
            line-height: 1.6;
        }}
        .insight-content strong {{
            color: #0056b3;
        }}
        .insight-content em {{
            color: #6c757d;
        }}
        .insight-content ul {{
            padding-left: 20px;
            margin-top: 10px;
            margin-bottom: 10px;
        }}
        .insight-content ol {{
            padding-left: 20px;
            margin-top: 10px;
            margin-bottom: 10px;
        }}
        .insight-content li {{
            margin-bottom: 8px;
        }}
        .insight-content ol > li {{
            font-weight: bold;
            margin-top: 15px;
            margin-bottom: 15px;
        }}
        .insight-content ol > li > ul {{
            font-weight: normal;
        }}
        .button {{
            padding: 8px 15px;
            background-color: #4CAF50;
            color: white;
            border: none;
            border-radius: 5px;
            cursor: pointer;
            font-size: 14px;
        }}
        .button:hover {{
            background-color: #45a049;
        }}
        .flex-right {{
            display: flex;
            justify-content: flex-end;
            margin-bottom: 10px;
        }}
        iframe {{
            border: none;
            width: 100%;
            height: 500px;
        }}
    </style>
</head>
<body>
    <div class="container">
        <h1>{title}</h1>
"""
    
    # Add visualizations using iframes
    for i, (plot_name, viz_path) in enumerate(visualizations.items()):
        # Create proper relative path
        rel_path = os.path.relpath(viz_path, os.path.dirname(output_path))
        
        html_content += f"""
        <div class="plot-container">
            <h2>{plot_name.replace('_', ' ').title()}</h2>
            <iframe src="{rel_path}"></iframe>
        </div>
"""
    
    # Add insights section
    if insights:
        html_content += """
        <div class="insights-container">
            <h2>AI-Generated Insights</h2>
            <div class="flex-right">
                <button onclick="copyInsights()" class="button">Copy All Insights</button>
            </div>
            <div id="insights-content">
"""
        
        # Add each insight section
        for insight_type, insight_data in insights.items():
            if "analysis" in insight_data:
                section_title = insight_type.replace('_', ' ').title()
                # Use our special insights parser for formatting
                analysis_content = parse_ai_insights(insight_data["analysis"])
                
                html_content += f"""
                <div class="insight-section">
                    <h3>{section_title} Insights</h3>
                    <div class="insight-content">
                        {analysis_content}
                    </div>
                </div>
"""
        
        # Close insights container and add copy script
        html_content += """
            </div>
        </div>
        <script>
        function copyInsights() {
            const insightsText = document.getElementById('insights-content').innerText;
            navigator.clipboard.writeText(insightsText)
                .then(() => {
                    alert('Insights copied to clipboard!');
                })
                .catch(err => {
                    console.error('Failed to copy: ', err);
                    alert('Failed to copy insights. Try selecting and copying manually.');
                });
        }
        </script>
"""
    
    # Close HTML
    html_content += """
    </div>
</body>
</html>
"""
    
    # Write the HTML file
    with open(output_path, 'w') as file:
        file.write(html_content)
    
    return output_path