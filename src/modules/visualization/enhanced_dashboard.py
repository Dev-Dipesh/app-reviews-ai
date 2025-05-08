"""
Enhanced dashboard creation with LLM insights integration.
"""
import os
import json
from datetime import datetime
from typing import Any, Dict, List, Optional

import pandas as pd
import plotly.graph_objects as go
import plotly.utils  # Add explicit import for PlotlyJSONEncoder
from plotly.subplots import make_subplots

def create_enhanced_dashboard(
    data: pd.DataFrame,
    output_dir: str = "reports",
    include_plots: Optional[List[str]] = None,
    title: str = "Review Analysis Dashboard",
    topic_words: Optional[Dict[Any, List[str]]] = None,
    insights: Optional[Dict[str, Any]] = None,
    visualizations: Optional[Dict[str, Any]] = None,
    **kwargs
) -> Dict[str, Any]:
    """
    Create an enhanced dashboard with visualizations and LLM insights.
    
    Args:
        data: DataFrame containing reviews
        output_dir: Directory to save the dashboard
        include_plots: List of plots to include
        title: Dashboard title
        topic_words: Dictionary mapping topic IDs to representative words
        insights: Dictionary of LLM-generated insights
        visualizations: Dictionary of pre-generated visualizations
        
    Returns:
        Dictionary with dashboard information
    """
    # Create timestamp for filename
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)
    
    # Set output path
    output_path = os.path.join(output_dir, f"dashboard_{timestamp}.html")
    
    # Define default plots to include
    if include_plots is None:
        include_plots = [
            "rating_distribution",
            "rating_trend",
            "sentiment_distribution",
            "topic_distribution"
        ]
    
    # Create HTML header
    html_content = f"""
    <!DOCTYPE html>
    <html>
    <head>
        <title>{title}</title>
        <meta charset="UTF-8">
        <!-- Use a specific version of Plotly -->
        <script src="https://cdn.plot.ly/plotly-2.29.1.min.js"></script>
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
        </style>
    </head>
    <body>
        <div class="container">
            <h1>{title}</h1>
    """
    
    # Add plots
    for i, plot_type in enumerate(include_plots):
        if visualizations and plot_type in visualizations:
            # Create plot div
            html_content += f"""
            <div class="plot-container">
                <h2>{plot_type.replace('_', ' ').title()}</h2>
                <div id="plot-{i}" style="width: 100%; height: 500px;"></div>
            </div>
            """
    
    # Add insights container
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
                analysis_content = insight_data["analysis"].replace("\n", "<br>")
                
                html_content += f"""
                <div class="insight-section">
                    <h3>{section_title} Insights</h3>
                    <div class="insight-content">
                        {analysis_content}
                    </div>
                </div>
                """
        
        # Close insights div and add copy script
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
    
    # Add Javascript for plots if visualizations are provided
    if visualizations:
        html_content += "<script>\n"
        
        # Add each plot
        for i, plot_type in enumerate(include_plots):
            if plot_type in visualizations:
                viz = visualizations[plot_type]
                if "figure" in viz and viz["figure"]:
                    try:
                        # A simpler approach: write the figure directly
                        html_content += f"Plotly.newPlot('plot-{i}', {json.dumps(viz['figure'].data, cls=plotly.utils.PlotlyJSONEncoder)}, {json.dumps(viz['figure'].layout, cls=plotly.utils.PlotlyJSONEncoder)});\n"
                    except Exception as e:
                        # Fallback if direct conversion fails
                        html_content += f"console.log('Error loading plot {plot_type}: {str(e)}');\n"
        
        html_content += "</script>\n"
    
    # Close HTML body and html tags
    html_content += """
        </div>
    </body>
    </html>
    """
    
    # Write the HTML file
    with open(output_path, 'w') as file:
        file.write(html_content)
    
    # If visualizations exist, create individual HTML files for each visualization
    if visualizations:
        viz_dir = os.path.join(output_dir, f"visualizations_{timestamp}")
        os.makedirs(viz_dir, exist_ok=True)
        
        for plot_type, viz in visualizations.items():
            if "figure" in viz and viz["figure"]:
                viz_path = os.path.join(viz_dir, f"{plot_type}.html")
                try:
                    viz["figure"].write_html(viz_path)
                except Exception as e:
                    print(f"Error saving {plot_type} visualization: {e}")
    
    return {
        "title": title,
        "type": "enhanced_dashboard",
        "file_path": output_path,
        "timestamp": timestamp
    }

def format_insights_for_jupyter(insights: Dict[str, Any]) -> str:
    """
    Format LLM insights for display in a Jupyter notebook.
    
    Args:
        insights: Dictionary of insights from LLM
    
    Returns:
        Formatted markdown string for display
    """
    if not insights:
        return "No insights available."
    
    markdown = "# AI-Generated Insights\n\n"
    
    # Add each insight type
    for insight_type, insight_data in insights.items():
        if "analysis" in insight_data:
            section_title = insight_type.replace('_', ' ').title()
            markdown += f"## {section_title} Insights\n\n"
            markdown += insight_data["analysis"] + "\n\n"
    
    return markdown