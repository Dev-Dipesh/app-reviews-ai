#!/usr/bin/env python3

import os
import sys
import json
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import plotly.io as pio
import plotly.utils  # This was the missing import
from datetime import datetime

def main():
    print("Starting dashboard generation...")
    
    # Create a mock dataframe for testing
    df = pd.DataFrame({
        'rating': [1, 2, 3, 4, 5, 4, 5, 1, 2, 3],
        'sentiment': ['negative', 'negative', 'neutral', 'positive', 'positive', 
                     'positive', 'positive', 'negative', 'negative', 'neutral'],
        'version': ['6.0.1', '6.0.2', '6.0.3', '7.0.0', '7.0.1', 
                   '6.0.1', '6.0.2', '6.0.3', '7.0.0', '7.0.1']
    })
    
    # Create output directory
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir = os.path.join("reports", f"full_dashboard_{timestamp}")
    os.makedirs(output_dir, exist_ok=True)
    
    # Basic test of the save_figure function using plotly.utils
    def save_figure(fig, filename):
        # Save as both HTML and JSON for embedding
        fig_path = os.path.join(output_dir, f"{filename}.html")
        fig.write_html(fig_path)
        
        # Also save as JSON for embedding in main HTML - this was failing without plotly.utils
        json_path = os.path.join(output_dir, f"{filename}.json")
        with open(json_path, 'w') as f:
            f.write(json.dumps(fig.to_dict(), cls=plotly.utils.PlotlyJSONEncoder))
        
        return fig_path
    
    # Create a simple test plot
    fig = px.histogram(df, x="rating", title="Test Histogram")
    test_path = save_figure(fig, "test_plot")
    
    print(f"Test figure saved to: {test_path}")
    print(f"Dashboard output directory: {output_dir}")
    
    # Create a simple HTML dashboard
    html_content = f"""<!DOCTYPE html>
    <html>
    <head>
        <title>Test Dashboard</title>
        <script src="https://cdn.plot.ly/plotly-2.29.1.min.js"></script>
    </head>
    <body>
        <h1>Test Dashboard</h1>
        <iframe width="100%" height="500px" src="test_plot.html"></iframe>
    </body>
    </html>
    """
    
    # Write the test dashboard
    dashboard_path = os.path.join(output_dir, "test_dashboard.html")
    with open(dashboard_path, "w") as f:
        f.write(html_content)
    
    print(f"Test dashboard created at: {dashboard_path}")
    print("Dashboard generation completed successfully!")

if __name__ == "__main__":
    main()