"""
Visualization utilities for review analysis.
Provides standardized functions for creating common visualizations.
"""

import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np
from typing import Dict, List, Any, Optional, Union


def create_rating_distribution(df, title="Rating Distribution", figsize=(12, 6)):
    """
    Create a bar chart showing the distribution of ratings.
    
    Args:
        df: DataFrame containing review data with a 'rating' column
        title: Title for the chart
        figsize: Tuple of (width, height) for the figure size
        
    Returns:
        matplotlib.figure.Figure: The generated figure, or None if data is invalid
    """
    if 'rating' not in df.columns:
        print("Rating column not found in data")
        return None
    
    plt.figure(figsize=figsize)
    rating_counts = df['rating'].value_counts().sort_index()
    
    # Calculate percentages
    total = rating_counts.sum()
    percentages = rating_counts / total * 100
    
    # Plot bars
    ax = rating_counts.plot(kind='bar', color='skyblue')
    plt.title(title, fontsize=16)
    plt.xlabel('Rating', fontsize=14)
    plt.ylabel('Count', fontsize=14)
    plt.xticks(fontsize=12)
    plt.yticks(fontsize=12)
    
    # Add labels with counts and percentages
    for i, (count, percentage) in enumerate(zip(rating_counts, percentages)):
        label = f"{count}\n({percentage:.1f}%)"
        ax.text(i, count + (rating_counts.max() * 0.02), 
                label, 
                ha='center', va='bottom',
                fontweight='bold')
    
    plt.tight_layout()
    plt.grid(axis='y', linestyle='--', alpha=0.7)
    return plt.gcf()


def create_sentiment_over_time(df, time_period="month", figsize=(14, 7)):
    """
    Create a line chart showing sentiment trends over time.
    
    Args:
        df: DataFrame containing review data with 'sentiment_category' and 'date' columns
        time_period: Time aggregation level ('day', 'week', 'month', 'quarter')
        figsize: Tuple of (width, height) for the figure size
        
    Returns:
        matplotlib.figure.Figure: The generated figure, or None if data is invalid
    """
    # Check required columns
    if 'sentiment_category' not in df.columns or 'date' not in df.columns:
        missing = []
        if 'sentiment_category' not in df.columns:
            missing.append('sentiment_category')
        if 'date' not in df.columns:
            missing.append('date')
        print(f"Required columns not found: {', '.join(missing)}")
        return None
    
    # Ensure date is datetime
    if not pd.api.types.is_datetime64_dtype(df['date']):
        df = df.copy()
        df['date'] = pd.to_datetime(df['date'], errors='coerce')
    
    # Create time period column
    df = df.copy()
    if time_period == "day":
        df['period'] = df['date'].dt.strftime('%Y-%m-%d')
        period_title = "Day"
    elif time_period == "week":
        df['period'] = df['date'].dt.strftime('%Y-%U')
        period_title = "Week"
    elif time_period == "quarter":
        df['period'] = df['date'].dt.year.astype(str) + '-Q' + df['date'].dt.quarter.astype(str)
        period_title = "Quarter"
    else:  # default to month
        df['period'] = df['date'].dt.strftime('%Y-%m')
        period_title = "Month"
    
    # Aggregate by period and sentiment
    sentiment_counts = df.groupby(['period', 'sentiment_category']).size().unstack(fill_value=0)
    
    # Calculate percentages
    totals = sentiment_counts.sum(axis=1)
    sentiment_pct = sentiment_counts.div(totals, axis=0) * 100
    
    # Create plot
    plt.figure(figsize=figsize)
    
    # Plot lines for each sentiment
    sentiment_colors = {'positive': 'green', 'neutral': 'gray', 'negative': 'red'}
    for sentiment in sentiment_pct.columns:
        plt.plot(sentiment_pct.index, sentiment_pct[sentiment], 
                 marker='o', linewidth=3, label=sentiment.capitalize(),
                 color=sentiment_colors.get(sentiment, 'blue'))
    
    # Add chart elements
    plt.title(f'Sentiment Trends by {period_title}', fontsize=16)
    plt.xlabel(period_title, fontsize=14)
    plt.ylabel('Percentage (%)', fontsize=14)
    plt.xticks(rotation=45, ha='right', fontsize=12)
    plt.yticks(fontsize=12)
    plt.legend(fontsize=12)
    plt.grid(linestyle='--', alpha=0.7)
    
    # Add percentage labels at the end of each line
    for sentiment in sentiment_pct.columns:
        last_value = sentiment_pct[sentiment].iloc[-1]
        plt.annotate(f'{last_value:.1f}%', 
                     xy=(sentiment_pct.index[-1], last_value),
                     xytext=(10, 0),
                     textcoords='offset points',
                     fontweight='bold',
                     color=sentiment_colors.get(sentiment, 'blue'))
    
    plt.tight_layout()
    return plt.gcf()


def create_version_comparison(df, metric='rating', figsize=(14, 7)):
    """
    Create a comparison chart of versions based on specified metric.
    
    Args:
        df: DataFrame containing review data with 'reviewCreatedVersion' and metric columns
        metric: Column name to use for comparison (e.g., 'rating', 'sentiment_compound')
        figsize: Tuple of (width, height) for the figure size
        
    Returns:
        matplotlib.figure.Figure: The generated figure, or None if data is invalid
    """
    # Check required columns
    if 'reviewCreatedVersion' not in df.columns or metric not in df.columns:
        missing = []
        if 'reviewCreatedVersion' not in df.columns:
            missing.append('reviewCreatedVersion')
        if metric not in df.columns:
            missing.append(metric)
        print(f"Required columns not found: {', '.join(missing)}")
        return None
    
    # Get top versions by count
    version_counts = df['reviewCreatedVersion'].value_counts()
    top_versions = version_counts[version_counts >= 10].index.tolist()
    
    if not top_versions:
        print("No versions with sufficient data found")
        return None
    
    # Filter to top versions
    version_df = df[df['reviewCreatedVersion'].isin(top_versions)].copy()
    
    # Calculate metric by version
    if metric == 'rating':
        # For rating, calculate mean
        metric_by_version = version_df.groupby('reviewCreatedVersion')['rating'].mean().sort_values()
        metric_label = 'Average Rating'
        color_map = 'RdYlGn'
    elif 'sentiment' in metric:
        # For sentiment metrics, calculate mean
        metric_by_version = version_df.groupby('reviewCreatedVersion')[metric].mean().sort_values()
        metric_label = f'Average {metric.replace("_", " ").title()}'
        color_map = 'coolwarm'
    else:
        # For other metrics, calculate mean
        metric_by_version = version_df.groupby('reviewCreatedVersion')[metric].mean().sort_values()
        metric_label = f'Average {metric}'
        color_map = 'viridis'
    
    # Create horizontal bar chart
    plt.figure(figsize=figsize)
    bars = plt.barh(metric_by_version.index, metric_by_version.values, 
                   color=plt.cm.get_cmap(color_map)(np.linspace(0, 1, len(metric_by_version))))
    
    # Add labels
    for i, (version, value) in enumerate(metric_by_version.items()):
        count = version_counts[version]
        plt.text(value + 0.05, i, f"{value:.2f} (n={count})", va='center', fontweight='bold')
    
    # Add chart elements
    plt.title(f'{metric_label} by App Version', fontsize=16)
    plt.xlabel(metric_label, fontsize=14)
    plt.ylabel('App Version', fontsize=14)
    plt.yticks(fontsize=12)
    plt.xticks(fontsize=12)
    plt.grid(axis='x', linestyle='--', alpha=0.7)
    
    # Add count annotation
    plt.figtext(0.5, 0.01, f"Note: Only showing versions with at least 10 reviews. ({len(top_versions)} of {len(version_counts)} versions)", 
                ha="center", fontsize=12, style='italic')
    
    plt.tight_layout()
    return plt.gcf()


def create_rating_time_heatmap(df, time_period="month", figsize=(16, 8)):
    """
    Create a heatmap showing rating distribution over time.
    
    Args:
        df: DataFrame containing review data with 'rating' and 'date' columns
        time_period: Time aggregation level ('day', 'week', 'month', 'quarter')
        figsize: Tuple of (width, height) for the figure size
        
    Returns:
        matplotlib.figure.Figure: The generated figure, or None if data is invalid
    """
    # Check required columns
    if 'rating' not in df.columns or 'date' not in df.columns:
        missing = []
        if 'rating' not in df.columns:
            missing.append('rating')
        if 'date' not in df.columns:
            missing.append('date')
        print(f"Required columns not found: {', '.join(missing)}")
        return None
    
    # Ensure date is datetime
    if not pd.api.types.is_datetime64_dtype(df['date']):
        df = df.copy()
        df['date'] = pd.to_datetime(df['date'], errors='coerce')
    
    # Create time period column
    df = df.copy()
    if time_period == "day":
        df['period'] = df['date'].dt.strftime('%Y-%m-%d')
        period_title = "Day"
    elif time_period == "week":
        df['period'] = df['date'].dt.strftime('%Y-%U')
        period_title = "Week"
    elif time_period == "quarter":
        df['period'] = df['date'].dt.year.astype(str) + '-Q' + df['date'].dt.quarter.astype(str)
        period_title = "Quarter"
    else:  # default to month
        df['period'] = df['date'].dt.strftime('%Y-%m')
        period_title = "Month"
    
    # Get valid ratings and periods
    ratings = sorted(df['rating'].unique().tolist())
    periods = sorted(df['period'].unique().tolist())
    
    # Count reviews by period and rating
    period_rating_counts = df.groupby(['period', 'rating']).size().unstack(fill_value=0)
    
    # Calculate percentages per period
    period_totals = period_rating_counts.sum(axis=1)
    period_rating_pct = period_rating_counts.div(period_totals, axis=0) * 100
    
    # Create heatmap
    plt.figure(figsize=figsize)
    ax = sns.heatmap(period_rating_pct.T, annot=True, fmt='.1f', cmap='RdYlGn', 
                     linewidths=0.5, cbar_kws={'label': 'Percentage (%)'}, vmin=0, vmax=100)
    
    # Customize chart
    plt.title(f'Rating Distribution by {period_title}', fontsize=16)
    plt.xlabel(period_title, fontsize=14)
    plt.ylabel('Rating', fontsize=14)
    plt.xticks(rotation=45, ha='right', fontsize=12)
    plt.yticks(fontsize=12)
    
    # Add annotation with sample sizes
    period_info = ", ".join([f"{p}: {c}" for p, c in period_totals.items() if c > 0][:5])
    if len(period_totals) > 5:
        period_info += "..."
        
    plt.figtext(0.5, 0.01, 
                f"Numbers represent percentage of ratings within each {time_period.lower()}.\nSample sizes: {period_info}", 
                ha="center", fontsize=12, style='italic')
    
    plt.tight_layout()
    return plt.gcf()


def apply_visualization_theme():
    """
    Apply a consistent theme to all visualizations.
    Call this function before creating visualizations to ensure consistency.
    """
    # Set matplotlib params
    plt.rcParams['figure.figsize'] = (12, 7)
    plt.rcParams['font.size'] = 12
    
    # Set seaborn style
    sns.set(style="whitegrid")
    colors = ['#3498db', '#e74c3c', '#2ecc71', '#f39c12', '#9b59b6', '#1abc9c', '#d35400', '#34495e']
    sns.set_palette(sns.color_palette(colors))


def save_visualization(fig, filepath, format='png', dpi=300):
    """
    Save a visualization to disk.
    
    Args:
        fig: Matplotlib figure to save
        filepath: Full path where the figure should be saved
        format: File format (png, jpg, svg, pdf)
        dpi: Resolution for raster formats
        
    Returns:
        bool: True if successful, False otherwise
    """
    if fig is None:
        print("Error: No figure provided")
        return False
    
    try:
        fig.savefig(filepath, format=format, dpi=dpi, bbox_inches='tight')
        return True
    except Exception as e:
        print(f"Error saving visualization: {e}")
        return False


def create_rating_by_time_chart(df, time_period="month", figsize=(14, 7)):
    """
    Create a line chart showing average rating over time.
    
    Args:
        df: DataFrame containing review data with 'rating' and 'date' columns
        time_period: Time aggregation level ('day', 'week', 'month', 'quarter')
        figsize: Tuple of (width, height) for the figure size
        
    Returns:
        matplotlib.figure.Figure: The generated figure, or None if data is invalid
    """
    # Check required columns
    if 'rating' not in df.columns or 'date' not in df.columns:
        missing = []
        if 'rating' not in df.columns:
            missing.append('rating')
        if 'date' not in df.columns:
            missing.append('date')
        print(f"Required columns not found: {', '.join(missing)}")
        return None
    
    # Ensure date is datetime
    if not pd.api.types.is_datetime64_dtype(df['date']):
        df = df.copy()
        df['date'] = pd.to_datetime(df['date'], errors='coerce')
    
    # Create time period column
    df = df.copy()
    if time_period == "day":
        df['period'] = df['date'].dt.strftime('%Y-%m-%d')
        period_title = "Day"
    elif time_period == "week":
        df['period'] = df['date'].dt.strftime('%Y-%U')
        period_title = "Week"
    elif time_period == "quarter":
        df['period'] = df['date'].dt.year.astype(str) + '-Q' + df['date'].dt.quarter.astype(str)
        period_title = "Quarter"
    else:  # default to month
        df['period'] = df['date'].dt.strftime('%Y-%m')
        period_title = "Month"
    
    # Calculate average rating by period
    rating_by_period = df.groupby('period')['rating'].agg(['mean', 'count']).reset_index()
    rating_by_period = rating_by_period.sort_values('period')
    
    # Create plot
    plt.figure(figsize=figsize)
    
    # Create line plot with markers
    plt.plot(rating_by_period['period'], rating_by_period['mean'], 
             marker='o', linestyle='-', linewidth=2, color='#3498db')
    
    # Add count as marker size
    count_sizes = 50 * rating_by_period['count'] / rating_by_period['count'].max()
    for i, (period, mean, count) in enumerate(zip(rating_by_period['period'], 
                                               rating_by_period['mean'], 
                                               rating_by_period['count'])):
        plt.scatter(period, mean, s=max(50, count_sizes[i]), 
                   alpha=0.7, color='#3498db', edgecolor='white', zorder=10)
        plt.annotate(f"{mean:.2f}", (period, mean), 
                     xytext=(0, 10), textcoords='offset points',
                     ha='center', fontweight='bold')
    
    # Add chart elements
    plt.title(f'Average Rating by {period_title}', fontsize=16)
    plt.xlabel(period_title, fontsize=14)
    plt.ylabel('Average Rating', fontsize=14)
    plt.xticks(rotation=45, ha='right', fontsize=12)
    plt.yticks(fontsize=12)
    plt.grid(linestyle='--', alpha=0.7)
    
    # Set y-axis limits to emphasize changes
    y_min = max(0, rating_by_period['mean'].min() - 0.5)
    y_max = min(5, rating_by_period['mean'].max() + 0.5)
    plt.ylim(y_min, y_max)
    
    # Add annotation about review counts
    plt.figtext(0.5, 0.01, "Note: Marker size indicates number of reviews", 
                ha="center", fontsize=12, style='italic')
    
    plt.tight_layout()
    return plt.gcf()


def create_review_volume_chart(df, time_period="month", figsize=(14, 7)):
    """
    Create a chart showing review volume over time.
    
    Args:
        df: DataFrame containing review data with 'date' column
        time_period: Time aggregation level ('day', 'week', 'month', 'quarter')
        figsize: Tuple of (width, height) for the figure size
        
    Returns:
        matplotlib.figure.Figure: The generated figure, or None if data is invalid
    """
    # Check required columns
    if 'date' not in df.columns:
        print("Required column 'date' not found")
        return None
    
    # Ensure date is datetime
    if not pd.api.types.is_datetime64_dtype(df['date']):
        df = df.copy()
        df['date'] = pd.to_datetime(df['date'], errors='coerce')
    
    # Create time period column
    df = df.copy()
    if time_period == "day":
        df['period'] = df['date'].dt.strftime('%Y-%m-%d')
        period_title = "Day"
    elif time_period == "week":
        df['period'] = df['date'].dt.strftime('%Y-%U')
        period_title = "Week"
    elif time_period == "quarter":
        df['period'] = df['date'].dt.year.astype(str) + '-Q' + df['date'].dt.quarter.astype(str)
        period_title = "Quarter"
    else:  # default to month
        df['period'] = df['date'].dt.strftime('%Y-%m')
        period_title = "Month"
    
    # Count reviews by period
    reviews_by_period = df.groupby('period').size().reset_index(name='count')
    reviews_by_period = reviews_by_period.sort_values('period')
    
    # Create plot
    plt.figure(figsize=figsize)
    
    # Create bar chart
    bars = plt.bar(reviews_by_period['period'], reviews_by_period['count'], 
                  color='#3498db', alpha=0.8)
    
    # Add labels
    for i, count in enumerate(reviews_by_period['count']):
        plt.text(i, count + (reviews_by_period['count'].max() * 0.02), 
                str(count), ha='center', fontweight='bold')
    
    # Add chart elements
    plt.title(f'Review Volume by {period_title}', fontsize=16)
    plt.xlabel(period_title, fontsize=14)
    plt.ylabel('Number of Reviews', fontsize=14)
    plt.xticks(rotation=45, ha='right', fontsize=12)
    plt.yticks(fontsize=12)
    plt.grid(axis='y', linestyle='--', alpha=0.7)
    
    plt.tight_layout()
    return plt.gcf()