"""
Implementation of visualization module using Matplotlib, Seaborn, and Plotly.
"""
import os
from datetime import datetime
from typing import Any, Dict, List, Optional, Tuple, Union

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import seaborn as sns
from plotly.subplots import make_subplots
from wordcloud import WordCloud

from src.config import config
from src.modules.visualization.interface import VisualizationInterface


class DataVisualizer(VisualizationInterface):
    """
    Implementation of visualization interface using various plotting libraries.
    """
    
    def __init__(self, config_override: Optional[Dict[str, Any]] = None):
        """
        Initialize visualization module.
        
        Args:
            config_override: Override for default configuration
        """
        # Initialize attributes with defaults
        self._output_dir = "reports/visualizations"
        self._theme = "dark"
        self._export_formats = ["png", "html"]
        self._fig_width = 12
        self._fig_height = 8
        self._default_cmap = "viridis"
        
        # Call parent constructor
        super().__init__(config_override)
        
        # Set attributes from config after validation
        self._output_dir = self.config.get("output_directory", "reports/visualizations")
        self._theme = self.config.get("theme", "dark")
        self._export_formats = self.config.get("export_formats", ["png", "html"])
        self._fig_width = self.config.get("fig_width", 12)
        self._fig_height = self.config.get("fig_height", 8)
        self._default_cmap = self.config.get("colormap", "viridis")
    
    def _validate_config(self) -> None:
        """
        Validate visualization configuration.
        
        Raises:
            ValueError: If required configuration is missing
        """
        if not self._output_dir:
            try:
                self._output_dir = config.get("visualization", "output_directory")
            except KeyError:
                self._output_dir = "reports/visualizations"  # Default output directory
        
        # Ensure output directory exists
        os.makedirs(self._output_dir, exist_ok=True)
        
        # Validate theme
        valid_themes = ["dark", "light", "default"]
        if self._theme not in valid_themes:
            self._theme = "default"  # Default to default theme
        
        # Validate export formats
        valid_formats = ["png", "jpg", "svg", "pdf", "html"]
        self._export_formats = [fmt for fmt in self._export_formats if fmt in valid_formats]
        if not self._export_formats:
            self._export_formats = ["png"]  # Default to PNG if no valid formats
    
    def initialize(self) -> None:
        """
        Initialize visualization module.
        
        Creates necessary directories and sets up visualization environment.
        
        Raises:
            RuntimeError: If initialization fails
        """
        try:
            # Create output directory
            os.makedirs(self._output_dir, exist_ok=True)
            
            # Set default style for matplotlib
            if self._theme == "dark":
                plt.style.use("dark_background")
            else:
                plt.style.use("default")
            
            # Set default figure size
            plt.rcParams["figure.figsize"] = (self._fig_width, self._fig_height)
            
            # Set default colormap
            plt.rcParams["image.cmap"] = self._default_cmap
            
            # Set Seaborn theme
            sns.set_theme(style="darkgrid" if self._theme == "dark" else "whitegrid")
            
            self.is_initialized = True
        except Exception as e:
            raise RuntimeError(f"Failed to initialize visualization module: {e}")
    
    def _save_figure(
        self, 
        fig: Union[plt.Figure, go.Figure],
        name: str,
        output_dir: Optional[str] = None,
        **kwargs
    ) -> Dict[str, str]:
        """
        Save figure in specified formats.
        
        Args:
            fig: Matplotlib or Plotly figure
            name: Base name for the file
            output_dir: Directory to save the figure in
            
        Returns:
            Dictionary mapping format to file path
        """
        output_dir = output_dir or self._output_dir
        os.makedirs(output_dir, exist_ok=True)
        
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        base_name = f"{name}_{timestamp}"
        
        paths = {}
        
        try:
            if isinstance(fig, plt.Figure):
                # Save Matplotlib figure
                for fmt in self._export_formats:
                    if fmt == "html":
                        continue  # Skip HTML for matplotlib
                    
                    file_path = os.path.join(output_dir, f"{base_name}.{fmt}")
                    fig.savefig(file_path, bbox_inches="tight", dpi=300)
                    paths[fmt] = file_path
            else:
                # Save Plotly figure
                for fmt in self._export_formats:
                    if fmt == "html":
                        file_path = os.path.join(output_dir, f"{base_name}.html")
                        fig.write_html(file_path)
                    else:
                        file_path = os.path.join(output_dir, f"{base_name}.{fmt}")
                        fig.write_image(file_path)
                    
                    paths[fmt] = file_path
        except Exception as e:
            print(f"Error saving figure: {e}")
        
        return paths
    
    def plot_rating_distribution(
        self,
        data: pd.DataFrame,
        rating_column: str = "rating",
        title: Optional[str] = None,
        **kwargs
    ) -> Dict[str, Any]:
        """
        Create a rating distribution visualization.
        
        Args:
            data: DataFrame containing reviews
            rating_column: Column containing ratings
            title: Plot title
            
        Returns:
            Dictionary with visualization information
        """
        if rating_column not in data.columns:
            raise ValueError(f"Rating column '{rating_column}' not found in data")
        
        # Set title
        if title is None:
            title = "Rating Distribution"
        
        # Determine whether to use Plotly or Matplotlib
        use_plotly = kwargs.get("use_plotly", True)
        
        result = {"title": title, "type": "rating_distribution"}
        
        if use_plotly:
            # Create rating distribution with Plotly
            fig = px.histogram(
                data,
                x=rating_column,
                title=title,
                template="plotly_dark" if self._theme == "dark" else "plotly_white",
                color_discrete_sequence=px.colors.sequential.Viridis,
                labels={rating_column: "Rating"},
                nbins=5
            )
            
            # Add average rating line
            avg_rating = data[rating_column].mean()
            fig.add_vline(
                x=avg_rating,
                line_dash="dash",
                line_color="red",
                annotation_text=f"Avg: {avg_rating:.2f}",
                annotation_position="top right"
            )
            
            # Add percentage labels on top of bars
            rating_counts = data[rating_column].value_counts().sort_index()
            total_ratings = rating_counts.sum()
            
            for rating, count in rating_counts.items():
                percentage = count / total_ratings * 100
                fig.add_annotation(
                    x=rating,
                    y=count,
                    text=f"{percentage:.1f}%",
                    showarrow=False,
                    yshift=10
                )
            
            # Update layout
            fig.update_layout(
                bargap=0.2,
                xaxis={"tickmode": "linear", "dtick": 1},
                yaxis_title="Number of Reviews"
            )
            
            # Save figure
            result["file_paths"] = self._save_figure(fig, "rating_distribution", **kwargs)
            result["figure"] = fig
        else:
            # Create rating distribution with Matplotlib
            plt.figure(figsize=(self._fig_width, self._fig_height))
            
            # Get rating counts
            rating_counts = data[rating_column].value_counts().sort_index()
            total_ratings = rating_counts.sum()
            
            # Create bar plot
            ax = sns.barplot(x=rating_counts.index, y=rating_counts.values)
            
            # Add percentage labels on top of bars
            for i, (rating, count) in enumerate(rating_counts.items()):
                percentage = count / total_ratings * 100
                ax.text(
                    i, 
                    count + (count * 0.02), 
                    f"{percentage:.1f}%",
                    ha="center"
                )
            
            # Add average rating line
            avg_rating = data[rating_column].mean()
            plt.axvline(
                avg_rating - 0.5,  # Adjust for bar alignment
                color="red",
                linestyle="--",
                label=f"Avg: {avg_rating:.2f}"
            )
            
            # Set labels and title
            plt.xlabel("Rating")
            plt.ylabel("Number of Reviews")
            plt.title(title)
            plt.legend()
            plt.tight_layout()
            
            # Save figure
            result["file_paths"] = self._save_figure(plt.gcf(), "rating_distribution", **kwargs)
            result["figure"] = plt.gcf()
            
            # Close figure to free memory
            if kwargs.get("close_fig", True):
                plt.close()
        
        return result
    
    def plot_rating_trend(
        self,
        data: pd.DataFrame,
        rating_column: str = "rating",
        date_column: str = "date",
        freq: str = "M",
        title: Optional[str] = None,
        **kwargs
    ) -> Dict[str, Any]:
        """
        Create a rating trend visualization.
        
        Args:
            data: DataFrame containing reviews
            rating_column: Column containing ratings
            date_column: Column containing dates
            freq: Frequency for resampling (D=daily, W=weekly, M=monthly)
            title: Plot title
            
        Returns:
            Dictionary with visualization information
        """
        if rating_column not in data.columns:
            raise ValueError(f"Rating column '{rating_column}' not found in data")
        
        if date_column not in data.columns:
            raise ValueError(f"Date column '{date_column}' not found in data")
        
        # Ensure date column is datetime type
        df = data.copy()
        if not pd.api.types.is_datetime64_any_dtype(df[date_column]):
            df[date_column] = pd.to_datetime(df[date_column], errors="coerce")
        
        # Drop rows with invalid dates
        df = df.dropna(subset=[date_column])
        
        # Set title
        if title is None:
            title = f"Rating Trend ({freq})"
        
        # Determine the date format based on frequency
        date_format = "%Y-%m-%d"
        if freq == "M":
            date_format = "%Y-%m"
        elif freq == "Y":
            date_format = "%Y"
        elif freq == "W":
            date_format = "%Y-%U"
        
        # Group by date and calculate average rating
        df = df.set_index(date_column)
        # For older pandas versions, we need to use 'M' directly instead of 'ME'
        # Let's just override any 'M'-containing frequency with 'M' for compatibility
        if isinstance(freq, str) and 'M' in freq:
            resampled_freq = 'M'
        else:
            resampled_freq = freq
        # Use consistent frequency string for all resampling operations
        rating_trend = df[rating_column].resample(resampled_freq).mean().reset_index()
        rating_trend["period"] = rating_trend[date_column].dt.strftime(date_format)
        
        # Calculate count per period for size reference - use same resampled_freq
        count_trend = df[rating_column].resample(resampled_freq).count().reset_index()
        count_trend.columns = [date_column, "count"]
        
        # Merge the two
        trend_data = pd.merge(rating_trend, count_trend, on=date_column)
        
        # Determine whether to use Plotly or Matplotlib
        use_plotly = kwargs.get("use_plotly", True)
        
        result = {"title": title, "type": "rating_trend"}
        
        if use_plotly:
            # Create rating trend with Plotly
            fig = make_subplots(specs=[[{"secondary_y": True}]])
            
            # Add rating trend line
            fig.add_trace(
                go.Scatter(
                    x=trend_data[date_column],
                    y=trend_data[rating_column],
                    mode="lines+markers",
                    name="Average Rating",
                    line=dict(width=3, color="#6400e0"),
                    marker=dict(size=8)
                ),
                secondary_y=False
            )
            
            # Add count bars if requested
            if kwargs.get("show_counts", True):
                fig.add_trace(
                    go.Bar(
                        x=trend_data[date_column],
                        y=trend_data["count"],
                        name="Number of Reviews",
                        marker_color="rgba(0, 150, 150, 0.6)",
                        opacity=0.7
                    ),
                    secondary_y=True
                )
            
            # Add a horizontal line at rating 3 for reference
            fig.add_hline(
                y=3,
                line_dash="dot",
                line_color="gray",
                annotation_text="Neutral (3)",
                annotation_position="top right"
            )
            
            # Update layout
            fig.update_layout(
                title=title,
                template="plotly_dark" if self._theme == "dark" else "plotly_white",
                legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
                hovermode="x unified"
            )
            
            # Update axes
            fig.update_xaxes(title_text="Date")
            fig.update_yaxes(title_text="Average Rating", range=[1, 5], secondary_y=False)
            fig.update_yaxes(title_text="Number of Reviews", secondary_y=True)
            
            # Save figure
            result["file_paths"] = self._save_figure(fig, "rating_trend", **kwargs)
            result["figure"] = fig
        else:
            # Create rating trend with Matplotlib
            fig, ax1 = plt.subplots(figsize=(self._fig_width, self._fig_height))
            
            # Plot average rating
            color = "tab:blue"
            ax1.set_xlabel("Date")
            ax1.set_ylabel("Average Rating", color=color)
            ax1.plot(
                trend_data[date_column], 
                trend_data[rating_column], 
                color=color, 
                marker="o",
                linewidth=2,
                markersize=6
            )
            ax1.tick_params(axis="y", labelcolor=color)
            ax1.set_ylim(1, 5)  # Ratings are typically 1-5
            
            # Add a horizontal line at rating 3 for reference
            ax1.axhline(
                y=3,
                linestyle=":",
                color="gray",
                alpha=0.7,
                label="Neutral (3)"
            )
            
            # Plot count on secondary axis if requested
            if kwargs.get("show_counts", True):
                ax2 = ax1.twinx()
                color = "tab:green"
                ax2.set_ylabel("Number of Reviews", color=color)
                ax2.bar(
                    trend_data[date_column], 
                    trend_data["count"], 
                    alpha=0.3, 
                    color=color
                )
                ax2.tick_params(axis="y", labelcolor=color)
            
            # Rotate x-axis labels if there are many periods
            if len(trend_data) > 12:
                plt.xticks(rotation=45, ha="right")
            
            # Set title and adjust layout
            plt.title(title)
            plt.tight_layout()
            
            # Add grid for better readability
            ax1.grid(True, alpha=0.3)
            
            # Save figure
            result["file_paths"] = self._save_figure(fig, "rating_trend", **kwargs)
            result["figure"] = fig
            
            # Close figure to free memory
            if kwargs.get("close_fig", True):
                plt.close()
        
        return result
    
    def plot_sentiment_distribution(
        self,
        data: pd.DataFrame,
        sentiment_column: str = "sentiment",
        title: Optional[str] = None,
        **kwargs
    ) -> Dict[str, Any]:
        """
        Create a sentiment distribution visualization.
        
        Args:
            data: DataFrame containing reviews
            sentiment_column: Column containing sentiment values
            title: Plot title
            
        Returns:
            Dictionary with visualization information
        """
        if sentiment_column not in data.columns:
            raise ValueError(f"Sentiment column '{sentiment_column}' not found in data")
        
        # Set title
        if title is None:
            title = "Sentiment Distribution"
        
        # Get sentiment counts
        sentiment_counts = data[sentiment_column].value_counts()
        
        # Determine whether to use Plotly or Matplotlib
        use_plotly = kwargs.get("use_plotly", True)
        
        result = {"title": title, "type": "sentiment_distribution"}
        
        if use_plotly:
            # Create a color map for sentiments
            colors = {
                "positive": "#2ca02c",  # Green
                "negative": "#d62728",  # Red
                "neutral": "#7f7f7f",   # Gray
                "mixed": "#9467bd"      # Purple
            }
            
            # Assign default colors for any other values
            color_values = [colors.get(str(s).lower(), "#1f77b4") for s in sentiment_counts.index]
            
            # Create sentiment distribution with Plotly
            fig = px.pie(
                values=sentiment_counts.values,
                names=sentiment_counts.index,
                title=title,
                template="plotly_dark" if self._theme == "dark" else "plotly_white",
                color=sentiment_counts.index,
                color_discrete_map={str(name): color for name, color in zip(sentiment_counts.index, color_values)}
            )
            
            # Update layout
            fig.update_traces(
                textposition="inside",
                textinfo="percent+label",
                hole=0.4,
                pull=[0.05 if s.lower() == kwargs.get("highlight", "") else 0 for s in sentiment_counts.index]
            )
            
            # Save figure
            result["file_paths"] = self._save_figure(fig, "sentiment_distribution", **kwargs)
            result["figure"] = fig
        else:
            # Create sentiment distribution with Matplotlib
            fig, ax = plt.subplots(figsize=(self._fig_width, self._fig_height))
            
            # Create a color map for sentiments
            colors = {
                "positive": "#2ca02c",  # Green
                "negative": "#d62728",  # Red
                "neutral": "#7f7f7f",   # Gray
                "mixed": "#9467bd"      # Purple
            }
            
            # Assign default colors for any other values
            color_values = [colors.get(str(s).lower(), "#1f77b4") for s in sentiment_counts.index]
            
            # Create the pie chart
            wedges, texts, autotexts = ax.pie(
                sentiment_counts.values,
                labels=sentiment_counts.index,
                autopct="%1.1f%%",
                startangle=90,
                colors=color_values,
                wedgeprops=dict(width=0.4 if kwargs.get("donut", True) else 0, edgecolor="w"),
                textprops=dict(color="black" if self._theme == "light" else "white")
            )
            
            # Set title and adjust layout
            plt.title(title)
            plt.tight_layout()
            
            # Save figure
            result["file_paths"] = self._save_figure(fig, "sentiment_distribution", **kwargs)
            result["figure"] = fig
            
            # Close figure to free memory
            if kwargs.get("close_fig", True):
                plt.close()
        
        return result
    
    def plot_word_cloud(
        self, 
        data: pd.DataFrame, 
        text_column: str = None,
        title: Optional[str] = None,
        figsize: Tuple[int, int] = (12, 8),
        save_path: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        Plot word cloud from text data.
        
        Args:
            data: DataFrame containing text data
            text_column: Column containing text data. If None, will try to find it.
            title: Plot title
            figsize: Figure size
            save_path: Path to save the figure
            
        Returns:
            Dictionary with plot metadata
        """
        # Create reports directory if it doesn't exist
        reports_dir = "reports"
        os.makedirs(reports_dir, exist_ok=True)
        
        # If data is empty, create an empty plot with a message
        if data.empty or len(data) == 0:
            plt.figure(figsize=figsize)
            plt.text(0.5, 0.5, 'No data available for word cloud', 
                     horizontalalignment='center',
                     verticalalignment='center',
                     transform=plt.gca().transAxes, 
                     fontsize=14)
            plt.title(title if title else "Word Cloud", fontsize=15)
            plt.axis('off')
            
            # Save to reports directory
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            file_name = f"word_cloud_{timestamp}.png"
            file_path = os.path.join(reports_dir, file_name)
            plt.savefig(file_path)
            plt.close()
            
            return {
                "title": title if title else "Word Cloud",
                "file_path": file_path,
                "text_column": text_column,
                "empty": True
            }
        
        # Ensure we have a text column
        if text_column is None:
            # Try to find a suitable text column
            text_candidates = ["text", "cleaned_text", "normalized_text", "review_text"]
            for candidate in text_candidates:
                if candidate in data.columns:
                    text_column = candidate
                    break
            
            if text_column is None:
                # Create an empty plot with a message
                plt.figure(figsize=figsize)
                plt.text(0.5, 0.5, 'No suitable text column found for word cloud', 
                         horizontalalignment='center',
                         verticalalignment='center',
                         transform=plt.gca().transAxes, 
                         fontsize=14)
                plt.title(title if title else "Word Cloud", fontsize=15)
                plt.axis('off')
                
                # Save to reports directory
                timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                file_name = f"word_cloud_{timestamp}.png"
                file_path = os.path.join(reports_dir, file_name)
                plt.savefig(file_path)
                plt.close()
                
                return {
                    "title": title if title else "Word Cloud",
                    "file_path": file_path,
                    "text_column": None,
                    "empty": True
                }
        
        # Combine all text
        all_text = " ".join(data[text_column].dropna().astype(str))
        
        # If no text available, create an empty plot with a message
        if not all_text or all_text.strip() == "":
            plt.figure(figsize=figsize)
            plt.text(0.5, 0.5, 'No text data available for word cloud', 
                     horizontalalignment='center',
                     verticalalignment='center',
                     transform=plt.gca().transAxes, 
                     fontsize=14)
            plt.title(title if title else "Word Cloud", fontsize=15)
            plt.axis('off')
            
            # Save to reports directory
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            file_name = f"word_cloud_{timestamp}.png"
            file_path = os.path.join(reports_dir, file_name)
            plt.savefig(file_path)
            plt.close()
            
            return {
                "title": title if title else "Word Cloud",
                "file_path": file_path,
                "text_column": text_column,
                "empty": True
            }
            
        # If we have minimal text, add some dummy words to avoid errors
        if len(all_text.split()) < 5:
            all_text += " sample text for word cloud visualization example words needed"
        
        # Generate word cloud
        wordcloud = WordCloud(
            width=800,
            height=500,
            background_color="white",
            max_words=100,
            min_font_size=10
        ).generate(all_text)
        
        # Plot
        plt.figure(figsize=figsize)
        plt.imshow(wordcloud, interpolation="bilinear")
        plt.axis("off")
        plt.title(title if title else "Word Cloud", fontsize=15)
        plt.tight_layout()
        
        # Save figure if path is provided
        if save_path is not None:
            plt.savefig(save_path)
            result_path = save_path
        else:
            # Create reports directory if it doesn't exist
            reports_dir = "reports"
            os.makedirs(reports_dir, exist_ok=True)
            
            # Save to reports directory
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            file_name = f"word_cloud_{timestamp}.png"
            file_path = os.path.join(reports_dir, file_name)
            plt.savefig(file_path)
            result_path = file_path
        
        plt.close()
        
        return {
            "title": title if title else "Word Cloud",
            "file_path": result_path,
            "text_column": text_column,
            "empty": False
        }
    
    def plot_topic_distribution(
        self,
        data: pd.DataFrame,
        topic_column: str = "primary_topic",
        topic_words: Optional[Dict[int, List[str]]] = None,
        title: Optional[str] = None,
        **kwargs
    ) -> Dict[str, Any]:
        """
        Create a topic distribution visualization.
        
        Args:
            data: DataFrame containing reviews
            topic_column: Column containing topic assignments
            topic_words: Dictionary mapping topic IDs to representative words
            title: Plot title
            
        Returns:
            Dictionary with visualization information
        """
        if topic_column not in data.columns:
            raise ValueError(f"Topic column '{topic_column}' not found in data")
        
        # Set title
        if title is None:
            title = "Topic Distribution"
        
        # Get topic counts
        topic_counts = data[topic_column].value_counts().sort_index()
        
        # Create topic labels
        if topic_words:
            topic_labels = {}
            for topic_id, words in topic_words.items():
                if words:
                    label = f"Topic {topic_id}: {', '.join(words[:3])}"
                    topic_labels[topic_id] = label
                else:
                    topic_labels[topic_id] = f"Topic {topic_id}"
        else:
            topic_labels = {topic_id: f"Topic {topic_id}" for topic_id in topic_counts.index}
        
        # Determine whether to use Plotly or Matplotlib
        use_plotly = kwargs.get("use_plotly", True)
        
        result = {"title": title, "type": "topic_distribution"}
        
        if use_plotly:
            # Create a DataFrame for plotting
            plot_df = pd.DataFrame({
                "topic": [topic_labels.get(idx, f"Topic {idx}") for idx in topic_counts.index],
                "count": topic_counts.values
            })
            
            # Create topic distribution with Plotly
            fig = px.bar(
                plot_df,
                x="topic",
                y="count",
                title=title,
                template="plotly_dark" if self._theme == "dark" else "plotly_white",
                color="count",
                color_continuous_scale=px.colors.sequential.Viridis
            )
            
            # Update layout
            fig.update_layout(
                xaxis_title="Topic",
                yaxis_title="Number of Reviews",
                showlegend=False
            )
            
            # Add percentage labels
            total = topic_counts.sum()
            for i, (topic_id, count) in enumerate(topic_counts.items()):
                topic_label = topic_labels.get(topic_id, f"Topic {topic_id}")
                percentage = count / total * 100
                fig.add_annotation(
                    x=topic_label,
                    y=count,
                    text=f"{percentage:.1f}%",
                    showarrow=False,
                    yshift=10
                )
            
            # Save figure
            result["file_paths"] = self._save_figure(fig, "topic_distribution", **kwargs)
            result["figure"] = fig
        else:
            # Create topic distribution with Matplotlib
            fig, ax = plt.subplots(figsize=(self._fig_width, self._fig_height))
            
            # Create the bar chart
            bars = ax.bar(range(len(topic_counts)), topic_counts.values)
            
            # Set topic labels
            plt.xticks(
                range(len(topic_counts)),
                [topic_labels.get(idx, f"Topic {idx}") for idx in topic_counts.index],
                rotation=45,
                ha="right"
            )
            
            # Add percentage labels
            total = topic_counts.sum()
            for i, (topic_id, count) in enumerate(topic_counts.items()):
                percentage = count / total * 100
                ax.text(
                    i, 
                    count + (count * 0.02), 
                    f"{percentage:.1f}%",
                    ha="center"
                )
            
            # Set labels and title
            plt.xlabel("Topic")
            plt.ylabel("Number of Reviews")
            plt.title(title)
            plt.tight_layout()
            
            # Save figure
            result["file_paths"] = self._save_figure(fig, "topic_distribution", **kwargs)
            result["figure"] = fig
            
            # Close figure to free memory
            if kwargs.get("close_fig", True):
                plt.close()
        
        return result
    
    def create_dashboard(
        self,
        data: pd.DataFrame,
        output_path: Optional[str] = None,
        include_plots: Optional[List[str]] = None,
        title: str = "Review Analysis Dashboard",
        **kwargs
    ) -> Dict[str, Any]:
        """
        Create a dashboard with multiple visualizations.
        
        Args:
            data: DataFrame containing reviews
            output_path: Path to save the dashboard
            include_plots: List of plots to include
            title: Dashboard title
            
        Returns:
            Dictionary with dashboard information
        """
        # Define default plots to include
        if include_plots is None:
            include_plots = [
                "rating_distribution",
                "rating_trend",
                "sentiment_distribution",
                "word_cloud"
            ]
            
            # Add topic distribution if available
            if "primary_topic" in data.columns:
                include_plots.append("topic_distribution")
        
        # Set output path
        if output_path is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            output_path = os.path.join(self._output_dir, f"dashboard_{timestamp}.html")
        
        # Check if data is empty
        if data.empty:
            # Create a simple HTML message for empty dashboard
            html_content = f"""
            <!DOCTYPE html>
            <html>
            <head>
                <title>{title}</title>
                <style>
                    body {{ font-family: Arial, sans-serif; margin: 40px; text-align: center; }}
                    .message {{ padding: 30px; border: 1px solid #ddd; border-radius: 10px; }}
                    h1 {{ color: #333; }}
                </style>
            </head>
            <body>
                <h1>{title}</h1>
                <div class="message">
                    <h2>No data available for visualization</h2>
                    <p>The dashboard could not be generated because no review data is available.</p>
                </div>
            </body>
            </html>
            """
            
            # Create output directory if it doesn't exist
            os.makedirs(os.path.dirname(output_path), exist_ok=True)
            
            # Write the HTML file
            with open(output_path, 'w') as file:
                file.write(html_content)
            
            return {
                "title": title,
                "type": "dashboard",
                "file_path": output_path,
                "empty": True,
                "plots": {}
            }
        
        # Determine dashboard dimensions
        num_plots = len(include_plots)
        rows = min(3, num_plots)
        cols = (num_plots // rows) + (1 if num_plots % rows else 0)
        
        # Create empty dashboard with subplots, with specs for special plot types
        specs = []
        for i in range(rows):
            row_specs = []
            for j in range(cols):
                # Default to xy plot type
                row_specs.append({"type": "xy"})
            specs.append(row_specs)
        
        fig = make_subplots(
            rows=rows,
            cols=cols,
            subplot_titles=[p.replace("_", " ").title() for p in include_plots],
            vertical_spacing=0.1,
            horizontal_spacing=0.05,
            specs=specs
        )
        
        # Generate each plot and add to dashboard
        plot_index = 0
        plot_results = {}
        errors = []
        
        for plot_type in include_plots:
            # Calculate row and column
            row = (plot_index // cols) + 1
            col = (plot_index % cols) + 1
            
            try:
                # Generate plot based on type
                if plot_type == "rating_distribution":
                    # Skip if no rating column
                    if "rating" not in data.columns:
                        continue
                        
                    result = self.plot_rating_distribution(
                        data, 
                        use_plotly=True,
                        close_fig=True,
                        **kwargs
                    )
                    
                    # Add to dashboard
                    for trace in result["figure"].data:
                        fig.add_trace(trace, row=row, col=col)
                
                elif plot_type == "rating_trend":
                    # Skip if required columns are missing
                    if "rating" not in data.columns or "date" not in data.columns:
                        continue
                        
                    result = self.plot_rating_trend(
                        data, 
                        use_plotly=True,
                        close_fig=True,
                        **kwargs
                    )
                    
                    # Add to dashboard
                    for trace in result["figure"].data:
                        fig.add_trace(trace, row=row, col=col)
                
                elif plot_type == "sentiment_distribution":
                    # Check if sentiment column exists
                    if "sentiment" in data.columns:
                        try:
                            # Instead of adding pie chart directly to xy-subplot (which causes error),
                            # create a text annotation explaining the issue
                            fig.add_annotation(
                                x=0.5, y=0.5,
                                text="Sentiment Distribution",
                                showarrow=False,
                                font=dict(size=16),
                                row=row, col=col
                            )
                            
                            fig.add_annotation(
                                x=0.5, y=0.3,
                                text="View in separate chart",
                                showarrow=False,
                                font=dict(size=12, color="gray"),
                                row=row, col=col
                            )
                            
                            # Create a separate sentiment chart
                            sentiment_result = self.plot_sentiment_distribution(
                                data, 
                                use_plotly=True,
                                close_fig=False,
                                **kwargs
                            )
                            
                            # Save the separate sentiment chart
                            plot_results["sentiment_distribution"] = sentiment_result
                            
                            # Also save it as a standalone file
                            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                            sentiment_path = os.path.join(
                                self._output_dir, 
                                f"sentiment_distribution_dashboard_{timestamp}.html"
                            )
                            sentiment_result["figure"].write_html(sentiment_path)
                            
                        except Exception as e:
                            error_msg = f"Error generating sentiment distribution: {str(e)}"
                            errors.append(error_msg)
                            print(error_msg)
                    else:
                        continue
                
                elif plot_type == "word_cloud":
                    # Word cloud is not compatible with Plotly dashboard
                    # Generate separately and save
                    # Filter out kwargs that aren't compatible with plot_word_cloud
                    word_cloud_kwargs = {k: v for k, v in kwargs.items() 
                                        if k not in ['topic_words']}
                    result = self.plot_word_cloud(
                        data,
                        **word_cloud_kwargs
                    )
                    
                    # Skip adding to dashboard
                    plot_results[plot_type] = result
                    continue
                
                elif plot_type == "topic_distribution":
                    # Check if topic column exists
                    if "primary_topic" in data.columns:
                        # Get topic words if available
                        topic_words = kwargs.get("topic_words", None)
                        
                        # Create a copy of kwargs without topic_words to avoid duplicate argument
                        filtered_kwargs = {k: v for k, v in kwargs.items() if k != "topic_words"}
                        
                        result = self.plot_topic_distribution(
                            data, 
                            topic_words=topic_words,
                            use_plotly=True,
                            close_fig=True,
                            **filtered_kwargs
                        )
                        
                        # Add to dashboard
                        for trace in result["figure"].data:
                            fig.add_trace(trace, row=row, col=col)
                    else:
                        continue
                
                # Store plot result
                plot_results[plot_type] = result
                plot_index += 1
                
            except Exception as e:
                error_msg = f"Error generating plot '{plot_type}': {e}"
                print(error_msg)
                errors.append(error_msg)
                continue
            
        # If no plots were successfully created, create a simple HTML message
        if plot_index == 0:
            html_content = f"""
            <!DOCTYPE html>
            <html>
            <head>
                <title>{title}</title>
                <style>
                    body {{ font-family: Arial, sans-serif; margin: 40px; text-align: center; }}
                    .message {{ padding: 30px; border: 1px solid #ddd; border-radius: 10px; }}
                    h1 {{ color: #333; }}
                    .error {{ color: #c00; }}
                </style>
            </head>
            <body>
                <h1>{title}</h1>
                <div class="message">
                    <h2>No visualizations could be generated</h2>
                    <p>The dashboard could not create any visualizations with the available data.</p>
                    {"<div class='error'><p>Errors encountered:</p><ul>" + "".join([f"<li>{e}</li>" for e in errors]) + "</ul></div>" if errors else ""}
                </div>
            </body>
            </html>
            """
            
            # Create output directory if it doesn't exist
            os.makedirs(os.path.dirname(output_path), exist_ok=True)
            
            # Write the HTML file
            with open(output_path, 'w') as file:
                file.write(html_content)
            
            return {
                "title": title,
                "type": "dashboard",
                "file_path": output_path,
                "empty": True,
                "plots": plot_results,
                "errors": errors
            }
        
        # Update layout for regular dashboard
        fig.update_layout(
            title_text=title,
            template="plotly_dark" if self._theme == "dark" else "plotly_white",
            height=max(300 * rows, 600),  # Minimum height of 600px
            width=max(500 * cols, 800),   # Minimum width of 800px
            showlegend=False
        )
        
        # Save dashboard
        try:
            # Create output directory if it doesn't exist
            os.makedirs(os.path.dirname(output_path), exist_ok=True)
            
            fig.write_html(output_path)
            file_path = output_path
        except Exception as e:
            error_msg = f"Error saving dashboard: {e}"
            print(error_msg)
            errors.append(error_msg)
            file_path = None
        
        return {
            "title": title,
            "type": "dashboard",
            "file_path": file_path,
            "plots": plot_results,
            "errors": errors if errors else None
        }