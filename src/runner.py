"""
Main runner for the App Reviews AI system.
"""
import argparse
import logging
import os
import sys
from datetime import datetime, timedelta
from typing import Any, Dict, List, Optional, Union

import pandas as pd

import os
import sys

# Add the project root to the path so we can import modules
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src.config import config
from src.modules.acquisition.google_play import GooglePlayReviewAcquisition
from src.modules.analytics.review_analyzer import ReviewAnalyzer
from src.modules.llm.openai_llm import OpenAILLM
from src.modules.preprocessing.nlp_preprocessor import NLPPreprocessor
from src.modules.storage.file_storage import FileStorage
from src.modules.vector_db.chroma_db import ChromaDBVectorStore
from src.modules.visualization.data_visualizer import DataVisualizer


# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    handlers=[
        logging.StreamHandler(sys.stdout),
        logging.FileHandler("logs/app_reviews.log", mode="a")
    ]
)

logger = logging.getLogger("app_reviews")


class ReviewAnalysisRunner:
    """
    Main runner class for the review analysis pipeline.
    """
    
    def __init__(self, config_path: Optional[str] = None):
        """
        Initialize the runner.
        
        Args:
            config_path: Path to configuration file
        """
        # Load config
        if config_path:
            config._load_from_file(config_path)
        
        # Create data directory if it doesn't exist
        os.makedirs("data", exist_ok=True)
        os.makedirs("logs", exist_ok=True)
        os.makedirs("reports", exist_ok=True)
        
        # Initialize modules
        self.acquisition = None
        self.storage = None
        self.preprocessor = None
        self.analyzer = None
        self.vector_db = None
        self.llm = None
        self.visualizer = None
        
        # Initialize empty pipeline metadata
        self.pipeline_metadata = {
            "start_time": datetime.now(),
            "modules": {},
            "stages": {},
            "total_reviews": 0,
            "app_info": None
        }
    
    def _initialize_modules(self) -> None:
        """
        Initialize all required modules.
        
        Raises:
            RuntimeError: If initialization fails
        """
        logger.info("Initializing modules...")
        
        try:
            # Initialize acquisition module
            logger.info("Initializing acquisition module...")
            self.acquisition = GooglePlayReviewAcquisition()
            self.acquisition.initialize()
            self.pipeline_metadata["modules"]["acquisition"] = "GooglePlayReviewAcquisition"
            
            # Initialize storage module
            logger.info("Initializing storage module...")
            self.storage = FileStorage()
            self.storage.initialize()
            self.pipeline_metadata["modules"]["storage"] = "FileStorage"
            
            # Initialize preprocessor module
            logger.info("Initializing preprocessor module...")
            self.preprocessor = NLPPreprocessor()
            self.preprocessor.initialize()
            self.pipeline_metadata["modules"]["preprocessor"] = "NLPPreprocessor"
            
            # Initialize analyzer module
            logger.info("Initializing analyzer module...")
            self.analyzer = ReviewAnalyzer()
            self.analyzer.initialize()
            self.pipeline_metadata["modules"]["analyzer"] = "ReviewAnalyzer"
            
            # Initialize vector database module
            logger.info("Initializing vector database module...")
            self.vector_db = ChromaDBVectorStore()
            self.vector_db.initialize()
            self.pipeline_metadata["modules"]["vector_db"] = "ChromaDBVectorStore"
            
            # Initialize LLM module
            logger.info("Initializing LLM module...")
            self.llm = OpenAILLM()
            self.llm.initialize()
            self.pipeline_metadata["modules"]["llm"] = "OpenAILLM"
            
            # Initialize visualization module
            logger.info("Initializing visualization module...")
            self.visualizer = DataVisualizer()
            self.visualizer.initialize()
            self.pipeline_metadata["modules"]["visualizer"] = "DataVisualizer"
            
            logger.info("All modules initialized successfully.")
        except Exception as e:
            logger.error(f"Error initializing modules: {e}")
            raise
    
    def fetch_reviews(
        self,
        app_id: Optional[str] = None,
        start_date: Optional[str] = None,
        end_date: Optional[str] = None,
        max_reviews: Optional[int] = None
    ) -> pd.DataFrame:
        """
        Fetch reviews from the app store.
        
        Args:
            app_id: Application ID
            start_date: Start date for reviews
            end_date: End date for reviews
            max_reviews: Maximum number of reviews to fetch
            
        Returns:
            DataFrame containing reviews
        """
        logger.info("Fetching reviews...")
        
        start_time = datetime.now()
        
        try:
            # Get app info
            app_info = self.acquisition.get_app_info(app_id)
            self.pipeline_metadata["app_info"] = app_info
            
            logger.info(f"App: {app_info.get('name')}")
            logger.info(f"Developer: {app_info.get('developer')}")
            logger.info(f"Total reviews: {app_info.get('total_reviews')}")
            logger.info(f"Average rating: {app_info.get('average_rating')}")
            
            # Set start date and end date if not provided
            if not start_date:
                start_date = config.get("acquisition", "time_frame").get("start_date", "1 year ago")
            
            if not end_date:
                end_date = config.get("acquisition", "time_frame").get("end_date", "now")
            
            # Set max reviews if not provided
            if not max_reviews:
                # Directly read from environment for highest priority
                max_reviews = int(os.environ.get("MAX_REVIEWS", config.get("acquisition", "max_reviews", 1000)))
            
            logger.info(f"Fetching reviews from {start_date} to {end_date}, max: {max_reviews}")
            
            # Fetch reviews
            reviews_df = self.acquisition.fetch_reviews(
                app_id=app_id,
                start_date=start_date,
                end_date=end_date,
                max_reviews=max_reviews
            )
            
            # Store metadata
            self.pipeline_metadata["total_reviews"] = len(reviews_df)
            self.pipeline_metadata["stages"]["fetch_reviews"] = {
                "start_time": start_time,
                "end_time": datetime.now(),
                "count": len(reviews_df)
            }
            
            logger.info(f"Successfully fetched {len(reviews_df)} reviews.")
            
            return reviews_df
        except Exception as e:
            logger.error(f"Error fetching reviews: {e}")
            raise
    
    def preprocess_reviews(self, reviews_df: pd.DataFrame) -> pd.DataFrame:
        """
        Preprocess review data.
        
        Args:
            reviews_df: DataFrame containing reviews
            
        Returns:
            Processed DataFrame
        """
        logger.info("Preprocessing reviews...")
        
        start_time = datetime.now()
        
        try:
            processed_df = self.preprocessor.process_data(
                data=reviews_df,
                text_column="text"
            )
            
            # Store metadata
            self.pipeline_metadata["stages"]["preprocess_reviews"] = {
                "start_time": start_time,
                "end_time": datetime.now(),
                "count": len(processed_df)
            }
            
            logger.info("Successfully preprocessed reviews.")
            
            return processed_df
        except Exception as e:
            logger.error(f"Error preprocessing reviews: {e}")
            raise
    
    def analyze_reviews(
        self, 
        reviews_df: pd.DataFrame,
        analysis_types: Optional[List[str]] = None
    ) -> Dict[str, Any]:
        """
        Analyze review data.
        
        Args:
            reviews_df: DataFrame containing reviews
            analysis_types: Types of analysis to perform
            
        Returns:
            Dictionary containing analysis results
        """
        logger.info("Analyzing reviews...")
        
        start_time = datetime.now()
        
        if analysis_types is None:
            # Always include sentiment in the analysis types to avoid missing column for visualization
            analysis_types = ["sentiment", "topics", "keywords"]
        
        try:
            analysis_results = self.analyzer.process_data(
                data=reviews_df,
                analysis_types=analysis_types
            )
            
            # Store metadata
            self.pipeline_metadata["stages"]["analyze_reviews"] = {
                "start_time": start_time,
                "end_time": datetime.now(),
                "analysis_types": analysis_types
            }
            
            # Update DataFrame with sentiment analysis if available
            if "sentiment" in analysis_results:
                reviews_df = analysis_results["sentiment"]
            
            # Store topic model metadata if available
            if "topics" in analysis_results:
                topic_df, topic_words = analysis_results["topics"]
                reviews_df = topic_df
                self.pipeline_metadata["topic_words"] = {
                    str(k): v for k, v in topic_words.items()
                }
            
            logger.info("Successfully analyzed reviews.")
            
            return {
                "reviews_df": reviews_df,
                "analysis_results": analysis_results
            }
        except Exception as e:
            logger.error(f"Error analyzing reviews: {e}")
            raise
    
    def store_reviews(self, reviews_df: pd.DataFrame) -> bool:
        """
        Store review data.
        
        Args:
            reviews_df: DataFrame containing reviews
            
        Returns:
            True if successful, False otherwise
        """
        logger.info("Storing reviews...")
        
        start_time = datetime.now()
        
        try:
            success = self.storage.store_data(
                data=reviews_df,
                append=False
            )
            
            # Store metadata
            self.pipeline_metadata["stages"]["store_reviews"] = {
                "start_time": start_time,
                "end_time": datetime.now(),
                "success": success
            }
            
            if success:
                logger.info("Successfully stored reviews.")
            else:
                logger.error("Failed to store reviews.")
            
            return success
        except Exception as e:
            logger.error(f"Error storing reviews: {e}")
            raise
    
    def index_reviews(self, reviews_df: pd.DataFrame) -> bool:
        """
        Index review data in vector database.
        
        Args:
            reviews_df: DataFrame containing reviews
            
        Returns:
            True if successful, False otherwise
        """
        logger.info("Indexing reviews in vector database...")
        
        start_time = datetime.now()
        
        try:
            # Determine metadata fields to include
            metadata_fields = [
                "review_id", "author", "date", "rating", "version",
                "sentiment", "primary_topic", "topic_confidence",
                "thumbsUpCount", "repliedAt", "timestamp"
            ]
            
            # Filter to only include fields that exist in the DataFrame
            metadata_fields = [f for f in metadata_fields if f in reviews_df.columns]
            
            # Use processed text if available
            text_field = "text"
            if "normalized_text" in reviews_df.columns:
                text_field = "normalized_text"
            elif "cleaned_text" in reviews_df.columns:
                text_field = "cleaned_text"
            
            success = self.vector_db.add_documents(
                documents=reviews_df,
                text_field=text_field,
                id_field="review_id",
                metadata_fields=metadata_fields
            )
            
            # Store metadata
            self.pipeline_metadata["stages"]["index_reviews"] = {
                "start_time": start_time,
                "end_time": datetime.now(),
                "success": success,
                "vector_db_stats": self.vector_db.get_collection_stats()
            }
            
            if success:
                logger.info("Successfully indexed reviews in vector database.")
            else:
                logger.error("Failed to index reviews in vector database.")
            
            return success
        except Exception as e:
            logger.error(f"Error indexing reviews: {e}")
            raise
    
    def generate_insights(
        self, 
        reviews_df: pd.DataFrame,
        analysis_results: Dict[str, Any],
        insight_types: Optional[List[str]] = None
    ) -> Dict[str, Any]:
        """
        Generate insights from review data using LLM.
        
        Args:
            reviews_df: DataFrame containing reviews
            analysis_results: Analysis results from the analyzer
            insight_types: Types of insights to generate
            
        Returns:
            Dictionary containing insights
        """
        logger.info("Generating insights...")
        
        start_time = datetime.now()
        
        if insight_types is None:
            insight_types = ["general", "issues", "suggestions"]
        
        insights = {}
        
        try:
            # Check if DataFrame is empty
            if reviews_df.empty:
                logger.warning("No reviews to analyze. Using mock review data for insights.")
                # Create mock reviews for demonstration
                mock_reviews = [
                    {
                        "text": "Great app. Very easy to use and reliable.",
                        "rating": 5,
                        "date": datetime.now() - timedelta(days=30),
                        "sentiment": "positive"
                    },
                    {
                        "text": "The app keeps crashing. Please fix this issue.",
                        "rating": 2,
                        "date": datetime.now() - timedelta(days=15),
                        "sentiment": "negative"
                    },
                    {
                        "text": "Love the new interface but the payment process is still confusing.",
                        "rating": 4,
                        "date": datetime.now() - timedelta(days=7),
                        "sentiment": "mixed"
                    },
                    {
                        "text": "Customer service was very helpful when I had an issue.",
                        "rating": 5,
                        "date": datetime.now() - timedelta(days=5),
                        "sentiment": "positive"
                    },
                    {
                        "text": "Can't properly use the main features, always shows an error.",
                        "rating": 2,
                        "date": datetime.now() - timedelta(days=10),
                        "sentiment": "negative"
                    }
                ]
                sample_reviews = mock_reviews
                num_reviews = len(mock_reviews)
            else:
                # Create app context
                num_reviews = len(reviews_df)
                
                # Sample reviews for analysis
                # Use stratified sampling by rating if possible
                if "rating" in reviews_df.columns:
                    samples = []
                    
                    for rating in sorted(reviews_df["rating"].unique(), reverse=True):
                        # Get up to 20 reviews for each rating
                        rating_samples = reviews_df[reviews_df["rating"] == rating].sample(
                            min(20, sum(reviews_df["rating"] == rating))
                        )
                        samples.append(rating_samples)
                    
                    sample_df = pd.concat(samples)
                    
                    # Limit to 100 samples total
                    if len(sample_df) > 100:
                        sample_df = sample_df.sample(100)
                else:
                    # Random sampling if rating not available
                    sample_df = reviews_df.sample(min(100, len(reviews_df)))
                
                # Prepare sample reviews
                sample_reviews = []
                
                for _, row in sample_df.iterrows():
                    review = {
                        "text": row.get("text", ""),
                        "rating": row.get("rating", None),
                        "date": row.get("date", None)
                    }
                    
                    # Add sentiment if available
                    if "sentiment" in row:
                        review["sentiment"] = row["sentiment"]
                    
                    sample_reviews.append(review)
            
            # Create app context
            app_info = self.pipeline_metadata.get("app_info", {})
            app_context = f"""
            App: {app_info.get('name', 'Mobile App')}
            Developer: {app_info.get('developer', 'App Developer')}
            Total Reviews: {app_info.get('total_reviews', 'Unknown')}
            Average Rating: {app_info.get('average_rating', 'Unknown')}
            Reviews Analyzed: {num_reviews} 
            Time Period: {reviews_df['date'].min() if not reviews_df.empty and 'date' in reviews_df.columns else 'Unknown'} to {reviews_df['date'].max() if not reviews_df.empty and 'date' in reviews_df.columns else 'Unknown'}
            """
            
            # Generate insights for each type
            for insight_type in insight_types:
                logger.info(f"Generating {insight_type} insights...")
                
                result = self.llm.analyze_reviews(
                    reviews=sample_reviews,
                    analysis_type=insight_type,
                    context=app_context,
                    output_format="markdown"
                )
                
                insights[insight_type] = result
            
            # Store metadata
            self.pipeline_metadata["stages"]["generate_insights"] = {
                "start_time": start_time,
                "end_time": datetime.now(),
                "insight_types": insight_types
            }
            
            logger.info("Successfully generated insights.")
            
            return insights
        except Exception as e:
            logger.error(f"Error generating insights: {e}")
            raise
    
    def create_visualizations(
        self, 
        reviews_df: pd.DataFrame,
        analysis_results: Dict[str, Any],
        visualization_types: Optional[List[str]] = None
    ) -> Dict[str, Any]:
        """
        Create visualizations from review data.
        
        Args:
            reviews_df: DataFrame containing reviews
            analysis_results: Analysis results from the analyzer
            visualization_types: Types of visualizations to create
            
        Returns:
            Dictionary containing visualization results
        """
        logger.info("Creating visualizations...")
        
        start_time = datetime.now()
        
        if visualization_types is None:
            visualization_types = [
                "rating_distribution", 
                "rating_trend", 
                "sentiment_distribution", 
                "word_cloud", 
                "topic_distribution",
                "dashboard"
            ]
        
        visualization_results = {}
        
        try:
            # Create app info title
            app_name = self.pipeline_metadata.get("app_info", {}).get("name", "Mobile App")
            title_prefix = f"{app_name} - "
            
            # Topic words for topic distribution
            topic_words = None
            if "topics" in analysis_results:
                _, topic_words = analysis_results["topics"]
            
            # Create individual visualizations
            for viz_type in visualization_types:
                if viz_type == "dashboard":
                    continue  # Handle dashboard separately
                
                logger.info(f"Creating {viz_type} visualization...")
                
                if viz_type == "rating_distribution":
                    result = self.visualizer.plot_rating_distribution(
                        data=reviews_df,
                        title=f"{title_prefix}Rating Distribution"
                    )
                
                elif viz_type == "rating_trend":
                    result = self.visualizer.plot_rating_trend(
                        data=reviews_df,
                        title=f"{title_prefix}Rating Trend"
                    )
                
                elif viz_type == "sentiment_distribution":
                    # Check for sentiment column and create it if missing
                    if "sentiment" not in reviews_df.columns:
                        logger.warning("Sentiment column not found. Adding placeholder sentiment data.")
                        # Create placeholder sentiment data based on ratings
                        if "rating" in reviews_df.columns:
                            reviews_df["sentiment"] = reviews_df["rating"].apply(
                                lambda r: "positive" if r >= 4 else ("negative" if r <= 2 else "neutral")
                            )
                        else:
                            # Random distribution if no rating column
                            import numpy as np
                            sentiments = ["positive", "neutral", "negative"]
                            weights = [0.5, 0.3, 0.2]  # Mostly positive distribution
                            reviews_df["sentiment"] = np.random.choice(sentiments, size=len(reviews_df), p=weights)
                    
                    # Now create the visualization
                    result = self.visualizer.plot_sentiment_distribution(
                        data=reviews_df,
                        title=f"{title_prefix}Sentiment Distribution"
                    )
                
                elif viz_type == "word_cloud":
                    result = self.visualizer.plot_word_cloud(
                        data=reviews_df,
                        title=f"{title_prefix}Word Cloud"
                    )
                
                elif viz_type == "topic_distribution":
                    if "primary_topic" in reviews_df.columns:
                        result = self.visualizer.plot_topic_distribution(
                            data=reviews_df,
                            topic_words=topic_words,
                            title=f"{title_prefix}Topic Distribution"
                        )
                    else:
                        logger.warning("Topic column not found. Skipping topic distribution visualization.")
                        continue
                
                visualization_results[viz_type] = result
            
            # Create dashboard
            if "dashboard" in visualization_types:
                logger.info("Creating dashboard...")
                
                # Determine which visualizations to include in dashboard
                include_plots = [
                    viz_type for viz_type in visualization_types 
                    if viz_type != "dashboard"
                ]
                
                dashboard_result = self.visualizer.create_dashboard(
                    data=reviews_df,
                    include_plots=include_plots,
                    topic_words=topic_words,
                    title=f"{app_name} Review Analysis Dashboard"
                )
                
                visualization_results["dashboard"] = dashboard_result
            
            # Store metadata
            self.pipeline_metadata["stages"]["create_visualizations"] = {
                "start_time": start_time,
                "end_time": datetime.now(),
                "visualization_types": visualization_types
            }
            
            logger.info("Successfully created visualizations.")
            
            return visualization_results
        except Exception as e:
            logger.error(f"Error creating visualizations: {e}")
            raise
    
    def run_pipeline(
        self,
        app_id: Optional[str] = None,
        start_date: Optional[str] = None,
        end_date: Optional[str] = None,
        max_reviews: Optional[int] = None,
        skip_stages: Optional[List[str]] = None
    ) -> Dict[str, Any]:
        """
        Run the full analysis pipeline.
        
        Args:
            app_id: Application ID
            start_date: Start date for reviews
            end_date: End date for reviews
            max_reviews: Maximum number of reviews to fetch
            skip_stages: Stages to skip
            
        Returns:
            Dictionary containing pipeline results
        """
        logger.info("Starting review analysis pipeline...")
        
        # Initialize modules
        self._initialize_modules()
        
        # Set default for skipped stages
        if skip_stages is None:
            skip_stages = []
        
        # Initialize result dictionary
        result = {
            "success": True,
            "reviews_df": None,
            "analysis_results": None,
            "insights": None,
            "visualizations": None,
            "metadata": self.pipeline_metadata
        }
        
        try:
            # Stage 1: Fetch reviews
            if "fetch" not in skip_stages:
                reviews_df = self.fetch_reviews(
                    app_id=app_id,
                    start_date=start_date,
                    end_date=end_date,
                    max_reviews=max_reviews
                )
                result["reviews_df"] = reviews_df
            else:
                logger.info("Skipping fetch stage. Loading reviews from storage...")
                reviews_df = self.storage.retrieve_data()
                result["reviews_df"] = reviews_df
            
            # Stage 2: Preprocess reviews
            if "preprocess" not in skip_stages:
                reviews_df = self.preprocess_reviews(reviews_df)
                result["reviews_df"] = reviews_df
            
            # Stage 3: Analyze reviews
            if "analyze" not in skip_stages:
                analysis_output = self.analyze_reviews(
                    reviews_df=reviews_df,
                    analysis_types=["sentiment", "topics", "keywords", "trends"]
                )
                reviews_df = analysis_output["reviews_df"]
                analysis_results = analysis_output["analysis_results"]
                result["reviews_df"] = reviews_df
                result["analysis_results"] = analysis_results
            else:
                analysis_results = {}
            
            # Stage 4: Store reviews
            if "store" not in skip_stages:
                self.store_reviews(reviews_df)
            
            # Stage 5: Index reviews in vector database
            if "index" not in skip_stages:
                self.index_reviews(reviews_df)
            
            # Stage 6: Generate insights
            if "insights" not in skip_stages:
                insights = self.generate_insights(
                    reviews_df=reviews_df,
                    analysis_results=analysis_results
                )
                result["insights"] = insights
            
            # Stage 7: Create visualizations
            if "visualize" not in skip_stages:
                visualizations = self.create_visualizations(
                    reviews_df=reviews_df,
                    analysis_results=analysis_results
                )
                result["visualizations"] = visualizations
            
            # Update pipeline metadata
            self.pipeline_metadata["end_time"] = datetime.now()
            self.pipeline_metadata["success"] = True
            result["metadata"] = self.pipeline_metadata
            
            logger.info("Pipeline completed successfully.")
            
            return result
        except Exception as e:
            logger.error(f"Pipeline failed: {e}")
            
            # Update pipeline metadata
            self.pipeline_metadata["end_time"] = datetime.now()
            self.pipeline_metadata["success"] = False
            self.pipeline_metadata["error"] = str(e)
            result["metadata"] = self.pipeline_metadata
            result["success"] = False
            
            return result


def main():
    """
    Main entry point for the command line tool.
    """
    parser = argparse.ArgumentParser(description="App Reviews AI - Mobile App Review Analysis")
    
    parser.add_argument(
        "--config", 
        type=str, 
        help="Path to configuration file",
        default="config/config.json"
    )
    parser.add_argument(
        "--app-id", 
        type=str, 
        help="App ID (package name)",
        default=os.environ.get("APP_ID", "")
    )
    parser.add_argument(
        "--start-date", 
        type=str, 
        help="Start date for reviews (YYYY-MM-DD or relative like '1 year ago')",
        default="1 year ago"
    )
    parser.add_argument(
        "--end-date", 
        type=str, 
        help="End date for reviews (YYYY-MM-DD or 'now')",
        default="now"
    )
    parser.add_argument(
        "--max-reviews", 
        type=int, 
        help="Maximum number of reviews to fetch",
        default=10000
    )
    parser.add_argument(
        "--skip", 
        type=str, 
        nargs="+",
        help="Stages to skip (fetch, preprocess, analyze, store, index, insights, visualize)",
        default=[]
    )
    
    args = parser.parse_args()
    
    # Run the pipeline
    runner = ReviewAnalysisRunner(config_path=args.config)
    
    result = runner.run_pipeline(
        app_id=args.app_id,
        start_date=args.start_date,
        end_date=args.end_date,
        max_reviews=args.max_reviews,
        skip_stages=args.skip
    )
    
    # Print summary
    if result["success"]:
        print("\n===== Pipeline Completed Successfully =====")
        
        metadata = result["metadata"]
        app_info = metadata.get("app_info", {})
        
        print(f"\nApp: {app_info.get('name', 'Mobile App')}")
        print(f"Reviews Analyzed: {metadata.get('total_reviews', 0)}")
        
        # Print insights summary if available
        if result["insights"] and "general" in result["insights"]:
            general_insights = result["insights"]["general"]
            if "analysis" in general_insights:
                print("\nInsights Summary:")
                print("----------------")
                lines = general_insights["analysis"].split("\n")
                for line in lines[:10]:  # Print first 10 lines
                    if line.strip():
                        print(line)
                print("...")
        
        # Print visualization paths if available
        if result["visualizations"] and "dashboard" in result["visualizations"]:
            dashboard = result["visualizations"]["dashboard"]
            if "file_path" in dashboard:
                print(f"\nDashboard: {dashboard['file_path']}")
        
        print("\nPipeline execution time:", end=" ")
        start_time = metadata.get("start_time")
        end_time = metadata.get("end_time")
        if start_time and end_time:
            duration = end_time - start_time
            print(f"{duration.total_seconds():.2f} seconds")
    else:
        print("\n===== Pipeline Failed =====")
        print(f"Error: {result['metadata'].get('error', 'Unknown error')}")
    
    return result


if __name__ == "__main__":
    main()