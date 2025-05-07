"""
Implementation of review analytics module.
"""
import warnings
from collections import Counter
from typing import Any, Dict, List, Optional, Tuple, Union

import nltk
import numpy as np
import pandas as pd
from nltk.sentiment import SentimentIntensityAnalyzer
from sklearn.cluster import KMeans
from sklearn.decomposition import LatentDirichletAllocation, NMF
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer

from src.config import config
from src.modules.analytics.interface import AnalyticsInterface

# Suppress sklearn warnings
warnings.filterwarnings("ignore", category=UserWarning)


class ReviewAnalyzer(AnalyticsInterface):
    """
    Implementation of analytics module using scikit-learn and NLTK.
    """
    
    def __init__(self, config_override: Optional[Dict[str, Any]] = None):
        """
        Initialize the review analyzer.
        
        Args:
            config_override: Override for default configuration
        """
        # Initialize attributes with defaults
        self._sentiment_analyzer = "nltk_vader"
        self._topic_method = "lda"
        self._num_topics = 10
        self._vectorizer = None
        self._sentiment_model = None
        self._topic_model = None
        
        # Call parent constructor
        super().__init__(config_override)
        
        # Set attributes from config after validation
        self._sentiment_analyzer = self.config.get("sentiment_analyzer", "nltk_vader")
        self._topic_method = self.config.get("topic_modeling", {}).get("method", "lda")
        self._num_topics = self.config.get("topic_modeling", {}).get("num_topics", 10)
    
    def _validate_config(self) -> None:
        """
        Validate analytics configuration.
        
        Raises:
            ValueError: If required configuration is missing or invalid
        """
        # Validate sentiment analyzer
        valid_sentiment_analyzers = ["nltk_vader", "textblob"]
        if self._sentiment_analyzer not in valid_sentiment_analyzers:
            raise ValueError(f"Invalid sentiment analyzer: {self._sentiment_analyzer}")
        
        # Validate topic modeling method
        valid_topic_methods = ["lda", "nmf"]
        if self._topic_method not in valid_topic_methods:
            raise ValueError(f"Invalid topic modeling method: {self._topic_method}")
        
        # Validate num_topics
        if self._num_topics <= 0:
            raise ValueError(f"Number of topics must be positive, got {self._num_topics}")
    
    def initialize(self) -> None:
        """
        Initialize the analytics module.
        
        Downloads required resources and initializes models.
        
        Raises:
            RuntimeError: If initialization fails
        """
        try:
            # Download required NLTK resources
            if self._sentiment_analyzer == "nltk_vader":
                nltk.download("vader_lexicon", quiet=True)
                self._sentiment_model = SentimentIntensityAnalyzer()
            
            # Initialize TF-IDF vectorizer for topic modeling and clustering
            self._vectorizer = TfidfVectorizer(
                max_features=10000,
                min_df=2,
                max_df=0.85,
                stop_words="english"
            )
            
            self.is_initialized = True
        except Exception as e:
            raise RuntimeError(f"Failed to initialize review analyzer: {e}")
    
    def analyze_sentiment(
        self, 
        data: pd.DataFrame, 
        text_column: str = "text",
        **kwargs
    ) -> pd.DataFrame:
        """
        Perform sentiment analysis on review text.
        
        Args:
            data: DataFrame containing reviews
            text_column: Column containing text to analyze
            
        Returns:
            DataFrame with added sentiment analysis columns
        """
        if text_column not in data.columns:
            raise ValueError(f"Text column '{text_column}' not found in data")
        
        # Create a copy to avoid modifying the original
        result = data.copy()
        
        # Use cleaned or normalized text if available
        if "cleaned_text" in result.columns:
            text_column = "cleaned_text"
        elif "normalized_text" in result.columns:
            text_column = "normalized_text"
        
        if self._sentiment_analyzer == "nltk_vader":
            # Apply VADER sentiment analyzer
            result["sentiment_scores"] = result[text_column].apply(
                lambda x: self._sentiment_model.polarity_scores(x) if isinstance(x, str) else {}
            )
            
            # Extract individual sentiment components
            result["sentiment_negative"] = result["sentiment_scores"].apply(
                lambda x: x.get("neg", 0)
            )
            result["sentiment_neutral"] = result["sentiment_scores"].apply(
                lambda x: x.get("neu", 0)
            )
            result["sentiment_positive"] = result["sentiment_scores"].apply(
                lambda x: x.get("pos", 0)
            )
            result["sentiment_compound"] = result["sentiment_scores"].apply(
                lambda x: x.get("compound", 0)
            )
            
            # Classify sentiment based on compound score
            result["sentiment"] = result["sentiment_compound"].apply(
                lambda x: "positive" if x >= 0.05 else ("negative" if x <= -0.05 else "neutral")
            )
            
            # Clean up intermediate column
            result = result.drop(columns=["sentiment_scores"])
        
        return result
    
    def extract_topics(
        self, 
        data: pd.DataFrame, 
        text_column: str = "text",
        n_topics: Optional[int] = None,
        **kwargs
    ) -> Tuple[pd.DataFrame, Dict[int, List[str]]]:
        """
        Extract topics from review text.
        
        Args:
            data: DataFrame containing reviews
            text_column: Column containing text to analyze
            n_topics: Number of topics to extract (overrides config)
            
        Returns:
            Tuple containing:
              - DataFrame with topic assignments
              - Dictionary mapping topic IDs to representative words
        """
        if text_column not in data.columns:
            raise ValueError(f"Text column '{text_column}' not found in data")
        
        # Use cleaned or normalized text if available
        if "normalized_text" in data.columns:
            text_column = "normalized_text"
        elif "cleaned_text" in data.columns:
            text_column = "cleaned_text"
        
        # Use provided n_topics or fallback to config
        n_topics = n_topics or self._num_topics
        
        # Create a copy to avoid modifying the original
        result = data.copy()
        
        # Filter out empty texts
        valid_mask = result[text_column].astype(bool)
        valid_texts = result.loc[valid_mask, text_column]
        
        if len(valid_texts) < n_topics:
            n_topics = max(2, len(valid_texts) // 2)
        
        # Create and fit vectorizer
        vectorizer = CountVectorizer(
            max_features=5000,
            min_df=2,
            max_df=0.85,
            stop_words="english"
        )
        
        try:
            dtm = vectorizer.fit_transform(valid_texts)
            feature_names = vectorizer.get_feature_names_out()
            
            # Initialize and fit topic model
            if self._topic_method == "lda":
                topic_model = LatentDirichletAllocation(
                    n_components=n_topics,
                    max_iter=10,
                    learning_method="online",
                    random_state=42
                )
            else:  # nmf
                topic_model = NMF(
                    n_components=n_topics,
                    random_state=42,
                    max_iter=1000
                )
            
            # Fit the model
            topic_model.fit(dtm)
            
            # Get topic-word distributions
            topic_words = {}
            for topic_idx, topic in enumerate(topic_model.components_):
                top_words_idx = topic.argsort()[:-11:-1]  # Get top 10 words
                top_words = [feature_names[i] for i in top_words_idx]
                topic_words[topic_idx] = top_words
            
            # Transform documents to get topic distributions
            doc_topic_dist = topic_model.transform(dtm)
            
            # Assign primary topic to each document
            result.loc[valid_mask, "primary_topic"] = np.argmax(doc_topic_dist, axis=1)
            result.loc[~valid_mask, "primary_topic"] = np.nan
            
            # Add topic confidence (probability of primary topic)
            result.loc[valid_mask, "topic_confidence"] = np.max(doc_topic_dist, axis=1)
            result.loc[~valid_mask, "topic_confidence"] = np.nan
            
            # Map topic ID to words for the return value
            return result, topic_words
        
        except Exception as e:
            print(f"Error in topic extraction: {e}")
            # Return original data with empty topic columns if error occurs
            result["primary_topic"] = np.nan
            result["topic_confidence"] = np.nan
            return result, {}
    
    def cluster_reviews(
        self, 
        data: pd.DataFrame, 
        text_column: str = "text",
        n_clusters: int = 5,
        **kwargs
    ) -> pd.DataFrame:
        """
        Cluster reviews based on content similarity.
        
        Args:
            data: DataFrame containing reviews
            text_column: Column containing text to analyze
            n_clusters: Number of clusters to create
            
        Returns:
            DataFrame with cluster assignments
        """
        if text_column not in data.columns:
            raise ValueError(f"Text column '{text_column}' not found in data")
        
        # Use normalized text if available
        if "normalized_text" in data.columns:
            text_column = "normalized_text"
        elif "cleaned_text" in data.columns:
            text_column = "cleaned_text"
        
        # Create a copy to avoid modifying the original
        result = data.copy()
        
        # Filter out empty texts
        valid_mask = result[text_column].astype(bool)
        valid_texts = result.loc[valid_mask, text_column]
        
        if len(valid_texts) < n_clusters:
            n_clusters = max(2, len(valid_texts) // 2)
        
        try:
            # Create and fit vectorizer
            tfidf = self._vectorizer.fit_transform(valid_texts)
            
            # Initialize and fit KMeans
            kmeans = KMeans(
                n_clusters=n_clusters,
                random_state=42,
                n_init=10
            )
            
            # Assign clusters
            clusters = kmeans.fit_predict(tfidf)
            
            # Add cluster assignments to result
            result.loc[valid_mask, "cluster"] = clusters
            result.loc[~valid_mask, "cluster"] = np.nan
            
            # Get cluster centers for interpretation
            if kwargs.get("include_centers", False):
                centers = kmeans.cluster_centers_
                order_centroids = centers.argsort()[:, ::-1]
                terms = self._vectorizer.get_feature_names_out()
                
                cluster_keywords = {}
                for i in range(n_clusters):
                    top_terms_idx = order_centroids[i, :10]  # Top 10 terms
                    cluster_keywords[i] = [terms[idx] for idx in top_terms_idx]
                
                # Add cluster keywords to result
                result["cluster_keywords"] = result["cluster"].map(
                    lambda x: cluster_keywords.get(x, []) if not pd.isna(x) else []
                )
            
            return result
        
        except Exception as e:
            print(f"Error in clustering: {e}")
            # Return original data with empty cluster column if error occurs
            result["cluster"] = np.nan
            return result
    
    def identify_trends(
        self, 
        data: pd.DataFrame, 
        date_column: str = "date",
        value_column: str = "rating", 
        freq: str = "M",
        **kwargs
    ) -> pd.DataFrame:
        """
        Identify trends in review data over time.
        
        Args:
            data: DataFrame containing reviews
            date_column: Column containing dates
            value_column: Column containing values to track
            freq: Frequency for resampling (D=daily, W=weekly, M=monthly)
            
        Returns:
            DataFrame with trend analysis
        """
        if date_column not in data.columns:
            raise ValueError(f"Date column '{date_column}' not found in data")
        
        if value_column not in data.columns:
            raise ValueError(f"Value column '{value_column}' not found in data")
        
        # Create a copy to avoid modifying the original
        df = data.copy()
        
        # Ensure date column is datetime type
        if not pd.api.types.is_datetime64_any_dtype(df[date_column]):
            df[date_column] = pd.to_datetime(df[date_column], errors="coerce")
        
        # Drop rows with invalid dates
        df = df.dropna(subset=[date_column])
        
        # Set date as index for resampling
        df = df.set_index(date_column)
        
        # Create trend analysis based on frequency
        try:
            # For older versions of pandas, we need to use 'M' instead of 'ME'
            # Let's just override any 'M'-containing frequency with 'M' for compatibility
            if isinstance(freq, str) and 'M' in freq:
                resampled_freq = 'M'
            else:
                resampled_freq = freq
            print(f"Using resampled frequency: {resampled_freq}")
            
            # Average value by time period
            mean_by_period = df[value_column].resample(resampled_freq).mean().reset_index()
            mean_by_period.columns = [date_column, f"avg_{value_column}"]
            
            # Count by time period - use same resampled frequency variable
            count_by_period = df[value_column].resample(resampled_freq).count().reset_index()
            count_by_period.columns = [date_column, "count"]
            
            # Percentage of values by rating (if value_column is rating)
            if value_column == "rating" and kwargs.get("include_rating_breakdown", True):
                rating_counts = {}
                unique_ratings = sorted(df[value_column].unique())
                
                for rating in unique_ratings:
                    # Count occurrences of this rating over time
                    rating_series = (df[value_column] == rating).astype(int)
                    # Use the same resampled frequency variable
                    rating_by_period = rating_series.resample(resampled_freq).sum().reset_index()
                    rating_by_period.columns = [date_column, f"rating_{rating}_count"]
                    rating_counts[rating] = rating_by_period
                
                # Merge all rating breakdowns
                rating_trend = count_by_period
                for rating, rating_df in rating_counts.items():
                    rating_trend = pd.merge(
                        rating_trend, 
                        rating_df, 
                        on=date_column,
                        how="left"
                    )
                    # Calculate percentage
                    rating_trend[f"rating_{rating}_pct"] = (
                        rating_trend[f"rating_{rating}_count"] / rating_trend["count"] * 100
                    )
                
                # Merge average with rating breakdown
                result = pd.merge(mean_by_period, rating_trend, on=date_column, how="left")
            else:
                # Just merge average and count
                result = pd.merge(mean_by_period, count_by_period, on=date_column, how="left")
            
            # Calculate rolling average if requested
            if kwargs.get("rolling_window", 0) > 0:
                window = kwargs["rolling_window"]
                result[f"rolling_avg_{window}"] = result[f"avg_{value_column}"].rolling(
                    window=window, min_periods=1
                ).mean()
            
            return result
        
        except Exception as e:
            print(f"Error in trend analysis: {e}")
            # Return empty DataFrame if error occurs
            return pd.DataFrame(columns=[date_column, f"avg_{value_column}", "count"])
    
    def extract_keywords(
        self, 
        data: pd.DataFrame, 
        text_column: str = "text",
        n_keywords: int = 20,
        **kwargs
    ) -> pd.DataFrame:
        """
        Extract important keywords from review text.
        
        Args:
            data: DataFrame containing reviews
            text_column: Column containing text to analyze
            n_keywords: Number of keywords to extract
            
        Returns:
            DataFrame with keyword information
        """
        if text_column not in data.columns:
            raise ValueError(f"Text column '{text_column}' not found in data")
        
        # Use normalized text if available
        if "normalized_text" in data.columns:
            text_column = "normalized_text"
        elif "cleaned_text" in data.columns:
            text_column = "cleaned_text"
        
        try:
            # Extract keywords using TF-IDF
            tfidf_vectorizer = TfidfVectorizer(
                max_features=10000,
                min_df=2,
                max_df=0.85,
                stop_words="english"
            )
            
            # Fit vectorizer to all text
            tfidf_matrix = tfidf_vectorizer.fit_transform(
                data[text_column].fillna("")
            )
            
            # Get feature names
            feature_names = tfidf_vectorizer.get_feature_names_out()
            
            # Sum TF-IDF scores across all documents for each term
            sums = tfidf_matrix.sum(axis=0)
            
            # Convert to array and get top terms
            term_scores = [(term, sums[0, idx]) for idx, term in enumerate(feature_names)]
            sorted_terms = sorted(term_scores, key=lambda x: x[1], reverse=True)
            
            # Extract top N keywords
            top_keywords = sorted_terms[:n_keywords]
            
            # Create result DataFrame
            keywords_df = pd.DataFrame(top_keywords, columns=["keyword", "score"])
            
            # Add frequency count
            if kwargs.get("include_frequency", True):
                # Tokenize all text
                all_tokens = []
                for text in data[text_column].fillna(""):
                    if isinstance(text, str):
                        all_tokens.extend(text.split())
                
                # Count frequencies
                token_counts = Counter(all_tokens)
                
                # Add to result
                keywords_df["frequency"] = keywords_df["keyword"].map(
                    lambda x: token_counts.get(x, 0)
                )
                
                # Calculate percentage of documents containing each keyword
                if kwargs.get("include_doc_pct", True):
                    doc_counts = {}
                    for keyword in keywords_df["keyword"]:
                        # Count documents containing this keyword
                        count = data[text_column].str.contains(
                            fr"\b{keyword}\b", regex=True, case=False, na=False
                        ).sum()
                        doc_counts[keyword] = count
                    
                    keywords_df["doc_count"] = keywords_df["keyword"].map(doc_counts)
                    keywords_df["doc_pct"] = keywords_df["doc_count"] / len(data) * 100
            
            return keywords_df
        
        except Exception as e:
            print(f"Error in keyword extraction: {e}")
            # Return empty DataFrame if error occurs
            return pd.DataFrame(columns=["keyword", "score"])
    
    def get_data(self, **kwargs) -> Dict[str, Any]:
        """
        Implementation of DataProvider interface method.
        
        Not applicable for this module, as it requires input data.
        
        Raises:
            NotImplementedError: This method is not supported
        """
        raise NotImplementedError("AnalyticsModule does not support get_data() without input")