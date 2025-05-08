"""
Data loading and preprocessing utilities for app reviews.
Provides standardized functions for loading from various sources and preprocessing.
"""

import os
import sys
import pandas as pd
import numpy as np
import re
from datetime import datetime
from typing import Dict, List, Any, Optional, Union

# Try to import project modules
try:
    from src.modules.preprocessing.nlp_preprocessor import NLPPreprocessor
except ImportError:
    # Fallback if imports fail
    class NLPPreprocessor:
        """Fallback NLP Preprocessor when the real one is not available"""
        def __init__(self, config=None):
            self.is_initialized = False
            self.config = config or {}
            print("Warning: Using fallback NLPPreprocessor")
        
        def initialize(self):
            self.is_initialized = True
            return True
        
        def clean_text(self, text):
            """Basic text cleaning"""
            if not isinstance(text, str):
                text = str(text)
            # Convert to lowercase
            text = text.lower()
            # Remove special characters and punctuation
            text = re.sub(r'[^\w\s]', '', text)
            # Remove numbers
            text = re.sub(r'\d+', '', text)
            # Remove extra spaces
            text = re.sub(r'\s+', ' ', text).strip()
            return text
        
        def normalize_text(self, text):
            """Basic stopword removal"""
            try:
                # Try to download NLTK resources if not available
                import nltk
                nltk.download('stopwords', quiet=True)
                from nltk.corpus import stopwords
                stopwords_list = set(stopwords.words('english'))
            except:
                # If NLTK is not available, use a small set of common stopwords
                stopwords_list = {'a', 'an', 'the', 'and', 'or', 'but', 'is', 'are', 
                                'was', 'were', 'to', 'of', 'in', 'for', 'with'}
            
            return ' '.join([word for word in text.split() if word not in stopwords_list])


def preprocess_reviews(df):
    """
    Preprocess review text data for analysis.
    
    This function standardizes column names, cleans text, and normalizes content
    to prepare for further analysis. It handles both the case where the project's
    NLPPreprocessor is available and a fallback method when it's not.
    
    Args:
        df: A pandas DataFrame containing review data
        
    Returns:
        A pandas DataFrame with standardized columns and processed text
    """
    # Make a copy to avoid modifying the original dataframe
    processed_df = df.copy()
    
    # Check if we need to rename columns to standardized format
    column_mapping = {
        'content': 'text',
        'score': 'rating',
        'at': 'date'
    }
    
    # Apply column mapping where columns exist
    for old_col, new_col in column_mapping.items():
        if old_col in processed_df.columns and new_col not in processed_df.columns:
            processed_df[new_col] = processed_df[old_col]
    
    # Ensure text column exists
    if 'text' not in processed_df.columns:
        print("Warning: No 'text' column found in data")
        if 'content' in processed_df.columns:
            processed_df['text'] = processed_df['content']
        else:
            # Create an empty text column if none exists
            processed_df['text'] = ""
    
    try:
        # Use the project's NLPPreprocessor if available
        preprocessor = NLPPreprocessor({"enable_lemmatization": True})
        
        # Make sure the preprocessor is initialized
        if not preprocessor.is_initialized:
            preprocessor.initialize()
        
        # Apply preprocessing to text column
        print("Cleaning and normalizing review text...")
        processed_df['cleaned_text'] = processed_df['text'].apply(
            lambda x: preprocessor.clean_text(str(x)) if pd.notna(x) else "")
        
        processed_df['normalized_text'] = processed_df['cleaned_text'].apply(
            lambda x: preprocessor.normalize_text(x) if pd.notna(x) else "")
        
        print("Preprocessing complete.")
    except Exception as e:
        print(f"Error during preprocessing: {e}")
        print("Falling back to basic text cleaning...")
        
        # Basic fallback preprocessing if the advanced module fails
        import re
        try:
            # Try to download NLTK resources if not available
            import nltk
            nltk.download('stopwords', quiet=True)
            from nltk.corpus import stopwords
            stopwords_list = set(stopwords.words('english'))
        except:
            # If NLTK is not available, use a small set of common stopwords
            stopwords_list = {'a', 'an', 'the', 'and', 'or', 'but', 'is', 'are', 
                            'was', 'were', 'to', 'of', 'in', 'for', 'with'}
        
        # Simple cleaning function
        def basic_clean(text):
            if not isinstance(text, str):
                text = str(text)
            # Convert to lowercase
            text = text.lower()
            # Remove special characters and punctuation
            text = re.sub(r'[^\w\s]', '', text)
            # Remove numbers
            text = re.sub(r'\d+', '', text)
            # Remove extra spaces
            text = re.sub(r'\s+', ' ', text).strip()
            return text
            
        # Simple stopword removal
        def remove_stopwords(text):
            return ' '.join([word for word in text.split() if word not in stopwords_list])
        
        # Apply basic cleaning
        processed_df['cleaned_text'] = processed_df['text'].apply(
            lambda x: basic_clean(x) if pd.notna(x) else "")
        
        # Apply stopword removal for normalization
        processed_df['normalized_text'] = processed_df['cleaned_text'].apply(
            lambda x: remove_stopwords(x) if pd.notna(x) else "")
        
        print("Basic preprocessing complete.")
    
    return processed_df


def load_reviews(use_mock_data=False, data_dir=None, raw_filename='reviews.csv', processed_filename='processed_reviews.csv'):
    """
    Load review data from CSV or generate mock data.
    
    Args:
        use_mock_data: If True, generate mock review data instead of loading from CSV
        data_dir: Custom data directory path (default: project_root/data)
        raw_filename: Name of the raw reviews CSV file
        processed_filename: Name of the processed reviews CSV file
        
    Returns:
        A pandas DataFrame containing review data (either loaded or generated)
    """
    # Set up project paths
    if data_dir is None:
        # Try to determine project root
        current_dir = os.getcwd()
        if 'notebooks' in current_dir:
            # If running from notebooks directory, go up one level
            project_root = os.path.abspath(os.path.join(current_dir, '..'))
        else:
            # Assume current directory is project root
            project_root = current_dir
        
        data_dir = os.path.join(project_root, 'data')
    
    raw_csv_path = os.path.join(data_dir, raw_filename)
    processed_csv_path = os.path.join(data_dir, processed_filename)
    
    # Create data directory if it doesn't exist
    os.makedirs(data_dir, exist_ok=True)
    
    if not use_mock_data:
        # Try to load preprocessed data first
        if os.path.exists(processed_csv_path):
            try:
                print(f"Loading preprocessed reviews from: {processed_csv_path}")
                df = pd.read_csv(processed_csv_path)
                
                # Convert date to datetime if it exists
                if 'date' in df.columns:
                    df['date'] = pd.to_datetime(df['date'], errors='coerce')
                
                print(f"Successfully loaded {len(df)} preprocessed reviews")
                return df
            except Exception as e:
                print(f"Error loading preprocessed data: {e}")
                print("Trying raw data instead...")
        
        # Try to load raw data if preprocessed data not available
        if os.path.exists(raw_csv_path):
            try:
                print(f"Loading raw reviews from: {raw_csv_path}")
                df = pd.read_csv(raw_csv_path)
                
                # Convert date to datetime if it exists
                if 'date' in df.columns:
                    df['date'] = pd.to_datetime(df['date'], errors='coerce')
                
                print(f"Successfully loaded {len(df)} raw reviews")
                
                # Preprocess the data
                processed_df = preprocess_reviews(df)
                
                # Save processed data for future use
                try:
                    processed_df.to_csv(processed_csv_path, index=False)
                    print(f"Saved preprocessed data to: {processed_csv_path}")
                except Exception as e:
                    print(f"Warning: Could not save preprocessed data: {e}")
                
                return processed_df
            except Exception as e:
                print(f"Error loading raw data: {e}")
                print("Falling back to mock data...")
    
    # Generate mock data if requested or if real data loading failed
    print("Generating mock review data...")
    df = generate_mock_reviews(500)
    print(f"Generated {len(df)} mock reviews")
    
    # Process the mock data
    processed_df = preprocess_reviews(df)
    return processed_df


def generate_mock_reviews(count=500):
    """
    Generate mock review data for testing.
    
    Args:
        count: Number of mock reviews to generate
        
    Returns:
        A pandas DataFrame containing synthetic review data
    """
    # Random seed for reproducibility
    np.random.seed(42)
    
    # Generate random dates between 2 years ago and today
    today = datetime.now()
    two_years_ago = today.replace(year=today.year - 2)
    days_range = (today - two_years_ago).days
    
    dates = [two_years_ago + pd.Timedelta(days=np.random.randint(0, days_range)) for _ in range(count)]
    
    # Generate ratings with a specific distribution (more 5-star and 1-star reviews)
    ratings_dist = [1, 2, 3, 4, 5]
    ratings_weights = [0.2, 0.1, 0.15, 0.15, 0.4]  # Weighted towards 5-star and 1-star
    ratings = np.random.choice(ratings_dist, size=count, p=ratings_weights)
    
    # Sample templates for review text
    positive_templates = [
        "Great app! {feature} works perfectly. I love the {adjective} experience.",
        "This is by far the best {service} app I've used. {feature} is so {adjective}.",
        "I'm impressed with the {adjective} design and {feature}. Will definitely keep using it.",
        "Five stars for the {adjective} {feature}. Makes {service} so easy!",
        "Love how {adjective} this app is. The {feature} is outstanding."
    ]
    
    neutral_templates = [
        "Decent app. {feature} works most of the time. {issue} could use improvement.",
        "It's okay for {service}. The {feature} is good but {issue} is frustrating.",
        "Works as expected. {feature} is useful but {issue} needs fixing.",
        "Not bad, not great. {feature} is nice but {issue} is annoying.",
        "Average app for {service}. {feature} is helpful but {issue} is a problem."
    ]
    
    negative_templates = [
        "Terrible app! {issue} makes it unusable. {feature} doesn't work at all.",
        "Very frustrating experience. {issue} happens constantly. Avoid this app!",
        "Don't waste your time. {issue} ruined my {service} experience completely.",
        "One star because {issue} is so bad. Even the {feature} doesn't work right.",
        "Awful app design. {issue} is a major problem and {feature} is confusing."
    ]
    
    # Content components
    features = [
        "booking system", "check-in process", "payment system", 
        "search functionality", "notification system", "account management",
        "rewards program", "flight status tracker", "seat selection", 
        "baggage tracking", "customer support chat", "route mapping"
    ]
    
    issues = [
        "app crashes", "payment failures", "login problems", 
        "slow performance", "confusing interface", "booking errors",
        "sync issues", "notification glitches", "connection problems", 
        "authentication errors", "lost reservations", "incorrect pricing"
    ]
    
    services = [
        "travel", "airline", "flight booking", "airport navigation", 
        "travel planning", "business travel", "vacation booking"
    ]
    
    adjectives = [
        "intuitive", "seamless", "user-friendly", "efficient", "responsive",
        "fast", "reliable", "sleek", "modern", "convenient", "innovative"
    ]
    
    # Generate versions (more recent for newer reviews)
    versions = []
    for date in dates:
        if date.year == today.year:
            # Recent reviews get newer versions
            version = f"{np.random.randint(4, 6)}.{np.random.randint(0, 10)}.{np.random.randint(0, 10)}"
        else:
            # Older reviews get older versions
            version = f"{np.random.randint(1, 4)}.{np.random.randint(0, 10)}.{np.random.randint(0, 10)}"
        versions.append(version)
    
    # Generate review texts based on rating
    texts = []
    for rating in ratings:
        if rating >= 4:
            template = np.random.choice(positive_templates)
            text = template.format(
                feature=np.random.choice(features),
                adjective=np.random.choice(adjectives),
                service=np.random.choice(services)
            )
        elif rating <= 2:
            template = np.random.choice(negative_templates)
            text = template.format(
                feature=np.random.choice(features),
                issue=np.random.choice(issues),
                service=np.random.choice(services)
            )
        else:
            template = np.random.choice(neutral_templates)
            text = template.format(
                feature=np.random.choice(features),
                issue=np.random.choice(issues),
                service=np.random.choice(services)
            )
        texts.append(text)
    
    # Create DataFrame
    df = pd.DataFrame({
        'text': texts,
        'rating': ratings,
        'date': dates,
        'reviewCreatedVersion': versions
    })
    
    # Sort by date (most recent first)
    df = df.sort_values('date', ascending=False)
    
    return df


def get_data_summary(df):
    """
    Generate summary statistics for the review dataset.
    
    Args:
        df: A pandas DataFrame containing review data
        
    Returns:
        A dictionary with summary statistics
    """
    # Ensure we have standard column names
    std_df = df.copy()
    
    # Standardize column names if needed
    if 'rating' not in std_df.columns and 'score' in std_df.columns:
        std_df['rating'] = std_df['score']
    
    if 'date' not in std_df.columns and 'at' in std_df.columns:
        std_df['date'] = std_df['at']
    
    # Basic statistics
    summary = {
        'total_reviews': len(std_df),
        'rating_distribution': std_df['rating'].value_counts().to_dict() if 'rating' in std_df.columns else {},
        'average_rating': round(std_df['rating'].mean(), 2) if 'rating' in std_df.columns else None,
        'date_range': None
    }
    
    # Date range if available
    if 'date' in std_df.columns:
        # Convert to datetime if not already
        if not pd.api.types.is_datetime64_dtype(std_df['date']):
            std_df['date'] = pd.to_datetime(std_df['date'], errors='coerce')
        
        # Get min and max dates (excluding NaT values)
        valid_dates = std_df['date'].dropna()
        if not valid_dates.empty:
            summary['date_range'] = {
                'start': valid_dates.min().strftime('%Y-%m-%d'),
                'end': valid_dates.max().strftime('%Y-%m-%d'),
                'days': (valid_dates.max() - valid_dates.min()).days
            }
    
    # Version information if available
    if 'reviewCreatedVersion' in std_df.columns:
        summary['versions'] = std_df['reviewCreatedVersion'].value_counts().to_dict()
    
    # Text statistics
    if 'text' in std_df.columns:
        # Calculate text lengths
        std_df['text_length'] = std_df['text'].apply(lambda x: len(str(x)) if pd.notna(x) else 0)
        
        summary['text_stats'] = {
            'avg_length': round(std_df['text_length'].mean(), 1),
            'max_length': std_df['text_length'].max(),
            'empty_reviews': (std_df['text_length'] == 0).sum()
        }
    
    return summary


# Additional utility functions can be added here
def convert_timestamp_format(df, column='date', input_format=None, output_format='%Y-%m-%d'):
    """
    Convert timestamp column to a standardized format.
    
    Args:
        df: DataFrame containing the timestamp column
        column: Name of the column to convert
        input_format: Format string for parsing (if None, try to infer)
        output_format: Format string for output
        
    Returns:
        DataFrame with converted timestamp column
    """
    result_df = df.copy()
    
    if column not in result_df.columns:
        print(f"Warning: Column '{column}' not found in DataFrame")
        return result_df
    
    # Convert to datetime
    if input_format:
        result_df[column] = pd.to_datetime(result_df[column], format=input_format, errors='coerce')
    else:
        result_df[column] = pd.to_datetime(result_df[column], errors='coerce')
    
    # Convert to string in desired format
    if output_format:
        result_df[column] = result_df[column].dt.strftime(output_format)
    
    return result_df