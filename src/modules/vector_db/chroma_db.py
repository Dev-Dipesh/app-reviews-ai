"""
Implementation of vector database using ChromaDB.
"""
import json
import os
from typing import Any, Dict, List, Optional, Union

import chromadb
import numpy as np
import pandas as pd
from chromadb.utils import embedding_functions

from src.config import config
from src.modules.vector_db.interface import VectorDBInterface


class ChromaDBVectorStore(VectorDBInterface):
    """
    Implementation of vector database interface using ChromaDB.
    """
    
    def __init__(self, config_override: Optional[Dict[str, Any]] = None):
        """
        Initialize ChromaDB vector store.
        
        Args:
            config_override: Override for default configuration
        """
        # Initialize attributes with defaults
        self._collection_name = "app_reviews"
        self._embedding_model = "openai"
        self._persist_directory = "data/vector_db"
        self._client = None
        self._collection = None
        self._embedding_function = None
        
        # Call parent constructor
        super().__init__(config_override)
        
        # Set attributes from config after validation
        self._collection_name = self.config.get("collection_name", "app_reviews")
        self._embedding_model = self.config.get("embedding_model", "openai")
        self._persist_directory = self.config.get("persist_directory", "data/vector_db")
    
    def _validate_config(self) -> None:
        """
        Validate ChromaDB configuration.
        
        Raises:
            ValueError: If required configuration is missing
        """
        if not self._collection_name:
            try:
                self._collection_name = config.get("vector_db", "collection_name")
            except KeyError:
                self._collection_name = "app_reviews"  # Default collection name
        
        if not self._embedding_model:
            try:
                self._embedding_model = config.get("vector_db", "embedding_model")
            except KeyError:
                self._embedding_model = "openai"  # Default embedding model
        
        if not self._persist_directory:
            try:
                self._persist_directory = config.get("vector_db", "persist_directory")
            except KeyError:
                self._persist_directory = "data/vector_db"  # Default directory
        
        # Create persist directory if it doesn't exist
        os.makedirs(self._persist_directory, exist_ok=True)
    
    def initialize(self) -> None:
        """
        Initialize vector database.
        
        Creates ChromaDB client, sets up embedding function, and ensures collection exists.
        
        Raises:
            RuntimeError: If initialization fails
        """
        try:
            # Create ChromaDB client
            self._client = chromadb.PersistentClient(path=self._persist_directory)
            
            # Check if we should use mock mode (local embeddings) based on environment variable
            # This also gets checked when handling fallbacks from API errors
            # Always read directly from environment to ensure we get the latest value
            use_mock_data = os.environ.get("USE_MOCK_DATA", "").lower() in ("true", "1", "yes", "y")
            print(f"DEBUG: Vector DB checking USE_MOCK_DATA: '{os.environ.get('USE_MOCK_DATA', '')}'")
            # Force mock mode for debugging
            if "MAX_REVIEWS" in os.environ and int(os.environ.get("MAX_REVIEWS", "0")) <= 50:
                # For small test runs (≤ 50 reviews), always use mock embedding to avoid API costs
                print("DEBUG: Small test run detected (MAX_REVIEWS ≤ 50), using mock embeddings")
                use_mock_data = True
            
            # For mock data, always use the simple mock embedding function
            # Skip trying to use SentenceTransformer which is causing issues
            if use_mock_data:
                # Create a very simple embedding function without any complex dependencies
                print("Using simple mock embedding function")
                
                # Define a custom EmbeddingFunction that conforms to ChromaDB's interface
                class VerySimpleMockEmbedding(embedding_functions.EmbeddingFunction):
                    def __call__(self, input):
                        """
                        Create very simple embeddings with pure Python
                        This avoids any dependencies on external libraries
                        """
                        import random
                        random.seed(42)  # Fixed seed for deterministic results
                        
                        # Create simple 5-dimensional embeddings 
                        embeddings = []
                        for text in input:
                            # Use a hash of the text for more stable embeddings
                            text_hash = sum(ord(c) for c in text[:100]) % 1000
                            
                            # Create a very simple embedding vector
                            embedding = [0.0] * 5
                            
                            # First value based on text length normalized
                            embedding[0] = min(1.0, len(text) / 1000.0)
                            
                            # Second value based on hash
                            embedding[1] = text_hash / 1000.0
                            
                            # Other dimensions random but deterministic
                            random.seed(text_hash)  # Text-specific seed
                            for i in range(2, 5):
                                embedding[i] = random.random()
                                
                            embeddings.append(embedding)
                        return embeddings
                
                # Use the very simple embedding function
                self._embedding_function = VerySimpleMockEmbedding()
                
            # For real data, try using OpenAI embeddings
            elif self._embedding_model == "openai":
                # Setup OpenAI embedding function for real data mode
                print("Using OpenAI embedding function for real data mode")
                try:
                    api_key = os.environ.get("OPENAI_API_KEY")
                    if not api_key:
                        api_key = config.get("llm", {}).get("api_key")
                    
                    self._embedding_function = embedding_functions.OpenAIEmbeddingFunction(
                        api_key=api_key,
                        model_name="text-embedding-ada-002"  # Using older model name for compatibility
                    )
                except Exception as e:
                    raise RuntimeError(f"Failed to initialize OpenAI embedding function: {e}")
            # For explicitly requested sentence transformer, try it
            elif self._embedding_model == "sentence_transformer":
                embedding_model = "all-MiniLM-L6-v2"
                print(f"Using specified local embedding model: {embedding_model}")
                
                try:
                    # Import within the try block to catch import errors
                    from sentence_transformers import SentenceTransformer
                    
                    # Create a custom embedding function that matches ChromaDB's expected interface
                    class SentenceTransformerEmbedding:
                        def __init__(self, model_name):
                            self.model = SentenceTransformer(model_name)
                            
                        def __call__(self, input):
                            # ChromaDB expects the parameter to be named 'input' instead of 'texts'
                            return self.model.encode(input).tolist()
                    
                    # Initialize with our custom class
                    self._embedding_function = SentenceTransformerEmbedding(model_name=embedding_model)
                    
                except Exception as e:
                    print(f"Warning: Failed to initialize SentenceTransformer: {e}")
                    print("Falling back to default ChromaDB embedding function")
                    raise RuntimeError(f"Could not initialize SentenceTransformer: {e}")
            
            # Fall back to default ChromaDB embedding if no other option works
            if self._embedding_function is None:
                print("No embedding function specified or initialized. Using default.")
                # Default will be used by ChromaDB if we don't specify one
            
            # Check if we should clear the collection before adding new documents
            clear_collection = os.environ.get("CLEAR_COLLECTION", "").lower() in ("true", "1", "yes", "y")
            
            # Create a unique collection name for mock data to avoid polluting real data collection
            if use_mock_data:
                # Add "_mock" suffix to the collection name
                if not self._collection_name.endswith("_mock"):
                    self._collection_name = f"{self._collection_name}_mock"
                print(f"Using separate collection for mock data: {self._collection_name}")
            
            # Get or create collection
            try:
                # Delete collection if clear_collection is True
                if clear_collection:
                    try:
                        print(f"Clearing existing collection: {self._collection_name}")
                        self._client.delete_collection(self._collection_name)
                    except Exception as e:
                        print(f"Warning: Could not delete collection: {e}")
                
                # Try to get existing collection
                self._collection = self._client.get_collection(
                    name=self._collection_name,
                    embedding_function=self._embedding_function
                )
                print(f"Using existing collection: {self._collection_name} with {self._collection.count()} documents")
            except Exception:
                # Collection doesn't exist, create it
                print(f"Creating new collection: {self._collection_name}")
                self._collection = self._client.create_collection(
                    name=self._collection_name,
                    embedding_function=self._embedding_function
                )
            
            self.is_initialized = True
        except Exception as e:
            raise RuntimeError(f"Failed to initialize ChromaDB vector store: {e}")
    
    def add_documents(
        self,
        documents: Union[List[Dict[str, Any]], pd.DataFrame],
        text_field: str = "text",
        id_field: Optional[str] = "review_id",
        metadata_fields: Optional[List[str]] = None,
        batch_size: int = 100,
        **kwargs
    ) -> bool:
        """
        Add documents to ChromaDB.
        
        Args:
            documents: List of documents or DataFrame to add
            text_field: Field containing text to embed
            id_field: Field containing unique document ID
            metadata_fields: Fields to include as metadata
            batch_size: Number of documents to process at once
            
        Returns:
            True if successful, False otherwise
        """
        if not self.is_initialized:
            raise RuntimeError("ChromaDB vector store not initialized")
        
        # If input is DataFrame, convert to list of dictionaries
        if isinstance(documents, pd.DataFrame):
            if text_field not in documents.columns:
                raise ValueError(f"Text field '{text_field}' not found in data")
            
            # Convert DataFrame to list of dictionaries
            documents = documents.to_dict(orient="records")
        
        # Ensure metadata_fields is a list
        metadata_fields = metadata_fields or []
        
        try:
            # Process documents in batches
            for i in range(0, len(documents), batch_size):
                batch = documents[i:i+batch_size]
                
                ids = []
                texts = []
                metadatas = []
                
                for doc in batch:
                    # Get document text
                    text = doc.get(text_field, "")
                    if not text or not isinstance(text, str):
                        continue  # Skip documents without valid text
                    
                    # Get document ID
                    if id_field and id_field in doc:
                        doc_id = str(doc[id_field])
                    else:
                        # Generate a unique ID
                        doc_id = f"doc_{i}_{len(ids)}"
                    
                    # Create metadata
                    metadata = {}
                    for field in metadata_fields:
                        if field in doc and field != text_field and field != id_field:
                            value = doc[field]
                            # Skip None values
                            if value is None:
                                continue
                                
                            # Convert non-JSON serializable values to strings
                            if isinstance(value, (np.int64, np.float64)):
                                value = value.item()
                            elif isinstance(value, (pd.Timestamp, pd.Timedelta)):
                                value = str(value)
                            # Convert any other non-primitive types to string
                            elif not isinstance(value, (str, int, float, bool)):
                                value = str(value)
                                
                            metadata[field] = value
                    
                    ids.append(doc_id)
                    texts.append(text)
                    metadatas.append(metadata)
                
                # Add batch to collection
                if ids:
                    self._collection.add(
                        ids=ids,
                        documents=texts,
                        metadatas=metadatas
                    )
            
            return True
        except Exception as e:
            print(f"Error adding documents to ChromaDB: {e}")
            return False
    
    def search(
        self,
        query: str,
        n_results: int = 10,
        filter_criteria: Optional[Dict[str, Any]] = None,
        **kwargs
    ) -> List[Dict[str, Any]]:
        """
        Search for documents similar to query.
        
        Args:
            query: Text query to search for
            n_results: Maximum number of results to return
            filter_criteria: Metadata filters to apply
            
        Returns:
            List of matching documents with similarity scores
        """
        if not self.is_initialized:
            raise RuntimeError("ChromaDB vector store not initialized")
        
        try:
            # Convert filter criteria to ChromaDB format if provided
            where = filter_criteria if filter_criteria else None
            
            # Perform search
            results = self._collection.query(
                query_texts=[query],
                n_results=n_results,
                where=where,
                **kwargs
            )
            
            # Format results
            formatted_results = []
            
            # Check if we have any results
            if results and results.get("documents") and results.get("documents")[0]:
                for i, doc in enumerate(results["documents"][0]):
                    result = {
                        "text": doc,
                        "score": results["distances"][0][i] if "distances" in results else None,
                    }
                    
                    # Add metadata if available
                    if "metadatas" in results and results["metadatas"][0]:
                        result.update(results["metadatas"][0][i])
                    
                    # Add document ID if available
                    if "ids" in results:
                        result["id"] = results["ids"][0][i]
                    
                    formatted_results.append(result)
            
            return formatted_results
        except Exception as e:
            print(f"Error searching ChromaDB: {e}")
            return []
    
    def delete(
        self,
        ids: Optional[List[str]] = None,
        filter_criteria: Optional[Dict[str, Any]] = None,
        **kwargs
    ) -> bool:
        """
        Delete documents from ChromaDB.
        
        Args:
            ids: Document IDs to delete
            filter_criteria: Metadata filters for documents to delete
            
        Returns:
            True if successful, False otherwise
        """
        if not self.is_initialized:
            raise RuntimeError("ChromaDB vector store not initialized")
        
        try:
            if ids:
                # Delete by IDs
                self._collection.delete(ids=ids)
            elif filter_criteria:
                # Delete by filter criteria
                self._collection.delete(where=filter_criteria)
            else:
                # No criteria provided, nothing to delete
                return False
            
            return True
        except Exception as e:
            print(f"Error deleting documents from ChromaDB: {e}")
            return False
    
    def get_collection_stats(self) -> Dict[str, Any]:
        """
        Get statistics about the ChromaDB collection.
        
        Returns:
            Dictionary with statistics about the collection
        """
        if not self.is_initialized:
            raise RuntimeError("ChromaDB vector store not initialized")
        
        try:
            # Get count of documents
            count = self._collection.count()
            
            # Get sample metadata to analyze fields
            sample = self._collection.peek(10)
            
            metadata_fields = set()
            if sample.get("metadatas"):
                for metadata in sample["metadatas"]:
                    metadata_fields.update(metadata.keys())
            
            return {
                "collection_name": self._collection_name,
                "document_count": count,
                "metadata_fields": list(metadata_fields),
                "embedding_model": self._embedding_model,
            }
        except Exception as e:
            print(f"Error getting ChromaDB collection stats: {e}")
            return {
                "collection_name": self._collection_name,
                "document_count": 0,
                "metadata_fields": [],
                "embedding_model": self._embedding_model,
                "error": str(e)
            }