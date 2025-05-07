"""
Interface for vector database modules.
"""
from abc import abstractmethod
from typing import Any, Dict, List, Optional, Union

import pandas as pd

from src.modules.base import DataConsumer, DataProvider


class VectorDBInterface(DataConsumer, DataProvider):
    """
    Interface for vector database operations.
    """
    
    @abstractmethod
    def _validate_config(self) -> None:
        """
        Validate vector database configuration.
        
        Should check for:
        - Database connection details
        - Embedding model configuration
        - Collection/index settings
        
        Raises:
            ValueError: If configuration is invalid
        """
        pass
    
    @abstractmethod
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
        Add documents to the vector database.
        
        Args:
            documents: List of documents or DataFrame to add
            text_field: Field containing text to embed
            id_field: Field containing unique document ID
            metadata_fields: Fields to include as metadata
            batch_size: Number of documents to process at once
            
        Returns:
            True if successful, False otherwise
        """
        pass
    
    @abstractmethod
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
        pass
    
    @abstractmethod
    def delete(
        self,
        ids: Optional[List[str]] = None,
        filter_criteria: Optional[Dict[str, Any]] = None,
        **kwargs
    ) -> bool:
        """
        Delete documents from the vector database.
        
        Args:
            ids: Document IDs to delete
            filter_criteria: Metadata filters for documents to delete
            
        Returns:
            True if successful, False otherwise
        """
        pass
    
    @abstractmethod
    def get_collection_stats(self) -> Dict[str, Any]:
        """
        Get statistics about the vector database collection.
        
        Returns:
            Dictionary with statistics about the collection
        """
        pass
    
    def process_data(
        self, 
        data: Union[List[Dict[str, Any]], pd.DataFrame], 
        **kwargs
    ) -> bool:
        """
        Implementation of DataConsumer interface method.
        
        Args:
            data: Data to add to vector database
            
        Returns:
            True if successful, False otherwise
        """
        return self.add_documents(data, **kwargs)
    
    def get_data(
        self, 
        query: Optional[str] = None,
        n_results: int = 100,
        filter_criteria: Optional[Dict[str, Any]] = None,
        **kwargs
    ) -> List[Dict[str, Any]]:
        """
        Implementation of DataProvider interface method.
        
        Args:
            query: Text query to search for
            n_results: Maximum number of results to return
            filter_criteria: Metadata filters to apply
            
        Returns:
            List of matching documents
        """
        if query is None:
            raise ValueError("Query is required for vector database retrieval")
        
        return self.search(query, n_results, filter_criteria, **kwargs)