"""
ChromaDB vector database operations for media semantic search.
"""

import chromadb
from chromadb.config import Settings as ChromaSettings
from typing import List, Dict, Any, Optional, Tuple
import hashlib
import logging
from pathlib import Path

from app.config import settings
from app.database.schemas import MediaDocument, MEDIA_COLLECTION_NAME

logger = logging.getLogger(__name__)


class VectorDatabase:
    """
    Vector database interface using ChromaDB for semantic search.
    
    Handles storage and retrieval of media file embeddings and metadata
    for efficient similarity search operations.
    """
    
    def __init__(self):
        """Initialize ChromaDB client and collection."""
        try:
            # Ensure database directory exists
            db_path = Path(settings.CHROMA_DB_PATH)
            db_path.mkdir(parents=True, exist_ok=True)
            
            # Initialize ChromaDB client
            self.client = chromadb.PersistentClient(
                path=str(db_path),
                settings=ChromaSettings(
                    anonymized_telemetry=False,
                    allow_reset=True
                )
            )
            
            # Get or create the media collection
            self.collection = self.client.get_or_create_collection(
                name=MEDIA_COLLECTION_NAME,
                metadata={
                    "hnsw:space": "cosine",  # Use cosine similarity
                    "hnsw:construction_ef": 200,
                    "hnsw:M": 16
                }
            )
            
            logger.info(f"Vector database initialized with {self.collection.count()} documents")
            
        except Exception as e:
            logger.error(f"Failed to initialize vector database: {e}")
            raise
    
    def add_media(self, document: MediaDocument) -> str:
        """
        Add a media file to the vector database.
        
        Args:
            document: MediaDocument containing file information and embedding
            
        Returns:
            str: The file ID of the added document
        """
        try:
            if not document.embedding:
                raise ValueError("Document must have an embedding")

            # Ensure collection exists before adding
            if not self.ensure_collection_exists():
                raise RuntimeError("Collection does not exist and could not be created")

            # Generate file ID if not provided
            if not document.file_id:
                document.file_id = self._generate_file_id(document.file_path)
            
            # Prepare data for ChromaDB
            metadata = document.to_chroma_metadata()
            searchable_text = document.to_searchable_text()
            
            # Add to collection
            self.collection.add(
                ids=[document.file_id],
                embeddings=[document.embedding],
                metadatas=[metadata],
                documents=[searchable_text]
            )
            
            logger.debug(f"Added media document: {document.file_id}")
            return document.file_id
            
        except Exception as e:
            logger.error(f"Failed to add media document: {e}")
            raise
    
    def search_similar(
        self,
        query_embedding: List[float],
        limit: int = 20,
        filters: Optional[Dict[str, Any]] = None
    ) -> List[Dict[str, Any]]:
        """
        Search for similar media files using vector similarity.

        Args:
            query_embedding: Vector embedding of the search query
            limit: Maximum number of results to return
            filters: Optional metadata filters

        Returns:
            List of search results with metadata and similarity scores
        """
        try:
            # Ensure collection exists before searching
            if not self.ensure_collection_exists():
                logger.warning("Collection does not exist or is not accessible, returning empty results")
                return []

            # Build where clause for filtering
            where_clause = self._build_where_clause(filters) if filters else None

            # Perform similarity search
            results = self.collection.query(
                query_embeddings=[query_embedding],
                n_results=min(limit, 100),  # ChromaDB limit
                where=where_clause,
                include=["metadatas", "documents", "distances"]
            )

            # Format results
            formatted_results = self._format_search_results(results)

            logger.debug(f"Vector search returned {len(formatted_results)} results")
            return formatted_results

        except Exception as e:
            logger.error(f"Vector search failed: {e}")
            # Try to recover by ensuring collection exists
            try:
                logger.info("Attempting to recover from search failure...")
                if self.ensure_collection_exists():
                    logger.info("Collection recovered, returning empty results for this search")
                    return []
                else:
                    logger.error("Collection recovery failed")
                    raise
            except Exception as recovery_error:
                logger.error(f"Search recovery failed: {recovery_error}")
                raise
    
    def get_media(self, file_id: str) -> Optional[MediaDocument]:
        """
        Get a media document by its file ID.
        
        Args:
            file_id: Unique identifier for the file
            
        Returns:
            MediaDocument if found, None otherwise
        """
        try:
            results = self.collection.get(
                ids=[file_id],
                include=["metadatas", "documents", "embeddings"]
            )
            
            if not results["ids"]:
                return None
            
            # Convert back to MediaDocument
            metadata = results["metadatas"][0]
            embedding = results["embeddings"][0] if results["embeddings"] is not None and len(results["embeddings"]) > 0 else None
            
            document = MediaDocument.from_chroma_metadata(file_id, metadata)
            document.embedding = embedding
            
            return document
            
        except Exception as e:
            logger.error(f"Failed to get media document {file_id}: {e}")
            return None
    
    def update_media(self, document: MediaDocument) -> bool:
        """
        Update an existing media document.
        
        Args:
            document: Updated MediaDocument
            
        Returns:
            bool: True if successful, False otherwise
        """
        try:
            metadata = document.to_chroma_metadata()
            searchable_text = document.to_searchable_text()
            
            update_data = {
                "ids": [document.file_id],
                "metadatas": [metadata],
                "documents": [searchable_text]
            }
            
            # Include embedding if provided
            if document.embedding:
                update_data["embeddings"] = [document.embedding]
            
            self.collection.update(**update_data)
            
            logger.debug(f"Updated media document: {document.file_id}")
            return True
            
        except Exception as e:
            logger.error(f"Failed to update media document {document.file_id}: {e}")
            return False
    
    def delete_media(self, file_id: str) -> bool:
        """
        Delete a media document from the database.
        
        Args:
            file_id: Unique identifier for the file
            
        Returns:
            bool: True if successful, False otherwise
        """
        try:
            # Ensure collection exists before deleting
            if not self.ensure_collection_exists():
                logger.warning(f"Collection does not exist, cannot delete document: {file_id}")
                return False

            self.collection.delete(ids=[file_id])
            logger.debug(f"Deleted media document: {file_id}")
            return True

        except Exception as e:
            logger.error(f"Failed to delete media document {file_id}: {e}")
            return False
    
    def list_media(
        self, 
        limit: int = 100, 
        offset: int = 0,
        filters: Optional[Dict[str, Any]] = None
    ) -> List[MediaDocument]:
        """
        List media documents with optional filtering and pagination.
        
        Args:
            limit: Maximum number of documents to return
            offset: Number of documents to skip
            filters: Optional metadata filters
            
        Returns:
            List of MediaDocument objects
        """
        try:
            # Ensure collection exists before listing
            if not self.ensure_collection_exists():
                logger.warning("Collection does not exist, returning empty list")
                return []

            # Build where clause for filtering
            where_clause = self._build_where_clause(filters) if filters else None

            # Get documents (ChromaDB doesn't support offset directly)
            results = self.collection.get(
                where=where_clause,
                limit=limit + offset,  # Get more than needed
                include=["metadatas"]
            )
            
            # Apply manual offset
            ids = results["ids"][offset:offset + limit]
            metadatas = results["metadatas"][offset:offset + limit]
            
            # Convert to MediaDocument objects
            documents = []
            for file_id, metadata in zip(ids, metadatas):
                document = MediaDocument.from_chroma_metadata(file_id, metadata)
                documents.append(document)
            
            return documents
            
        except Exception as e:
            logger.error(f"Failed to list media documents: {e}")
            return []
    
    def get_stats(self) -> Dict[str, Any]:
        """
        Get database statistics.
        
        Returns:
            Dictionary containing database statistics
        """
        try:
            # Ensure collection exists before getting stats
            if not self.ensure_collection_exists():
                return {
                    "total_documents": 0,
                    "collection_name": MEDIA_COLLECTION_NAME,
                    "database_path": settings.CHROMA_DB_PATH,
                    "error": "Collection does not exist"
                }

            total_count = self.collection.count()

            # Get sample of documents to calculate stats
            sample_results = self.collection.get(
                limit=min(1000, total_count),
                include=["metadatas"]
            )
            
            # Calculate statistics
            stats = {
                "total_documents": total_count,
                "collection_name": MEDIA_COLLECTION_NAME,
                "database_path": settings.CHROMA_DB_PATH
            }
            
            if sample_results["metadatas"]:
                # Count by media type
                media_types = {}
                formats = {}
                total_size = 0
                
                for metadata in sample_results["metadatas"]:
                    # Media type counts
                    media_type = metadata.get("media_type", "unknown")
                    media_types[media_type] = media_types.get(media_type, 0) + 1
                    
                    # Format counts
                    format_type = metadata.get("format", "unknown")
                    formats[format_type] = formats.get(format_type, 0) + 1
                    
                    # Total size
                    file_size = metadata.get("file_size", 0)
                    if isinstance(file_size, (int, float)):
                        total_size += file_size
                
                stats.update({
                    "media_types": media_types,
                    "formats": formats,
                    "estimated_total_size_bytes": int(total_size * (total_count / len(sample_results["metadatas"])))
                })
            
            return stats
            
        except Exception as e:
            logger.error(f"Failed to get database stats: {e}")
            return {"error": str(e)}
    
    def clear_collection(self) -> bool:
        """
        Clear all documents from the collection.

        WARNING: This operation cannot be undone.

        Returns:
            bool: True if successful, False otherwise
        """
        try:
            logger.info("Starting collection clear operation...")

            # First, try to delete the collection if it exists
            try:
                self.client.delete_collection(MEDIA_COLLECTION_NAME)
                logger.info("Existing collection deleted successfully")
            except Exception as delete_error:
                logger.warning(f"Collection deletion failed (may not exist): {delete_error}")
                # Continue anyway - collection might not exist

            # Recreate the collection with the same configuration as __init__
            self.collection = self.client.get_or_create_collection(
                name=MEDIA_COLLECTION_NAME,
                metadata={
                    "hnsw:space": "cosine",  # Use cosine similarity
                    "hnsw:construction_ef": 200,
                    "hnsw:M": 16
                }
            )

            # Verify the collection was created successfully
            count = self.collection.count()
            logger.warning(f"Vector database collection cleared and recreated (count: {count})")
            return True

        except Exception as e:
            logger.error(f"Failed to clear collection: {e}")
            # Try to reinitialize the collection as a fallback
            try:
                logger.info("Attempting to reinitialize collection as fallback...")
                self.collection = self.client.get_or_create_collection(
                    name=MEDIA_COLLECTION_NAME,
                    metadata={
                        "hnsw:space": "cosine",
                        "hnsw:construction_ef": 200,
                        "hnsw:M": 16
                    }
                )
                logger.info("Collection reinitialized successfully")
                return True
            except Exception as fallback_error:
                logger.error(f"Fallback collection initialization failed: {fallback_error}")
                return False

    def ensure_collection_exists(self) -> bool:
        """
        Ensure the collection exists and is accessible.

        This method can be called before operations to verify the collection
        is available and recreate it if necessary.

        Returns:
            bool: True if collection exists and is accessible, False otherwise
        """
        try:
            # Try to access the collection
            if self.collection is None:
                logger.warning("Collection reference is None, attempting to reinitialize...")
                self._reinitialize_collection()
                return self.collection is not None

            # Test collection accessibility
            count = self.collection.count()
            logger.debug(f"Collection exists and accessible (count: {count})")
            return True

        except Exception as e:
            logger.warning(f"Collection not accessible: {e}")
            # Try to reinitialize
            try:
                self._reinitialize_collection()
                return self.collection is not None
            except Exception as reinit_error:
                logger.error(f"Failed to reinitialize collection: {reinit_error}")
                return False

    def _reinitialize_collection(self) -> None:
        """
        Reinitialize the collection reference.

        This is used as a recovery mechanism when the collection becomes
        inaccessible or doesn't exist.
        """
        logger.info("Reinitializing collection reference...")
        self.collection = self.client.get_or_create_collection(
            name=MEDIA_COLLECTION_NAME,
            metadata={
                "hnsw:space": "cosine",  # Use cosine similarity
                "hnsw:construction_ef": 200,
                "hnsw:M": 16
            }
        )
        logger.info("Collection reference reinitialized successfully")
    
    def get_collection(self):
        """Get the ChromaDB collection object."""
        return self.collection
    
    def _generate_file_id(self, file_path: str) -> str:
        """Generate a unique file ID from the file path."""
        return hashlib.md5(file_path.encode()).hexdigest()
    
    def _build_where_clause(self, filters: Dict[str, Any]) -> Dict[str, Any]:
        """
        Build ChromaDB where clause from filters.
        
        Args:
            filters: Dictionary of filter conditions
            
        Returns:
            ChromaDB where clause
        """
        where_clause = {}
        
        for key, value in filters.items():
            if key == "media_type" and value != "all":
                where_clause["media_type"] = {"$eq": value}
            elif key == "format" and value:
                where_clause["format"] = {"$eq": value}
            elif key == "file_size_min" and value:
                where_clause["file_size"] = {"$gte": value}
            elif key == "file_size_max" and value:
                if "file_size" in where_clause:
                    where_clause["file_size"]["$lte"] = value
                else:
                    where_clause["file_size"] = {"$lte": value}
        
        return where_clause
    
    def _format_search_results(self, results: Dict[str, Any]) -> List[Dict[str, Any]]:
        """
        Format ChromaDB search results for API response.
        
        Args:
            results: Raw ChromaDB query results
            
        Returns:
            Formatted list of search results
        """
        formatted = []
        
        if not results["ids"] or not results["ids"][0]:
            return formatted
        
        ids = results["ids"][0]
        metadatas = results["metadatas"][0]
        documents = results["documents"][0]
        distances = results["distances"][0]
        
        for i, file_id in enumerate(ids):
            # Convert distance to similarity score (cosine distance -> similarity)
            similarity_score = 1 - distances[i]
            
            # Create MediaDocument from metadata
            document = MediaDocument.from_chroma_metadata(file_id, metadatas[i])
            
            formatted.append({
                "file_id": file_id,
                "metadata": document,
                "similarity_score": similarity_score,
                "searchable_text": documents[i]
            })
        
        return formatted
