"""
Vector database module using FAISS for storing and searching embeddings.
"""
import os
import pickle
import numpy as np
import faiss
from typing import List, Dict, Any, Optional
import logging
from pathlib import Path

from backend.config import EMBEDDING_DIMENSION, VECTOR_STORE_DIR, SIMILARITY_THRESHOLD

logger = logging.getLogger(__name__)


class VectorDatabase:
    """FAISS-based vector database for storing and searching embeddings."""
    
    def __init__(self, dimension: int = EMBEDDING_DIMENSION, persist_path: Optional[str] = None):
        """
        Initialize the vector database.
        
        Args:
            dimension: Dimension of embeddings (default: 1536)
            persist_path: Path to persist the database (default: VECTOR_STORE_DIR)
        """
        self.dimension = dimension
        self.persist_path = persist_path or VECTOR_STORE_DIR
        os.makedirs(self.persist_path, exist_ok=True)
        
        # Initialize FAISS index
        self.index = faiss.IndexFlatL2(dimension)
        
        # Store metadata for each vector
        self.metadatas: List[Dict[str, Any]] = []
        
        # Track document sources
        self.documents: Dict[str, List[int]] = {}  # doc_id -> list of vector indices
        
        # Cache for search results
        self._search_cache: Dict[str, List[Dict[str, Any]]] = {}
        self._cache_size_limit = 100
        
        logger.info(f"Initialized vector database with dimension {dimension}")
    
    def add_embeddings(self, embeddings: np.ndarray, metadatas: List[Dict[str, Any]]) -> None:
        """
        Add embeddings to the database.
        
        Args:
            embeddings: Numpy array of shape (n, dimension)
            metadatas: List of metadata dictionaries for each embedding
        """
        try:
            if embeddings.shape[0] != len(metadatas):
                raise ValueError(f"Mismatch: {embeddings.shape[0]} embeddings but {len(metadatas)} metadatas")
            
            if embeddings.shape[1] != self.dimension:
                raise ValueError(f"Embedding dimension {embeddings.shape[1]} != {self.dimension}")
            
            # Normalize embeddings for cosine similarity (L2 normalization)
            faiss.normalize_L2(embeddings)
            
            # Get starting index
            start_idx = self.index.ntotal
            
            # Add to FAISS index
            self.index.add(embeddings.astype('float32'))
            
            # Store metadata
            self.metadatas.extend(metadatas)
            
            # Track document sources
            for i, metadata in enumerate(metadatas):
                doc_id = metadata.get('doc_id', 'unknown')
                if doc_id not in self.documents:
                    self.documents[doc_id] = []
                self.documents[doc_id].append(start_idx + i)
            
            logger.info(f"Added {len(metadatas)} embeddings to vector database")
        
        except Exception as e:
            logger.error(f"Error adding embeddings: {e}")
            raise
    
    def search(
        self,
        query_embedding: np.ndarray,
        k: int = 5,
        filter_dict: Optional[Dict[str, Any]] = None
    ) -> List[Dict[str, Any]]:
        """
        Search for similar embeddings.
        
        Args:
            query_embedding: Query embedding vector (1, dimension)
            k: Number of results to return
            filter_dict: Optional metadata filters
            
        Returns:
            List of dictionaries with 'content', 'metadata', 'score', and 'index'
        """
        try:
            if self.index is None:
                raise ValueError("Vector database not loaded. Call load_from_disk() first.")
            
            # Create cache key
            cache_key = f"{hash(query_embedding.tobytes())}_{k}_{hash(str(filter_dict))}"
            
            # Check cache first
            if cache_key in self._search_cache:
                return self._search_cache[cache_key]
            
            if self.index.ntotal == 0:
                logger.warning("Vector database is empty")
                return []
            
            # Normalize query embedding
            query_embedding = query_embedding.reshape(1, -1).astype('float32')
            faiss.normalize_L2(query_embedding)
            
            # Search
            k = min(k, self.index.ntotal)
            distances, indices = self.index.search(query_embedding, k)
            
            # Prepare results
            results = []
            for i, (distance, idx) in enumerate(zip(distances[0], indices[0])):
                if idx == -1:  # FAISS returns -1 for invalid indices
                    continue
                
                # Convert L2 distance to similarity score (1 / (1 + distance))
                similarity_score = 1.0 / (1.0 + distance)
                
                # Apply similarity threshold
                if similarity_score < SIMILARITY_THRESHOLD:
                    continue
                
                # Get metadata
                metadata = self.metadatas[idx].copy()
                
                # Skip deleted items
                if metadata.get('_deleted', False):
                    continue
                
                # Apply filters if provided
                if filter_dict:
                    match = True
                    for key, value in filter_dict.items():
                        if metadata.get(key) != value:
                            match = False
                            break
                    if not match:
                        continue
                
                result = {
                    'content': metadata.get('content', ''),
                    'metadata': metadata,
                    'score': float(similarity_score),
                    'distance': float(distance),
                    'index': int(idx)
                }
                results.append(result)
            
            # Sort by score (descending)
            results.sort(key=lambda x: x['score'], reverse=True)
            
            # Cache results (with size limit)
            if len(self._search_cache) >= self._cache_size_limit:
                # Remove oldest entry (simple FIFO)
                oldest_key = next(iter(self._search_cache))
                del self._search_cache[oldest_key]
            
            self._search_cache[cache_key] = results
            
            return results
        
        except Exception as e:
            logger.error(f"Error searching vector database: {e}")
            raise
    
    def delete_by_source(self, source_doc: str) -> int:
        """
        Delete all vectors associated with a source document.
        
        Args:
            source_doc: Source document identifier (doc_id or filename)
            
        Returns:
            Number of vectors deleted
        """
        try:
            # Find all indices where doc_id matches the source_doc
            indices_to_remove = set()
            
            for idx, metadata in enumerate(self.metadatas):
                # Match by doc_id (the document identifier passed in)
                if metadata.get('doc_id') == source_doc:
                    indices_to_remove.add(idx)
            
            num_removed = len(indices_to_remove)
            
            if num_removed == 0:
                logger.warning(f"No vectors found for document {source_doc}")
                return 0
            
            # Rebuild index without deleted vectors
            kept_embeddings = []
            kept_metadatas = []
            old_to_new_idx = {}  # Map old indices to new indices
            
            # Extract embeddings from FAISS index safely
            try:
                # Get embeddings using the index's internal storage
                if self.index.ntotal > 0:
                    # Use reconstruct to safely extract embeddings one by one
                    for old_idx in range(self.index.ntotal):
                        if old_idx not in indices_to_remove:
                            # Reconstruct embedding at this index
                            embedding = self.index.reconstruct(int(old_idx))
                            kept_embeddings.append(embedding)
                            kept_metadatas.append(self.metadatas[old_idx])
                            old_to_new_idx[old_idx] = len(kept_embeddings) - 1
            except Exception as e:
                logger.warning(f"Failed to reconstruct embeddings safely, trying alternative method: {e}")
                # Fallback: rebuild from metadata if reconstruction fails
                for old_idx in range(len(self.metadatas)):
                    if old_idx not in indices_to_remove:
                        kept_metadatas.append(self.metadatas[old_idx])
                        old_to_new_idx[old_idx] = len(kept_metadatas) - 1
            
            # Rebuild FAISS index with kept embeddings
            if len(kept_embeddings) > 0:
                kept_embeddings_array = np.array(kept_embeddings, dtype='float32')
                # Normalize for cosine similarity
                faiss.normalize_L2(kept_embeddings_array)
                self.index = faiss.IndexFlatL2(self.dimension)
                self.index.add(kept_embeddings_array)
                self.metadatas = kept_metadatas
            else:
                # All vectors deleted, reset index
                self.index = faiss.IndexFlatL2(self.dimension)
                self.metadatas = []
            
            # Update documents tracking with remapped indices
            new_documents = {}
            for doc_id, indices in self.documents.items():
                if doc_id == source_doc:
                    continue  # Skip the deleted document
                
                # Remap indices for remaining documents
                remapped_indices = []
                for old_idx in indices:
                    if old_idx in old_to_new_idx:
                        remapped_indices.append(old_to_new_idx[old_idx])
                
                if remapped_indices:
                    new_documents[doc_id] = remapped_indices
            
            self.documents = new_documents
            
            # Clear cache as indices changed
            self._search_cache.clear()
            
            logger.info(f"Deleted {num_removed} vectors for document {source_doc}. Index now contains {self.index.ntotal} vectors.")
            return num_removed
        
        except Exception as e:
            logger.error(f"Error deleting document {source_doc}: {e}")
            raise
    
    def clear_all(self) -> None:
        """
        Clear all vectors from the database.
        
        This completely resets the vector database.
        """
        try:
            self.index = faiss.IndexFlatL2(self.dimension)
            self.metadatas = []
            self.documents = {}
            self._search_cache.clear()
            
            logger.info("Vector database cleared completely")
        
        except Exception as e:
            logger.error(f"Error clearing vector database: {e}")
            raise
    
    def save_to_disk(self) -> None:
        """Save the vector database to disk."""
        try:
            index_path = os.path.join(self.persist_path, "faiss.index")
            metadata_path = os.path.join(self.persist_path, "metadata.pkl")
            documents_path = os.path.join(self.persist_path, "documents.pkl")
            
            # Save FAISS index
            faiss.write_index(self.index, index_path)
            
            # Save metadata
            with open(metadata_path, 'wb') as f:
                pickle.dump(self.metadatas, f)
            
            # Save documents mapping
            with open(documents_path, 'wb') as f:
                pickle.dump(self.documents, f)
            
            logger.info(f"Saved vector database to {self.persist_path}")
        
        except Exception as e:
            logger.error(f"Error saving vector database: {e}")
            raise
    
    def load_from_disk(self) -> bool:
        """
        Load the vector database from disk.
        
        Returns:
            True if loaded successfully, False otherwise
        """
        try:
            index_path = os.path.join(self.persist_path, "faiss.index")
            metadata_path = os.path.join(self.persist_path, "metadata.pkl")
            documents_path = os.path.join(self.persist_path, "documents.pkl")
            
            if not all(os.path.exists(p) for p in [index_path, metadata_path, documents_path]):
                logger.warning("Vector database files not found, starting fresh")
                return False
            
            # Load FAISS index
            self.index = faiss.read_index(index_path)
            
            # Load metadata
            with open(metadata_path, 'rb') as f:
                self.metadatas = pickle.load(f)
            
            # Load documents mapping
            with open(documents_path, 'rb') as f:
                self.documents = pickle.load(f)
            
            logger.info(f"Loaded vector database from {self.persist_path} ({self.index.ntotal} vectors)")
            return True
        
        except Exception as e:
            logger.error(f"Error loading vector database: {e}")
            return False
    
    def get_stats(self) -> Dict[str, Any]:
        """
        Get statistics about the vector database.
        
        Returns:
            Dictionary with statistics
        """
        return {
            'total_vectors': self.index.ntotal,
            'dimension': self.dimension,
            'num_documents': len(self.documents),
            'documents': list(self.documents.keys())
        }

