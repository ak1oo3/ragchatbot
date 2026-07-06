"""
Embeddings module for generating vector embeddings using Azure OpenAI.
"""
import numpy as np
from typing import List
import logging
from tenacity import retry, stop_after_attempt, wait_exponential
from openai import AzureOpenAI

from backend.config import (
    AZURE_OPENAI_ENDPOINT,
    AZURE_OPENAI_API_KEY,
    EMBEDDINGS_API_VERSION,
    EMBEDDING_MODEL,
    EMBEDDING_DIMENSION
)

logger = logging.getLogger(__name__)


class EmbeddingGenerator:
    """Generate embeddings using Azure OpenAI."""
    
    def __init__(self):
        """Initialize Azure OpenAI embeddings client."""
        try:
            self.client = AzureOpenAI(
                api_key=AZURE_OPENAI_API_KEY,
                api_version=EMBEDDINGS_API_VERSION,
                azure_endpoint=AZURE_OPENAI_ENDPOINT
            )
            self.model = EMBEDDING_MODEL
            logger.info("Initialized Azure OpenAI embeddings client")
        except Exception as e:
            logger.error(f"Failed to initialize embeddings client: {e}")
            raise
    
    @retry(stop=stop_after_attempt(3), wait=wait_exponential(multiplier=1, min=4, max=10))
    def generate_embedding(self, text: str) -> np.ndarray:
        """
        Generate embedding for a single text string.
        
        Args:
            text: Text to embed
            
        Returns:
            Numpy array of embedding vector (1536 dimensions)
        """
        try:
            if not text or not text.strip():
                # Return zero vector for empty text
                return np.zeros(EMBEDDING_DIMENSION, dtype=np.float32)
            
            # Generate embedding using Azure OpenAI
            response = self.client.embeddings.create(
                model=self.model,
                input=text
            )
            embedding = response.data[0].embedding
            
            # Convert to numpy array
            embedding_array = np.array(embedding, dtype=np.float32)
            
            # Ensure correct dimension
            if embedding_array.shape[0] != EMBEDDING_DIMENSION:
                logger.warning(f"Unexpected embedding dimension: {embedding_array.shape[0]}")
            
            return embedding_array
        
        except Exception as e:
            logger.error(f"Error generating embedding: {e}")
            raise
    
    @retry(stop=stop_after_attempt(3), wait=wait_exponential(multiplier=1, min=4, max=10))
    def batch_generate_embeddings(self, texts: List[str], batch_size: int = 100) -> np.ndarray:
        """
        Generate embeddings for multiple texts in batches.
        
        Args:
            texts: List of texts to embed
            batch_size: Number of texts to process in each batch
            
        Returns:
            Numpy array of shape (len(texts), EMBEDDING_DIMENSION)
        """
        try:
            if not texts:
                return np.array([], dtype=np.float32).reshape(0, EMBEDDING_DIMENSION)
            
            all_embeddings = []
            
            # Process in batches
            for i in range(0, len(texts), batch_size):
                batch = texts[i:i + batch_size]
                
                # Filter out empty texts
                valid_texts = [t if t and t.strip() else " " for t in batch]
                
                try:
                    # Generate embeddings for batch using Azure OpenAI
                    response = self.client.embeddings.create(
                        model=self.model,
                        input=valid_texts
                    )
                    batch_embeddings = [item.embedding for item in response.data]
                    
                    # Convert to numpy arrays
                    for emb in batch_embeddings:
                        embedding_array = np.array(emb, dtype=np.float32)
                        all_embeddings.append(embedding_array)
                    
                    logger.info(f"Generated embeddings for batch {i//batch_size + 1}/{(len(texts)-1)//batch_size + 1}")
                
                except Exception as e:
                    logger.error(f"Error processing batch {i//batch_size + 1}: {e}")
                    # Add zero vectors for failed batch
                    for _ in batch:
                        all_embeddings.append(np.zeros(EMBEDDING_DIMENSION, dtype=np.float32))
            
            # Stack into single array
            embeddings_array = np.vstack(all_embeddings)
            
            return embeddings_array
        
        except Exception as e:
            logger.error(f"Error in batch embedding generation: {e}")
            raise
    
    def get_embedding_dimension(self) -> int:
        """
        Get the dimension of embeddings.
        
        Returns:
            Embedding dimension (1536 for text-embedding-ada-002)
        """
        return EMBEDDING_DIMENSION


# Global instance
_embedding_generator = None


def get_embedding_generator() -> EmbeddingGenerator:
    """Get or create the global embedding generator instance."""
    global _embedding_generator
    if _embedding_generator is None:
        _embedding_generator = EmbeddingGenerator()
    return _embedding_generator

