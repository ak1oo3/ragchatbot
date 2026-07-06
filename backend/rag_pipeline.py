"""
RAG Pipeline for retrieval and generation.
"""
import logging
import os
from typing import List, Dict, Any, Optional
from openai import AzureOpenAI
from tenacity import retry, stop_after_attempt, wait_exponential

from backend.config import (
    AZURE_OPENAI_ENDPOINT,
    AZURE_OPENAI_API_KEY,
    CHAT_API_VERSION,
    CHAT_MODEL,
    TOP_K_RESULTS
)
from backend.vector_db import VectorDatabase

logger = logging.getLogger(__name__)


class RAGPipeline:
    """RAG pipeline for retrieval and generation."""
    
    def __init__(
        self,
        embeddings_generator,
        vector_db: VectorDatabase,
        azure_openai_client: Optional[AzureOpenAI] = None
    ):
        """
        Initialize RAG pipeline.
        
        Args:
            embeddings_generator: Embedding generator instance
            vector_db: Vector database instance
            azure_openai_client: Azure OpenAI client (optional, will create if not provided)
        """
        self.embeddings_generator = embeddings_generator
        self.vector_db = vector_db
        
        if azure_openai_client is None:
            self.llm_client = AzureOpenAI(
                api_key=AZURE_OPENAI_API_KEY,
                api_version=CHAT_API_VERSION,
                azure_endpoint=AZURE_OPENAI_ENDPOINT
            )
        else:
            self.llm_client = azure_openai_client
        
        logger.info("Initialized RAG pipeline")
    
    @retry(stop=stop_after_attempt(3), wait=wait_exponential(multiplier=1, min=4, max=10))
    def _retrieve_context(self, question: str, k: int = TOP_K_RESULTS) -> List[Dict[str, Any]]:
        """
        Retrieve relevant context for a question.
        
        Args:
            question: User question
            k: Number of documents to retrieve
            
        Returns:
            List of retrieved documents with metadata
        """
        try:
            # Generate embedding for question
            query_embedding = self.embeddings_generator.generate_embedding(question)
            logger.info(f"Query embedding shape: {query_embedding.shape}")
            logger.info(f"Vector DB has {self.vector_db.index.ntotal} vectors")
            
            # Search vector database
            results = self.vector_db.search(query_embedding, k=k)
            
            logger.info(f"Retrieved {len(results)} relevant documents")
            if results:
                logger.info(f"Top result score: {results[0].get('score', 0.0)}")
                logger.info(f"Top result content preview: {results[0].get('content', '')[:100]}...")
            else:
                logger.warning("No results found in vector database search")
                # Try with lower threshold for debugging
                logger.info("Attempting search with debug info...")
                distances, indices = self.vector_db.index.search(query_embedding.reshape(1, -1), min(k, self.vector_db.index.ntotal))
                logger.info(f"Raw search results - distances: {distances[0][:5]}, indices: {indices[0][:5]}")
            
            return results
        
        except Exception as e:
            logger.error(f"Error retrieving context: {e}")
            return []
    
    def _format_prompt(self, question: str, context: List[Dict[str, Any]]) -> str:
        """
        Format the prompt with context and question.
        
        Args:
            question: User question
            context: Retrieved context documents
            
        Returns:
            Formatted prompt string
        """
        if not context:
            return f"""You are a helpful AI assistant. Answer the user's question.

Question: {question}

Answer:"""
        
        # Format context
        context_parts = []
        for i, doc in enumerate(context, 1):
            content = doc.get('content', '')
            metadata = doc.get('metadata', {})
            source = metadata.get('source_doc', 'Unknown')
            page_num = metadata.get('page_num', 'N/A')
            content_type = metadata.get('content_type', 'text')
            
            context_parts.append(
                f"[Document {i} - {source}, Page {page_num}, Type: {content_type}]\n{content}\n"
            )
        
        context_text = "\n".join(context_parts)
        
        prompt = f"""You are a helpful AI assistant. Answer the user's question based on the provided context.

If the answer is not in the context, say so clearly. Always cite your sources with document name and page number.

Context:
{context_text}

Question: {question}

Answer:"""
        
        return prompt
    
    @retry(stop=stop_after_attempt(3), wait=wait_exponential(multiplier=1, min=4, max=10))
    def _generate_response(self, question: str, context: List[Dict[str, Any]]) -> str:
        """
        Generate response using GPT-4o.
        
        Args:
            question: User question
            context: Retrieved context documents
            
        Returns:
            Generated answer
        """
        try:
            prompt = self._format_prompt(question, context)
            
            response = self.llm_client.chat.completions.create(
                model=CHAT_MODEL,
                messages=[
                    {"role": "system", "content": "You are a helpful AI assistant that provides accurate answers based on provided context."},
                    {"role": "user", "content": prompt}
                ],
                temperature=0.7,
                max_tokens=1000
            )
            
            answer = response.choices[0].message.content
            return answer
        
        except Exception as e:
            logger.error(f"Error generating response: {e}")
            raise
    
    def query(self, question: str, k: int = TOP_K_RESULTS) -> Dict[str, Any]:
        """
        Process a query and return answer with sources.
        
        Args:
            question: User question
            k: Number of documents to retrieve
            
        Returns:
            Dictionary with:
            {
                'answer': str,
                'sources': List[Dict]
            }
        """
        try:
            if not question or not question.strip():
                return {
                    'answer': 'Please provide a valid question.',
                    'sources': []
                }
            
            # Retrieve context
            context = self._retrieve_context(question, k)
            
            if not context:
                return {
                    'answer': 'I could not find relevant information to answer your question. Please try rephrasing or upload more documents.',
                    'sources': []
                }
            
            # Generate answer
            answer = self._generate_response(question, context)
            
            # Extract sources
            sources = []
            for doc in context:
                metadata = doc.get('metadata', {})
                sources.append({
                    'source_doc': metadata.get('source_doc', 'Unknown'),
                    'page_num': metadata.get('page_num', 'N/A'),
                    'content_type': metadata.get('content_type', 'text'),
                    'score': doc.get('score', 0.0),
                    'content_preview': doc.get('content', '')[:200] + '...' if len(doc.get('content', '')) > 200 else doc.get('content', '')
                })
            
            result = {
                'answer': answer,
                'sources': sources
            }
            
            logger.info(f"Generated answer for query: {question[:50]}...")
            return result
        
        except Exception as e:
            logger.error(f"Error processing query: {e}")
            return {
                'answer': f'An error occurred while processing your query: {str(e)}',
                'sources': []
            }
