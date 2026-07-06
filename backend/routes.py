"""
API routes for the RAG application.
"""
import os
import shutil
import threading
import logging
from typing import Dict, Any, List
from fastapi import APIRouter, UploadFile, File, HTTPException, BackgroundTasks
from fastapi.responses import JSONResponse, FileResponse
from pydantic import BaseModel
import uuid
import time
import numpy as np

from backend.config import (
    UPLOAD_DIR,
    MAX_FILE_SIZE,
    TOP_K_RESULTS,
    AZURE_OPENAI_ENDPOINT,
    AZURE_OPENAI_API_KEY,
    CHAT_API_VERSION
)
from backend.utils import (
    generate_upload_id,
    generate_doc_id,
    sanitize_filename,
    format_error_message
)
from backend.pdf_processor import PDFProcessor
from backend.embeddings import get_embedding_generator
from backend.vector_db import VectorDatabase
from backend.rag_pipeline import RAGPipeline
from openai import AzureOpenAI

logger = logging.getLogger(__name__)

# Initialize router
router = APIRouter()

# Global state (in production, use a proper state management)
upload_status: Dict[str, Dict[str, Any]] = {}
processing_lock = threading.Lock()

# Initialize components (will be set in app.py)
pdf_processor: PDFProcessor = None
vector_db: VectorDatabase = None
rag_pipeline: RAGPipeline = None
azure_openai_client: AzureOpenAI = None


def initialize_components(
    processor: PDFProcessor,
    vdb: VectorDatabase,
    pipeline: RAGPipeline,
    client: AzureOpenAI
):
    """Initialize global components."""
    global pdf_processor, vector_db, rag_pipeline, azure_openai_client
    pdf_processor = processor
    vector_db = vdb
    rag_pipeline = pipeline
    azure_openai_client = client


class SearchRequest(BaseModel):
    """Search request model."""
    query: str
    k: int = TOP_K_RESULTS


class SearchResponse(BaseModel):
    """Search response model."""
    answer: str
    sources: List[Dict[str, Any]]


def process_pdf_background(upload_id: str, file_path: str, doc_id: str):
    """Background task to process PDF."""
    try:
        with processing_lock:
            upload_status[upload_id]['status'] = 'processing'
            upload_status[upload_id]['progress'] = 10
        
        # Process PDF
        result = pdf_processor.process_pdf_complete(
            file_path,
            doc_id,
            azure_openai_client=azure_openai_client
        )
        
        with processing_lock:
            upload_status[upload_id]['progress'] = 50
        
        # Generate embeddings for text chunks
        text_chunks = result['text_chunks']
        text_embeddings = None
        text_metadatas = []
        
        if text_chunks:
            text_embeddings = get_embedding_generator().batch_generate_embeddings(text_chunks)
            for i, chunk in enumerate(text_chunks):
                text_metadatas.append({
                    'content': chunk,
                    'content_type': 'text',
                    'doc_id': doc_id,
                    'source_doc': result['metadata']['source_doc'],
                    'page_num': 'N/A',  # Could extract from chunk if needed
                    'chunk_index': i
                })
        
        with processing_lock:
            upload_status[upload_id]['progress'] = 85
        
        # Add embeddings to vector database
        if text_embeddings is not None and len(text_embeddings) > 0:
            vector_db.add_embeddings(text_embeddings, text_metadatas)
            vector_db.save_to_disk()
        
        with processing_lock:
            upload_status[upload_id]['status'] = 'completed'
            upload_status[upload_id]['progress'] = 100
            upload_status[upload_id]['doc_id'] = doc_id
            upload_status[upload_id]['metadata'] = result['metadata']
        
        logger.info(f"Successfully processed PDF: {file_path}")
    
    except Exception as e:
        with processing_lock:
            upload_status[upload_id]['status'] = 'failed'
            upload_status[upload_id]['error'] = format_error_message(e)
        logger.error(f"Error processing PDF {file_path}: {e}")


@router.post("/upload")
async def upload_pdf(
    background_tasks: BackgroundTasks,
    file: UploadFile = File(...)
):
    """
    Upload a PDF file for processing.
    
    Returns:
        Upload ID and status
    """
    try:
        # Validate file
        if not file.filename.endswith('.pdf'):
            raise HTTPException(status_code=400, detail="Only PDF files are allowed")
        
        # Save file
        upload_id = generate_upload_id()
        doc_id = generate_doc_id(file.filename)
        sanitized_filename = sanitize_filename(file.filename)
        file_path = os.path.join(UPLOAD_DIR, f"{doc_id}_{sanitized_filename}")
        
        # Check file size
        content = await file.read()
        if len(content) > MAX_FILE_SIZE:
            raise HTTPException(status_code=400, detail=f"File size exceeds maximum allowed size: {MAX_FILE_SIZE} bytes")
        
        # Write file
        with open(file_path, 'wb') as f:
            f.write(content)
        
        # Initialize upload status
        with processing_lock:
            upload_status[upload_id] = {
                'status': 'queued',
                'progress': 0,
                'filename': file.filename,
                'doc_id': doc_id,
                'created_at': time.time()
            }
        
        # Process in background
        background_tasks.add_task(process_pdf_background, upload_id, file_path, doc_id)
        
        return JSONResponse({
            'upload_id': upload_id,
            'doc_id': doc_id,
            'filename': file.filename,
            'status': 'queued',
            'message': 'File uploaded successfully, processing started'
        })
    
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error uploading file: {e}")
        raise HTTPException(status_code=500, detail=format_error_message(e))


@router.get("/upload-status/{upload_id}")
async def get_upload_status(upload_id: str):
    """
    Get the status of a PDF upload/processing.
    
    Args:
        upload_id: Upload ID returned from /upload
        
    Returns:
        Status information
    """
    try:
        with processing_lock:
            if upload_id not in upload_status:
                raise HTTPException(status_code=404, detail="Upload ID not found")
            
            status = upload_status[upload_id].copy()
            return JSONResponse(status)
    
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error getting upload status: {e}")
        raise HTTPException(status_code=500, detail=format_error_message(e))


@router.post("/search", response_model=SearchResponse)
async def search(query_request: SearchRequest):
    """
    Search the document database and generate an answer.
    
    Args:
        query_request: Search request with query and optional k parameter
        
    Returns:
        Answer with sources and images
    """
    try:
        if not query_request.query or not query_request.query.strip():
            raise HTTPException(status_code=400, detail="Query cannot be empty")
        
        if len(query_request.query) > 1000:
            raise HTTPException(status_code=400, detail="Query is too long (max 1000 characters)")
        
        # Process query
        result = rag_pipeline.query(query_request.query, k=query_request.k)
        
        return JSONResponse(result)
    
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error processing search: {e}")
        raise HTTPException(status_code=500, detail=format_error_message(e))


@router.get("/documents")
async def list_documents():
    """
    List all processed documents.
    
    Returns:
        List of documents with metadata
    """
    try:
        stats = vector_db.get_stats()
        documents = []
        
        for doc_id in stats['documents']:
            # Get document info from upload status
            doc_info = None
            for upload_id, status in upload_status.items():
                if status.get('doc_id') == doc_id:
                    doc_info = {
                        'doc_id': doc_id,
                        'filename': status.get('filename', 'Unknown'),
                        'status': status.get('status', 'unknown'),
                        'metadata': status.get('metadata', {})
                    }
                    break
            
            if doc_info:
                documents.append(doc_info)
            else:
                documents.append({
                    'doc_id': doc_id,
                    'filename': 'Unknown',
                    'status': 'unknown'
                })
        
        return JSONResponse({
            'total_documents': len(documents),
            'total_vectors': stats['total_vectors'],
            'documents': documents
        })
    
    except Exception as e:
        logger.error(f"Error listing documents: {e}")
        raise HTTPException(status_code=500, detail=format_error_message(e))


@router.delete("/document/{doc_id}")
async def delete_document(doc_id: str):
    """
    Delete a document from the database.
    
    Args:
        doc_id: Document ID to delete
        
    Returns:
        Deletion confirmation
    """
    try:
        # Find and remove uploaded PDF file
        upload_dir = UPLOAD_DIR
        deleted_files = []
        
        if os.path.exists(upload_dir):
            for file in os.listdir(upload_dir):
                if file.startswith(doc_id):
                    file_path = os.path.join(upload_dir, file)
                    if os.path.isfile(file_path):
                        os.remove(file_path)
                        deleted_files.append(file_path)
                        logger.info(f"Deleted uploaded PDF: {file_path}")
        
        # Delete from vector database
        num_deleted = vector_db.delete_by_source(doc_id)
        
        # Save updated database
        vector_db.save_to_disk()
        
        # Remove from upload status (if it exists)
        with processing_lock:
            for upload_id, status in list(upload_status.items()):
                if status.get('doc_id') == doc_id:
                    del upload_status[upload_id]
        
        return JSONResponse({
            'doc_id': doc_id,
            'deleted': True,
            'vectors_deleted': num_deleted,
            'files_deleted': len(deleted_files),
            'message': f'Document {doc_id} and all associated files deleted successfully'
        })
    
    except Exception as e:
        logger.error(f"Error deleting document: {e}")
        raise HTTPException(status_code=500, detail=format_error_message(e))


@router.post("/clear-database")
async def clear_database():
    """
    Clear all vectors from the vector database.
    
    WARNING: This permanently deletes all vectors and cannot be undone.
    
    Returns:
        Confirmation of clearing
    """
    try:
        vector_db.clear_all()
        vector_db.save_to_disk()
        
        # Clear upload status
        with processing_lock:
            upload_status.clear()
        
        return JSONResponse({
            'status': 'success',
            'message': 'Vector database cleared successfully',
            'total_vectors': vector_db.index.ntotal
        })
    
    except Exception as e:
        logger.error(f"Error clearing database: {e}")
        raise HTTPException(status_code=500, detail=format_error_message(e))


@router.get("/health")
async def health_check():
    """
    Health check endpoint.
    
    Returns:
        Health status
    """
    try:
        stats = vector_db.get_stats()
        return JSONResponse({
            'status': 'healthy',
            'vector_db': {
                'total_vectors': stats['total_vectors'],
                'num_documents': stats['num_documents']
            },
            'components': {
                'pdf_processor': pdf_processor is not None,
                'vector_db': vector_db is not None,
                'rag_pipeline': rag_pipeline is not None
            }
        })
    
    except Exception as e:
        logger.error(f"Health check failed: {e}")
        return JSONResponse({
            'status': 'unhealthy',
            'error': format_error_message(e)
        }, status_code=503)
