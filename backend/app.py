#fastapi application

import logging
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
import uvicorn

from backend.config import (
    API_HOST,
    API_PORT,
    LOG_LEVEL,
    EMBEDDING_DIMENSION,
    VECTOR_STORE_DIR,
    AZURE_OPENAI_ENDPOINT,
    AZURE_OPENAI_API_KEY,
    AZURE_OPENAI_API_VERSION
)
from backend.pdf_processor import PDFProcessor
from backend.embeddings import get_embedding_generator
from backend.vector_db import VectorDatabase
from backend.rag_pipeline import RAGPipeline
from backend.routes import router, initialize_components
from openai import AzureOpenAI

# Configure logging
logging.basicConfig(
    level=getattr(logging, LOG_LEVEL),
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Initialize FastAPI app
app = FastAPI(
    title="PDF RAG Search API",
    description="RAG-based search system for PDF documents",
    version="1.0.0"
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # In production, specify allowed origins
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Global components
pdf_processor: PDFProcessor = None
vector_db: VectorDatabase = None
rag_pipeline: RAGPipeline = None
azure_openai_client: AzureOpenAI = None


@app.on_event("startup")
async def startup_event():
    """Initialize components on startup."""
    global pdf_processor, vector_db, rag_pipeline, azure_openai_client
    
    try:
        logger.info("Initializing application components...")
        
        # Initialize PDF processor
        pdf_processor = PDFProcessor()
        logger.info("PDF processor initialized")
        
        # Initialize embeddings generator
        embeddings_generator = get_embedding_generator()
        logger.info("Embeddings generator initialized")
        
        # Initialize vector database
        vector_db = VectorDatabase(
            dimension=EMBEDDING_DIMENSION,
            persist_path=VECTOR_STORE_DIR
        )
        
        # Try to load existing database
        if vector_db.load_from_disk():
            logger.info("Loaded existing vector database")
        else:
            logger.info("Starting with empty vector database")
        
        logger.info("Vector database initialized")
        
        # Initialize Azure OpenAI client
        azure_openai_client = AzureOpenAI(
            api_key=AZURE_OPENAI_API_KEY,
            api_version=AZURE_OPENAI_API_VERSION,
            azure_endpoint=AZURE_OPENAI_ENDPOINT
        )
        logger.info("Azure OpenAI client initialized")
        
        # Initialize RAG pipeline
        rag_pipeline = RAGPipeline(
            embeddings_generator=embeddings_generator,
            vector_db=vector_db,
            azure_openai_client=azure_openai_client
        )
        logger.info("RAG pipeline initialized")
        
        # Initialize routes with components
        initialize_components(pdf_processor, vector_db, rag_pipeline, azure_openai_client)
        
        logger.info("Application startup complete")
    
    except Exception as e:
        logger.error(f"Error during startup: {e}")
        raise


@app.on_event("shutdown")
async def shutdown_event():
    """Cleanup on shutdown."""
    global vector_db
    
    try:
        if vector_db:
            vector_db.save_to_disk()
            logger.info("Saved vector database to disk")
    except Exception as e:
        logger.error(f"Error during shutdown: {e}")


@app.exception_handler(Exception)
async def global_exception_handler(request, exc):
    """Global exception handler."""
    logger.error(f"Unhandled exception: {exc}", exc_info=True)
    return JSONResponse(
        status_code=500,
        content={"detail": "An internal server error occurred"}
    )


# Include routes
app.include_router(router, prefix="/api", tags=["api"])


@app.get("/")
async def root():
    """Root endpoint."""
    return {
        "message": "PDF RAG Search API",
        "version": "1.0.0",
        "endpoints": {
            "health": "/api/health",
            "upload": "/api/upload",
            "search": "/api/search",
            "documents": "/api/documents",
            "docs": "/docs"
        }
    }


if __name__ == "__main__":
    uvicorn.run(
        "backend.app:app",
        host=API_HOST,
        port=API_PORT,
        reload=True,
        log_level=LOG_LEVEL.lower()
    )

