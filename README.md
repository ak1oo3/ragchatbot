# PDF RAG Search Application

A complete multimodal RAG (Retrieval-Augmented Generation) application for searching PDF documents using Azure OpenAI. This application extracts text and images from PDFs, generates embeddings, and provides semantic search capabilities with AI-powered answers.

## Features

- 📄 **Multi-PDF Upload**: Upload multiple PDF files simultaneously
- 🔍 **Semantic Search**: Natural language queries with semantic understanding
- 🖼️ **Image Processing**: Extracts and describes images from PDFs using GPT-4o Vision
- 📊 **Vector Database**: FAISS-based vector storage for fast similarity search
- 💬 **AI-Powered Answers**: GPT-4o generates contextual answers with source citations
- 🎨 **Modern UI**: Beautiful Streamlit interface with real-time progress tracking
- 🔄 **Async Processing**: Non-blocking PDF processing with background tasks

## Architecture

```
┌─────────────┐
│  Streamlit  │  Frontend Interface
│   Frontend  │
└──────┬──────┘
       │
       ▼
┌─────────────┐
│   FastAPI   │  REST API Backend
│   Backend   │
└──────┬──────┘
       │
       ├──► PDF Processor (PyMuPDF)
       ├──► Embeddings (Azure OpenAI)
       ├──► Vector DB (FAISS)
       └──► RAG Pipeline (GPT-4o)
```

## Project Structure

```
project_root/
├── backend/
│   ├── app.py              # FastAPI main application
│   ├── config.py           # Configuration and Azure credentials
│   ├── pdf_processor.py    # PDF text and image extraction
│   ├── embeddings.py       # Vector embedding generation
│   ├── vector_db.py        # FAISS vector database
│   ├── rag_pipeline.py     # RAG retrieval and generation
│   ├── routes.py           # API endpoints
│   └── utils.py            # Helper functions
├── frontend/
│   └── app.py              # Streamlit frontend
├── data/
│   ├── uploads/            # Temporary PDF storage
│   ├── vector_store/       # FAISS index and metadata
│   └── processed_images/   # Extracted images
├── requirements.txt
├── .env                    # Environment variables (create from .env.example)
└── README.md
```

## Prerequisites

- Python 3.8 or higher
- Azure OpenAI account with deployed models:
  - **Embedding deployment**: `embed_chat` (using `text-embedding-ada-002` model)
  - **Chat deployment**: `gpt-4o-2` (using `gpt-4o` model)

## Installation

1. **Clone or navigate to the project directory:**
   ```bash
   cd "C:\UsersDesktop\New folder"
   ```

2. **Create a virtual environment (recommended):**
   ```bash
   python -m venv venv
   
   # On Windows:
   venv\Scripts\activate
   
   # On Linux/Mac:
   source venv/bin/activate
   ```

3. **Install dependencies:**
   ```bash
   pip install -r requirements.txt
   ```

4. **Configure environment variables:**
   
   Create a `.env` file with your Azure OpenAI credentials:
   ```bash
   cp .env.example .env
   ```
   
   Edit `.env` with your credentials:
   ```
   AZURE_OPENAI_ENDPOINT=https://your-resource-name.openai.azure.com/
   AZURE_OPENAI_API_KEY=your-api-key-here
   EMBEDDING_MODEL=embed_chat
   CHAT_MODEL=gpt-4o-2
   ```
   
   **Where to find these values:**
   - Go to [Azure Portal](https://portal.azure.com)
   - Navigate to your Azure OpenAI resource
   - Copy the **Endpoint** from "Keys and Endpoint"
   - Copy your **API Key** from "Keys and Endpoint"
   - Verify your **deployment names**: 
     - `embed_chat` for embeddings
     - `gpt-4o-2` for chat and vision tasks

## Usage

### Starting the Backend API

1. **Start the FastAPI server:**
   ```bash
   # Option 1: Using uvicorn directly
   uvicorn backend.app:app --host 0.0.0.0 --port 8000 --reload
   
   # Option 2: Using Python module
   python -m backend.app
   ```

   The API will be available at `http://localhost:8000`
   - API Documentation: `http://localhost:8000/docs`
   - Health Check: `http://localhost:8000/api/health`

### Starting the Frontend

1. **In a new terminal, start the Streamlit app:**
   ```bash
   streamlit run frontend/app.py
   ```

   The frontend will open in your browser at `http://localhost:8501`

### Using the Application

1. **Upload Documents:**
   - Navigate to "Upload Documents" page
   - Select one or more PDF files
   - Click "Upload and Process"
   - Wait for processing to complete (progress bar will show status)

2. **Search Documents:**
   - Navigate to "Search" page
   - Enter your question in natural language
   - Adjust the number of results (1-10)
   - Click "Search"
   - View the AI-generated answer with source citations and related images

3. **Manage Documents:**
   - Navigate to "Manage Documents" page
   - View all uploaded documents
   - Delete documents if needed

## API Endpoints

### POST `/api/upload`
Upload a PDF file for processing.

**Request:** Multipart form data with `file` field
**Response:**
```json
{
  "upload_id": "uuid",
  "doc_id": "hash",
  "filename": "document.pdf",
  "status": "queued"
}
```

### GET `/api/upload-status/{upload_id}`
Get the processing status of an upload.

**Response:**
```json
{
  "status": "completed",
  "progress": 100,
  "doc_id": "hash",
  "metadata": {...}
}
```

### POST `/api/search`
Search the document database.

**Request:**
```json
{
  "query": "What is the main topic?",
  "k": 5
}
```

**Response:**
```json
{
  "answer": "The main topic is...",
  "sources": [
    {
      "source_doc": "document.pdf",
      "page_num": 1,
      "content_type": "text",
      "score": 0.95
    }
  ],
  "images": [...]
}
```

### GET `/api/documents`
List all processed documents.

### DELETE `/api/document/{doc_id}`
Delete a document from the database.

### GET `/api/health`
Health check endpoint.

## Configuration

Key configuration options in `backend/config.py`:

- `CHUNK_SIZE`: Text chunk size (default: 1000 tokens)
- `CHUNK_OVERLAP`: Overlap between chunks (default: 200 tokens)
- `TOP_K_RESULTS`: Number of documents to retrieve (default: 5)
- `SIMILARITY_THRESHOLD`: Minimum similarity score (default: 0.7)
- `MAX_FILE_SIZE`: Maximum file size in bytes (default: 50MB)

## Technical Details

### PDF Processing
- Uses **PyMuPDF (fitz)** for fast text and image extraction
- Preserves document structure and page numbers
- Handles corrupt PDFs and image-heavy documents gracefully

### Embeddings
- Uses **Azure OpenAI embed_chat**
- 1536-dimensional vectors
- Batch processing for efficiency

### Vector Database
- Uses **FAISS IndexFlatL2** for similarity search
- L2 normalization for cosine similarity
- Persistent storage on disk

### RAG Pipeline
- Semantic retrieval using vector similarity
- Context formatting with source citations
- GPT-4o for answer generation
- Image extraction and display

## Performance

- **PDF Processing**: < 30 seconds per file (up to 50MB)
- **Search Query**: < 3 seconds
- **Concurrent Users**: 10-50 simultaneous users
- **Document Capacity**: 100+ PDFs indexed simultaneously

## Error Handling

The application handles:
- Corrupt or invalid PDFs
- Large files with streaming
- API rate limiting with exponential backoff
- Network timeouts with retries
- Memory errors with incremental processing

## Troubleshooting

### API Not Available
- Ensure the backend is running on port 8000
- Check firewall settings
- Verify API_BASE_URL in frontend/app.py

### Upload Fails
- Check file size (max 50MB)
- Ensure file is a valid PDF
- Check backend logs for errors

### Search Returns No Results
- Verify documents were processed successfully
- Check vector database has embeddings
- Try different query phrasing

### Image Processing Issues
- Ensure GPT-4o Vision API is available
- Check image extraction logs
- Verify image paths are accessible

## Development

### Running Tests
```bash
# Add test files to test/ directory
pytest tests/
```

### Code Structure
- **Backend**: FastAPI with async support
- **Frontend**: Streamlit with real-time updates
- **Processing**: Threading for background tasks
- **Storage**: File-based persistence

## Security Notes

- API keys are stored in environment variables
- File uploads are validated and sanitized
- Path traversal protection implemented
- CORS configured (adjust for production)

## License

This project is provided as-is for educational and development purposes.

## Support

For issues or questions:
1. Check the logs in the console
2. Verify Azure OpenAI credentials
3. Ensure all dependencies are installed
4. Check API health endpoint

## Future Enhancements

- [ ] Support for more file formats (DOCX, TXT)
- [ ] Advanced filtering and metadata search
- [ ] User authentication and multi-tenancy
- [ ] Export search results
- [ ] Batch upload API
- [ ] Webhook notifications for processing completion
- [ ] Advanced analytics and usage statistics

