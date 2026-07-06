"""
Streamlit frontend for the PDF RAG Search Application.
"""
import streamlit as st
import requests
import time
import os
from typing import Dict, Any, List

# API configuration
API_BASE_URL = os.getenv("API_BASE_URL", "http://127.0.0.1:8000/api")

# Page configuration
st.set_page_config(
    page_title="PDF RAG Search",
    page_icon="📚",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS
st.markdown("""
    <style>
    .main-header {
        font-size: 3rem;
        font-weight: bold;
        color: #1f77b4;
        text-align: center;
        margin-bottom: 2rem;
    }
    .upload-section {
        background-color: #f0f2f6;
        padding: 2rem;
        border-radius: 10px;
        margin-bottom: 2rem;
    }
    .result-section {
        background-color: #ffffff;
        padding: 1.5rem;
        border-radius: 10px;
        margin-top: 2rem;
        box-shadow: 0 2px 4px rgba(0,0,0,0.1);
    }
    .source-item {
        background-color: #f8f9fa;
        padding: 1rem;
        border-radius: 5px;
        margin: 0.5rem 0;
        border-left: 4px solid #1f77b4;
    }
    </style>
""", unsafe_allow_html=True)


def check_api_health() -> bool:
    """Check if the API is available."""
    try:
        response = requests.get(f"{API_BASE_URL}/health", timeout=5)
        return response.status_code == 200
    except Exception:
        return False


def upload_pdf(file) -> Dict[str, Any]:
    """Upload a PDF file to the API."""
    try:
        files = {"file": (file.name, file.getvalue(), "application/pdf")}
        response = requests.post(f"{API_BASE_URL}/upload", files=files, timeout=60)
        response.raise_for_status()
        return response.json()
    except Exception as e:
        st.error(f"Error uploading file: {str(e)}")
        return None


def get_upload_status(upload_id: str) -> Dict[str, Any]:
    """Get the status of a PDF upload."""
    try:
        response = requests.get(f"{API_BASE_URL}/upload-status/{upload_id}", timeout=5)
        response.raise_for_status()
        return response.json()
    except Exception as e:
        st.error(f"Error getting upload status: {str(e)}")
        return None


def search_query(query: str, k: int = 5) -> Dict[str, Any]:
    """Search the document database."""
    try:
        payload = {"query": query, "k": k}
        response = requests.post(f"{API_BASE_URL}/search", json=payload, timeout=30)
        response.raise_for_status()
        return response.json()
    except Exception as e:
        st.error(f"Error searching: {str(e)}")
        return None


def list_documents() -> Dict[str, Any]:
    """List all processed documents."""
    try:
        response = requests.get(f"{API_BASE_URL}/documents", timeout=10)
        response.raise_for_status()
        return response.json()
    except Exception as e:
        st.error(f"Error listing documents: {str(e)}")
        return None


def delete_document(doc_id: str) -> bool:
    """Delete a document."""
    try:
        response = requests.delete(f"{API_BASE_URL}/document/{doc_id}", timeout=10)
        response.raise_for_status()
        result = response.json()
        print(f"Delete response: {result}")  # Debug logging
        return True
    except requests.exceptions.RequestException as e:
        st.error(f"Error deleting document: {str(e)}")
        print(f"Delete error: {e}")  # Debug logging
        return False
    except Exception as e:
        st.error(f"Unexpected error deleting document: {str(e)}")
        print(f"Delete error: {e}")  # Debug logging
        return False


def display_image(image_path: str):
    """Display an image from file path."""
    try:
        if not image_path or image_path.strip() == '':
            st.warning("No image path provided")
            return
        
        # Ensure we're using relative path only (not absolute path)
        # Extract just the relative part if full path is provided
        if 'processed_images' in image_path:
            # Extract from processed_images onwards
            parts = image_path.split('processed_images')
            if len(parts) > 1:
                image_path = parts[-1].lstrip('/').lstrip('\\')
        
        # Normalize path separators to forward slashes for URL
        image_path = image_path.replace('\\', '/')
        
        # Construct the full URL for the image API endpoint
        image_url = f"{API_BASE_URL}/images/{image_path}"
        
        # Debug: show the URL being used
        st.caption(f"Loading from: {image_url}")
        
        # Use st.image with the API URL
        # Streamlit will handle the HTTP request internally
        st.image(image_url, caption=f"Image from document")
    except Exception as e:
        st.warning(f"Could not display image: {str(e)}")


def main():
    """Main application."""
    # Header
    st.markdown('<h1 class="main-header">📚 PDF RAG Search Application</h1>', unsafe_allow_html=True)
    
    # Check API health
    if not check_api_health():
        st.error("⚠️ API server is not available. Please make sure the backend is running.")
        st.info("Start the backend with: `python -m backend.app` or `uvicorn backend.app:app`")
        return
    
    # Sidebar
    with st.sidebar:
        st.header("Navigation")
        page = st.radio(
            "Choose a page",
            ["Upload Documents", "Search", "Manage Documents"]
        )
        
        st.markdown("---")
        st.header("API Status")
        if check_api_health():
            st.success("✅ API Connected")
        else:
            st.error("❌ API Disconnected")
    
    # Main content
    if page == "Upload Documents":
        upload_page()
    elif page == "Search":
        search_page()
    elif page == "Manage Documents":
        manage_documents_page()


def upload_page():
    """Upload documents page."""
    st.header("📤 Upload PDF Documents")
    
    st.markdown("""
    Upload PDF files to add them to the searchable database. The system will:
    - Extract text and images from your PDFs
    - Generate embeddings for semantic search
    - Make them searchable through natural language queries
    """)
    
    uploaded_files = st.file_uploader(
        "Choose PDF files",
        type=['pdf'],
        accept_multiple_files=True,
        help="You can upload multiple PDF files at once"
    )
    
    if uploaded_files:
        if st.button("Upload and Process", type="primary"):
            for uploaded_file in uploaded_files:
                with st.spinner(f"Uploading {uploaded_file.name}..."):
                    result = upload_pdf(uploaded_file)
                    
                    if result:
                        upload_id = result.get('upload_id')
                        doc_id = result.get('doc_id')
                        
                        st.success(f"✅ {uploaded_file.name} uploaded successfully!")
                        st.info(f"Upload ID: {upload_id}")
                        
                        # Monitor processing
                        progress_bar = st.progress(0)
                        status_text = st.empty()
                        
                        while True:
                            status = get_upload_status(upload_id)
                            if status:
                                current_status = status.get('status', 'unknown')
                                progress = status.get('progress', 0)
                                
                                progress_bar.progress(progress / 100)
                                status_text.text(f"Status: {current_status.upper()} ({progress}%)")
                                
                                if current_status == 'completed':
                                    st.success(f"✅ Processing complete for {uploaded_file.name}!")
                                    st.json(status.get('metadata', {}))
                                    break
                                elif current_status == 'failed':
                                    st.error(f"❌ Processing failed: {status.get('error', 'Unknown error')}")
                                    break
                                
                                time.sleep(2)
                            else:
                                break


def search_page():
    """Search page."""
    st.header("🔍 Search Documents")
    
    st.markdown("""
    Ask questions about your uploaded documents using natural language.
    The system will find relevant information and generate comprehensive answers.
    """)
    
    # Search input
    query = st.text_input(
        "Enter your question",
        placeholder="e.g., What are the main findings in the research paper?",
        help="Ask any question about your uploaded documents"
    )
    
    col1, col2 = st.columns([3, 1])
    with col1:
        k = st.slider("Number of results", min_value=1, max_value=10, value=5)
    with col2:
        search_button = st.button("Search", type="primary", use_container_width=True)
    
    if search_button and query:
        with st.spinner("Searching documents..."):
            results = search_query(query, k)
            
            if results:
                st.markdown("---")
                st.markdown('<div class="result-section">', unsafe_allow_html=True)
                
                # Display answer
                st.subheader("💡 Answer")
                st.markdown(results.get('answer', 'No answer generated'))
                
                # Display sources
                sources = results.get('sources', [])
                if sources:
                    st.subheader("📚 Sources")
                    for i, source in enumerate(sources, 1):
                        with st.expander(f"Source {i}: {source.get('source_doc', 'Unknown')} (Page {source.get('page_num', 'N/A')})"):
                            st.write(f"**Content Type:** {source.get('content_type', 'text')}")
                            st.write(f"**Relevance Score:** {source.get('score', 0):.3f}")
                            st.write(f"**Preview:**")
                            st.write(source.get('content_preview', ''))
                
                # Display images
                images = results.get('images', [])
                if images:
                    st.subheader("🖼️ Related Images")
                    for img in images:
                        image_path = img.get('image_path', '')
                        if image_path:
                            col1, col2 = st.columns([1, 3])
                            with col1:
                                display_image(image_path)
                            with col2:
                                st.write(f"**Page:** {img.get('page_num', 'N/A')}")
                                st.write(f"**Source:** {img.get('source_doc', 'Unknown')}")
                                st.write(f"**Description:** {img.get('description', 'No description')}")
                
                st.markdown('</div>', unsafe_allow_html=True)
    
    elif search_button and not query:
        st.warning("Please enter a search query")


def manage_documents_page():
    """Manage documents page."""
    st.header("📋 Manage Documents")
    
    st.markdown("""
    View and manage your uploaded documents.
    """)
    
    if st.button("Refresh Document List", type="primary"):
        st.cache_data.clear()
        st.rerun()
    
    documents_data = list_documents()
    
    if documents_data:
        st.metric("Total Documents", documents_data.get('total_documents', 0))
        st.metric("Total Vectors", documents_data.get('total_vectors', 0))
        
        st.markdown("---")
        st.subheader("Document List")
        
        documents = documents_data.get('documents', [])
        if documents:
            for doc in documents:
                with st.container():
                    col1, col2, col3 = st.columns([3, 1, 1])
                    
                    with col1:
                        st.write(f"**{doc.get('filename', 'Unknown')}**")
                        st.write(f"Document ID: `{doc.get('doc_id', 'N/A')}`")
                        status = doc.get('status', 'unknown')
                        if status == 'completed':
                            st.success(f"Status: {status}")
                        elif status == 'processing':
                            st.info(f"Status: {status}")
                        else:
                            st.warning(f"Status: {status}")
                    
                    with col2:
                        metadata = doc.get('metadata', {})
                        if metadata:
                            st.write(f"**Pages:** {metadata.get('total_pages', 'N/A')}")
                            st.write(f"**Chunks:** {metadata.get('num_chunks', 'N/A')}")
                    
                    with col3:
                        doc_id = doc.get('doc_id')
                        if doc_id:
                            if st.button("Delete", key=f"delete_{doc_id}", type="secondary"):
                                with st.spinner(f"Deleting {doc.get('filename', 'document')}..."):
                                    if delete_document(doc_id):
                                        st.success(f"✅ Deleted {doc.get('filename', 'document')} successfully!")
                                        time.sleep(1)
                                        st.cache_data.clear()
                                        st.rerun()
                                    else:
                                        st.error(f"❌ Failed to delete {doc.get('filename', 'document')}")
                    
                    st.markdown("---")
        else:
            st.info("No documents uploaded yet.")
    else:
        st.warning("Could not retrieve document list.")


if __name__ == "__main__":
    main()

