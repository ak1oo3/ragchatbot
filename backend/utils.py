"""
Utility functions for the RAG application.
"""
import os
import uuid
import hashlib
from typing import Optional
from datetime import datetime
import logging

logger = logging.getLogger(__name__)


def generate_upload_id() -> str:
    """Generate a unique upload ID."""
    return str(uuid.uuid4())


def generate_doc_id(filename: str) -> str:
    """Generate a unique document ID based on filename."""
    timestamp = datetime.now().isoformat()
    unique_str = f"{filename}_{timestamp}"
    return hashlib.md5(unique_str.encode()).hexdigest()


def validate_pdf_file(file_path: str) -> bool:
    """Validate that the file is a PDF."""
    if not os.path.exists(file_path):
        return False
    
    allowed_extensions = {'.pdf'}
    file_ext = os.path.splitext(file_path)[1].lower()
    return file_ext in allowed_extensions


def validate_file_size(file_path: str, max_size: int) -> bool:
    """Validate that the file size is within limits."""
    if not os.path.exists(file_path):
        return False
    
    file_size = os.path.getsize(file_path)
    return file_size <= max_size


def sanitize_filename(filename: str) -> str:
    """Sanitize filename to prevent path traversal and other issues."""
    # Remove path components
    filename = os.path.basename(filename)
    # Remove or replace dangerous characters
    dangerous_chars = ['<', '>', ':', '"', '|', '?', '*', '\\', '/']
    for char in dangerous_chars:
        filename = filename.replace(char, '_')
    return filename


def format_error_message(error: Exception) -> str:
    """Format exception into a user-friendly error message."""
    error_type = type(error).__name__
    error_msg = str(error)
    return f"{error_type}: {error_msg}"


def log_error(error: Exception, context: Optional[str] = None):
    """Log error with context."""
    if context:
        logger.error(f"{context}: {format_error_message(error)}")
    else:
        logger.error(format_error_message(error))

