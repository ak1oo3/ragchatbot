"""
PDF processing module for extracting text from PDF files.
"""
import os
import fitz  # PyMuPDF
from typing import List, Tuple, Dict, Any, Optional
from PIL import Image
import io
import logging
from langchain.text_splitter import RecursiveCharacterTextSplitter
from tenacity import retry, stop_after_attempt, wait_exponential

from backend.config import (
    CHUNK_SIZE,
    CHUNK_OVERLAP,
    PROCESSED_IMAGES_DIR,
    MAX_FILE_SIZE
)
from backend.utils import validate_pdf_file, validate_file_size, log_error

logger = logging.getLogger(__name__)


class PDFProcessor:
    """Process PDF files to extract text and images."""
    
    def __init__(self):
        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=CHUNK_SIZE,
            chunk_overlap=CHUNK_OVERLAP,
            length_function=len,
            separators=["\n\n", "\n", ". ", " ", ""]
        )
    
    def extract_text_from_pdf(self, pdf_path: str) -> str:
        """
        Extract all text from a PDF file.
        
        Args:
            pdf_path: Path to the PDF file
            
        Returns:
            Extracted text as a string
        """
        try:
            with fitz.open(pdf_path) as doc:
                text_parts = []
                
                for page_num in range(len(doc)):
                    page = doc[page_num]
                    text = page.get_text()
                    if text.strip():
                        text_parts.append(f"--- Page {page_num + 1} ---\n{text}")
                
                full_text = "\n\n".join(text_parts)
            
            if not full_text.strip():
                logger.warning(f"No text extracted from {pdf_path}")
            
            return full_text
        
        except Exception as e:
            log_error(e, f"Error extracting text from {pdf_path}")
            raise
    
    def extract_images_from_pdf(self, pdf_path: str) -> List[Tuple[bytes, int, int]]:
        """
        Extract all embedded raster and vector images from a PDF file.
        
        Only extracts images that are explicitly embedded or drawn in the PDF.
        Text pages are NOT rendered as images - use extract_text_from_pdf for text content.
        
        Args:
            pdf_path: Path to the PDF file
            
        Returns:
            List of tuples: (image_bytes, page_num, image_index)
        """
        try:
            with fitz.open(pdf_path) as doc:
                images = []
                
                for page_num in range(len(doc)):
                    page = doc[page_num]
                    
                    # Extract only explicitly embedded raster images
                    image_list = page.get_images(full=True)
                    
                    for img_index, img in enumerate(image_list):
                        try:
                            xref = img[0]
                            base_image = doc.extract_image(xref)
                            image_bytes = base_image["image"]
                            images.append((image_bytes, page_num + 1, img_index))
                            logger.info(f"Extracted embedded image {img_index} from page {page_num + 1}")
                        except Exception as e:
                            logger.warning(f"Failed to extract image {img_index} from page {page_num + 1}: {e}")
                            continue
                
                logger.info(f"Total images extracted from {pdf_path}: {len(images)}")
                return images
        
        except Exception as e:
            log_error(e, f"Error extracting images from {pdf_path}")
            raise
    
    def chunk_text(self, text: str) -> List[str]:
        """
        Split text into semantic chunks.
        
        Args:
            text: Full text to chunk
            
        Returns:
            List of text chunks
        """
        try:
            chunks = self.text_splitter.split_text(text)
            return chunks
        except Exception as e:
            log_error(e, "Error chunking text")
            raise
    
    @retry(stop=stop_after_attempt(3), wait=wait_exponential(multiplier=1, min=4, max=10))
    def get_image_description(self, image_bytes: bytes, azure_openai_client) -> str:
        """
        Generate a description of an image using GPT-4o Vision.
        
        Args:
            image_bytes: Image data as bytes
            azure_openai_client: Azure OpenAI client instance
            
        Returns:
            Description of the image
        """
        try:
            from openai import AzureOpenAI
            
            # Convert image bytes to base64
            import base64
            image_base64 = base64.b64encode(image_bytes).decode('utf-8')
            
            # Prepare the message for API
            response = azure_openai_client.chat.completions.create(
                model="gpt-4o-2",
                messages=[
                    {
                        "role": "user",
                        "content": [
                            {
                                "type": "text",
                                "text": "Describe this image in detail, including any text, diagrams, charts, or visual elements. Be specific and comprehensive."
                            },
                            {
                                "type": "image_url",
                                "image_url": {
                                    "url": f"data:image/jpeg;base64,{image_base64}"
                                }
                            }
                        ]
                    }
                ],
                max_tokens=500
            )
            
            description = response.choices[0].message.content
            return description
        
        except Exception as e:
            log_error(e, "Error generating image description")
            # Return a fallback description
            return "Image extracted from document (description unavailable)"
    
    def save_image(self, image_bytes: bytes, doc_id: str, page_num: int, img_index: int) -> str:
        """
        Save an extracted image to disk.
        
        Args:
            image_bytes: Image data as bytes
            doc_id: Document identifier
            page_num: Page number
            img_index: Image index on the page
            
        Returns:
            Relative path to saved image (relative to PROCESSED_IMAGES_DIR)
        """
        try:
            # Create directory for this document's images
            doc_image_dir = os.path.join(PROCESSED_IMAGES_DIR, doc_id)
            os.makedirs(doc_image_dir, exist_ok=True)
            
            # Save image
            image_filename = f"page_{page_num}_img_{img_index}.png"
            image_path = os.path.join(doc_image_dir, image_filename)
            
            # Convert to PIL Image and save
            image = Image.open(io.BytesIO(image_bytes))
            image.save(image_path, format='PNG')
            
            # Return relative path for API endpoint
            relative_path = os.path.relpath(image_path, PROCESSED_IMAGES_DIR)
            # Normalize to forward slashes for URL compatibility
            relative_path = relative_path.replace('\\', '/')
            
            return relative_path
        
        except Exception as e:
            log_error(e, f"Error saving image for doc {doc_id}, page {page_num}")
            return ""
    
    def process_pdf_complete(
        self,
        pdf_path: str,
        doc_id: str,
        azure_openai_client=None,
        process_images: bool = True
    ) -> Dict[str, Any]:
        """
        Complete PDF processing: extract text, chunk it, extract images, and describe them.
        
        Args:
            pdf_path: Path to the PDF file
            doc_id: Unique document identifier
            azure_openai_client: Azure OpenAI client for image descriptions
            process_images: Whether to process images (default: True)
            
        Returns:
            Dictionary with processed content:
            {
                'text_chunks': List[str],
                'images': List[Dict],
                'metadata': Dict
            }
        """
        try:
            # Validate file
            if not validate_pdf_file(pdf_path):
                raise ValueError(f"Invalid PDF file: {pdf_path}")
            
            if not validate_file_size(pdf_path, MAX_FILE_SIZE):
                raise ValueError(f"File size exceeds maximum allowed size: {MAX_FILE_SIZE} bytes")
            
            logger.info(f"Processing PDF: {pdf_path}")
            
            # Extract text with context manager
            full_text = self.extract_text_from_pdf(pdf_path)
            
            # Chunk text if not empty
            text_chunks = self.chunk_text(full_text) if full_text.strip() else []
            
            logger.info(f"Extracted {len(text_chunks)} text chunks from {pdf_path}")
            
            # Extract and process images
            images_data = []
            if process_images:
                extracted_images = self.extract_images_from_pdf(pdf_path)
                logger.info(f"Extracted {len(extracted_images)} images from {pdf_path}")
                
                for img_bytes, page_num, img_index in extracted_images:
                    try:
                        # Save image
                        image_path = self.save_image(img_bytes, doc_id, page_num, img_index)
                        
                        if not image_path:
                            logger.warning(f"Failed to save image for page {page_num}, index {img_index} (empty path returned)")
                            continue
                        
                        # Generate description if client is available
                        description = ""
                        if azure_openai_client:
                            try:
                                description = self.get_image_description(img_bytes, azure_openai_client)
                            except Exception as e:
                                logger.warning(f"Failed to generate image description: {e}")
                                description = f"Image from page {page_num}"
                        
                        images_data.append({
                            'image_bytes': img_bytes,
                            'image_path': image_path,
                            'page_num': page_num,
                            'image_index': img_index,
                            'description': description
                        })
                        logger.info(f"Successfully saved image: {image_path}")
                    except Exception as e:
                        logger.warning(f"Failed to process image from page {page_num}: {e}")
                        continue
            
            # Get total pages without opening PDF again
            total_pages = 0
            try:
                with fitz.open(pdf_path) as doc:
                    total_pages = len(doc)
            except Exception as e:
                logger.warning(f"Could not determine page count: {e}")
            
            # Prepare metadata
            metadata = {
                'source_doc': os.path.basename(pdf_path),
                'doc_id': doc_id,
                'total_pages': total_pages,
                'num_chunks': len(text_chunks),
                'num_images': len(images_data)
            }
            
            result = {
                'text_chunks': text_chunks,
                'images': images_data,
                'metadata': metadata
            }
            
            logger.info(f"Successfully processed PDF: {pdf_path}")
            return result
        
        except Exception as e:
            log_error(e, f"Error processing PDF {pdf_path}")
            raise

