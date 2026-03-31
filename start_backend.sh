#!/bin/bash
echo "Starting PDF RAG Backend API..."
python -m uvicorn backend.app:app --host 0.0.0.0 --port 8000 --reload

