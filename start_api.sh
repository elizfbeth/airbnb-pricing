#!/bin/bash

# Startup script for FastAPI

# Get the directory where this script is located
SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"

# Activate virtual environment
source "$SCRIPT_DIR/.venv/bin/activate"

# Change to api directory
cd "$SCRIPT_DIR/api"

# Run FastAPI
echo "Starting FastAPI Server..."
echo "API docs available at: http://localhost:8000/docs"
uvicorn main:app --reload --host 0.0.0.0 --port 8000
