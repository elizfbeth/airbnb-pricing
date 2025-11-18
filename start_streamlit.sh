#!/bin/bash

# Startup script for Streamlit Dashboard

# Get the directory where this script is located
SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"

# Activate virtual environment
source "$SCRIPT_DIR/.venv/bin/activate"

# Change to app directory
cd "$SCRIPT_DIR/app"

# Run streamlit
echo "Starting Streamlit Dashboard..."
echo "Open your browser to: http://localhost:8501"
streamlit run streamlit_app.py
