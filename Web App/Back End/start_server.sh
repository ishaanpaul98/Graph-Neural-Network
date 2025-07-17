#!/bin/bash

# Simple startup script for EC2
cd /home/ubuntu/movie-api

# Activate virtual environment if it exists
if [ -d "venv" ]; then
    source venv/bin/activate
fi

# Install dependencies if requirements.txt exists
if [ -f "requirements.txt" ]; then
    pip install -r requirements.txt
fi

# Start the Flask app with gunicorn
gunicorn --bind 0.0.0.0:8000 --workers 4 --timeout 120 app:app 