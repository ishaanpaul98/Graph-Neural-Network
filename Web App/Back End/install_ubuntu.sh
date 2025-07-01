#!/bin/bash

# Update package list
sudo apt-get update

# Install system dependencies
sudo apt-get install -y \
    python3.11 \
    python3.11-pip \
    python3.11-venv \
    gcc \
    g++ \
    build-essential \
    curl \
    git

# Create virtual environment
python3.11 -m venv venv
source venv/bin/activate

# Upgrade pip
pip install --upgrade pip

# Install Python dependencies
pip install -r requirements.txt

echo "Installation completed successfully!"
echo "To run the application:"
echo "1. Activate virtual environment: source venv/bin/activate"
echo "2. Run the app: python app.py" 