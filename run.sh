#!/bin/bash

# Check if virtual environment exists
if [ ! -d "fresh_venv" ]; then
    echo "Creating virtual environment..."
    python3 -m venv fresh_venv
    source fresh_venv/bin/activate
    pip install --upgrade pip setuptools wheel
    pip install -r requirements.txt
    pip install requests
else
    source fresh_venv/bin/activate
fi

# Check if model exists
if [ ! -f "phishing_model.pkl" ]; then
    echo "Training the model..."
    python phishing_detection.py
fi

# Run the API server
echo "Starting the API server..."
echo "API will be available at http://localhost:5000"
echo "Press Ctrl+C to stop the server"
python phishing_api.py 