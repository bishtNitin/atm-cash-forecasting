#!/bin/bash
# Setup script for ATM Cash Forecasting System

echo "======================================"
echo "ATM Cash Forecasting - Setup Script"
echo "======================================"

# Check Python version
echo ""
echo "Checking Python version..."
python --version

# Install dependencies
echo ""
echo "Installing dependencies..."
pip install -r requirements.txt

# Create necessary directories
echo ""
echo "Creating directories..."
mkdir -p data/raw data/processed models

echo ""
echo "======================================"
echo "Setup completed successfully!"
echo "======================================"
echo ""
echo "Next steps:"
echo "1. Train the model: python src/train.py"
echo "2. Launch dashboard: streamlit run app.py"
echo ""
