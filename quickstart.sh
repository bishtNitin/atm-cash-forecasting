#!/bin/bash
# Quick Start Script for ATM Cash Forecasting System

echo "=================================================="
echo "ATM Cash Forecasting System - Quick Start"
echo "=================================================="
echo ""

# Check if virtual environment exists
if [ ! -d "venv" ]; then
    echo "Creating virtual environment..."
    python3 -m venv venv
fi

# Activate virtual environment
echo "Activating virtual environment..."
source venv/bin/activate

# Install dependencies
echo "Installing dependencies..."
pip install -q pandas numpy pyyaml holidays scipy scikit-learn lightgbm streamlit plotly matplotlib seaborn

# Generate sample data if it doesn't exist
if [ ! -f "data/raw/synthetic_atm_data.csv" ]; then
    echo ""
    echo "Generating sample data..."
    python -c "from src.utils.data_generator import generate_synthetic_data; generate_synthetic_data(n_atms=100)"
fi

# Run training
echo ""
echo "Running training pipeline..."
python train.py

# Launch Streamlit
echo ""
echo "=================================================="
echo "Launching Streamlit UI..."
echo "=================================================="
echo ""
streamlit run app.py
