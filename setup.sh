#!/bin/bash

# Setup script for Hybrid Diabetes Diagnosis System
# This script sets up the environment and installs dependencies

set -e  # Exit on error

echo "=================================================="
echo "Hybrid Diabetes Diagnosis System - Setup"
echo "=================================================="
echo

# Check Python version
echo "Checking Python version..."
python_version=$(python3 --version 2>&1 | awk '{print $2}')
echo "Found Python $python_version"

# Check if Python is at least 3.8
required_version="3.8"
if ! python3 -c "import sys; exit(0 if sys.version_info >= (3, 8) else 1)"; then
    echo "ERROR: Python 3.8 or higher is required"
    exit 1
fi
echo "✓ Python version OK"
echo

# Create virtual environment
echo "Creating virtual environment..."
if [ ! -d "venv" ]; then
    python3 -m venv venv
    echo "✓ Virtual environment created"
else
    echo "✓ Virtual environment already exists"
fi
echo

# Activate virtual environment
echo "Activating virtual environment..."
source venv/bin/activate
echo "✓ Virtual environment activated"
echo

# Upgrade pip
echo "Upgrading pip..."
pip install --upgrade pip --quiet
echo "✓ pip upgraded"
echo

# Install dependencies
echo "Installing dependencies..."
pip install -r requirements.txt --quiet
echo "✓ Dependencies installed"
echo

# Create .env file if it doesn't exist
if [ ! -f ".env" ]; then
    echo "Creating .env file from template..."
    cp .env.example .env
    echo "✓ .env file created"
    echo
    echo "⚠️  IMPORTANT: Edit .env and add your GROQ_API_KEY"
    echo "   Get a free API key at: https://console.groq.com"
else
    echo "✓ .env file already exists"
fi
echo

# Create output directory if it doesn't exist
mkdir -p output
echo "✓ Output directory ready"
echo

# Make run script executable
chmod +x run_hybrid.py
echo "✓ run_hybrid.py is now executable"
echo

echo "=================================================="
echo "Setup Complete!"
echo "=================================================="
echo
echo "Next steps:"
echo "  1. Add your GROQ_API_KEY to .env file (optional but recommended)"
echo "  2. Run the system: python run_hybrid.py"
echo "  3. Or use: ./run_hybrid.py"
echo
echo "For help: python run_hybrid.py --help"
echo
echo "To activate the virtual environment in future sessions:"
echo "  source venv/bin/activate"
echo
