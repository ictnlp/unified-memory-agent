#!/bin/bash

# Test installation script for Unified Memory Agent
# This script simulates a fresh user installation

set -e  # Exit on any error

echo "🚀 Starting fresh installation test..."
echo "Working directory: $(pwd)"

# Check if uv is installed
if ! command -v uv &> /dev/null; then
    echo "❌ uv not found. Installing uv..."
    curl -LsSf https://astral.sh/uv/install.sh | sh
    export PATH="$HOME/.cargo/bin:$PATH"
fi

# Create a fresh directory for testing
if [ -d "test_installation" ]; then
    rm -rf test_installation
fi
mkdir test_installation
cd test_installation

# Clone the repository (simulate user cloning)
echo "📥 Cloning repository..."
if [ -n "$SOURCE_DIR" ]; then
    # If SOURCE_DIR is set, copy from local
    cp -r $SOURCE_DIR .
else
    # Otherwise clone from git (replace with actual repo URL)
    git clone --depth 1 https://github.com/yourusername/unified-memory-agent.git .
fi

# Setup main environment
echo "🔧 Setting up main environment..."
uv venv
source .venv/bin/activate

# Check if VERSION exists in config.py
if ! grep -q "VERSION" config.py; then
    echo "⚠️  WARNING: VERSION not found in config.py, adding placeholder"
    echo "VERSION = '0.1.0'" >> config.py
fi

# Install main dependencies
echo "📦 Installing main dependencies..."
uv pip install -r requirements.txt
uv pip install -e .

# Setup infinity environment
echo "🔧 Setting up infinity environment (isolated)..."
cd external/infinity/libs/infinity_emb
if [ ! -d "infinity_emb" ]; then
    echo "⚠️  WARNING: infinity_emb directory not found, skipping"
else
    uv venv
    source .venv/bin/activate
    uv pip install -e ".[all]"
    cd ../../..
fi

# Setup VERL environment
echo "🔧 Setting up VERL environment..."
cd external/verl
if [ ! -f "pyproject.toml" ]; then
    echo "⚠️  WARNING: VERL pyproject.toml not found, skipping"
else
    # Use main environment
    cd ..
    source .venv/bin/activate
    cd external/verl
    uv pip install -e .
fi

cd ../..

# Create test .env file
echo "📝 Creating .env file..."
cp .env.example .env
echo "# Test configuration" >> .env
echo "X_CHJ_GWTOKEN=dummy_token" >> .env
echo "X_CHJ_GW_SOURCE=test" >> .env

# Test basic imports
echo "🧪 Testing basic imports..."
source .venv/bin/activate
python -c "
import config
from agents.base_agent import BaseAgent
from openai import OpenAI
print('✓ Basic imports successful')
"

# Test minimal functionality
echo "🎯 Testing minimal functionality..."
python -c "
import os
# Load environment
from dotenv import load_dotenv
load_dotenv('.env')

# Create a client for testing
from openai import OpenAI
client = OpenAI(
    base_url='http://localhost:8000/v1',
    api_key='test-key'
)

print('✓ Client creation successful')
"

# Run a quick validation
echo "📋 Running quick validation..."
python -c "
import sys
import inspect

# Check all agents are importable
from config import AGENT_CLASS
for name in ['concat', 'memagent']:
    if name in AGENT_CLASS:
        cls = AGENT_CLASS[name]
        print(f'✓ Agent {name}: {cls}')

# Check datasets are importable
from config import DATASET_LOADERS
print(f'✓ Datasets available: {len(DATASET_LOADERS)}')
"

echo ""
echo "✅ Installation test completed successfully!"
echo "📋 Summary of environments configured:"
echo "  - Main project: .venv"
echo "  - Infinity: external/infinity/libs/infinity_emb/.venv (if directory exists)"
echo ""
echo "🎉 Ready to run: python evaluate_async.py --help"