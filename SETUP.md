# Unified Memory Agent Evaluation

## Setup Instructions

This project requires setting up three separate environments:

### 1. Main Project Environment

```bash
cd /mnt/pfs-guan-ssai/nlu/zhangkehao/unified-memory-agent

# Create and activate virtual environment
uv venv
source .venv/bin/activate  # On Windows: .venv\Scripts\activate

# Install dependencies
uv pip install -r requirements.txt
# Or install with optional development dependencies
uv pip install -e ".[dev,all]"

# Install the project itself
uv pip install -e .
```

### 2. Infinity Environment (Isolated)

```bash
cd /mnt/pfs-guan-ssai/nlu/zhangkehao/unified-memory-agent/external/infinity/libs/infinity_emb

# Create and activate virtual environment
uv venv
source .venv/bin/activate  # On Windows: .venv\Scripts\activate

# Install in development mode with all extras
uv pip install -e ".[all]"
```

### 3. VERL Environment (Can use main project's virtual env)

```bash
# Ensure you're in the main project's virtual environment
cd /mnt/pfs-guan-ssai/nlu/zhangkehao/unified-memory-agent/external/verl

# Install in development mode
uv pip install -e .
```

## Environment Variables

Copy `.env.example` to `.env` and configure:

```bash
cp .env.example .env
# Edit .env with your API keys and endpoints
```

## Quick Start

After setting up all environments:

```bash
# Run evaluation
python evaluate_async.py --task locomo --agent concat

# Generate statistics
python generate_stats.py --task all --save_txt
```

## Development

For development with all tools:

```bash
# Install development dependencies
uv pip install -e ".[dev]"

# Run tests
pytest

# Format code
black .
isort .

# Lint
flake8 .

# Type check
mypy .
```