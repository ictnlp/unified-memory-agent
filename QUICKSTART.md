# Quick Start Guide

## One-Click Setup (Recommended)

Run our automated setup script:

```bash
# Clone the repository
git clone https://github.com/yourusername/unified-memory-agent.git
cd unified-memory-agent

# Run the setup script
./test_setup.sh
```

The script will:
- ✅ Check/install `uv` package manager
- ✅ Create and configure all required environments
- ✅ Install dependencies for main project, infinity, and VERL
- ✅ Run basic tests to ensure everything works
- ✅ Create a `.env` file with dummy values

## Manual Setup

If you prefer manual setup, see [SETUP.md](SETUP.md) for detailed instructions.

## Quick Test

After setup, test the installation:

```bash
# Help command
python evaluate_async.py --help

# Available tasks
python evaluate_async.py --list-tasks

# Available agents
python evaluate_async.py --list-agents
```

## Common Issues

### 1. uv not found
```bash
curl -LsSf https://astral.sh/uv/install.sh | sh
source ~/.cargo/env
```

### 2. Infinity installation fails
Infinity requires Python 3.8+. Ensure your system Python is compatible:
```bash
python --version  # Should be 3.8 or higher
```

### 3. VERL installation issues
VERL might need additional system dependencies on some platforms:
```bash
# Ubuntu/Debian
sudo apt-get install build-essential

# macOS
xcode-select --install
```

### 4. Environment variable errors
Copy and configure your `.env` file:
```bash
cp .env.example .env
# Edit .env with your actual API keys
```

## Next Steps

1. Configure your API keys in `.env`
2. Download test datasets (see Datasets section in README.md)
3. Run your first evaluation:
   ```bash
   python evaluate_async.py --task locomo --agent concat
   ```

## Need Help?

- 📖 Check [README.md](README.md) for full documentation
- 🐛 Report issues on GitHub
- 💬 Start a Discussion for questions