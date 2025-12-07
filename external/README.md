# External Dependencies Migration Guide

This document explains the external dependencies structure and how to set up the project on a new machine.

## Directory Structure

```
external/
├── gam/                  # GAM Framework (精简版)
├── memalpha/             # Mem-alpha Core (精简版)
├── mem1/                 # MEM1 Core (精简版)
├── verl/                 # verl Complete (含虚拟环境)
└── *_SOURCE.txt          # 源代码追溯信息
```

## Setup on New Machine

### 1. Clone/Copy Project

Copy the entire `unified-memory-agent` directory including `external/`.

### 2. Data Files Setup

The following data files are **NOT included** in the repository (too large):

- `Mem-alpha/data/memalpha/test.parquet`
- `Mem-alpha/data/memoryagentbench/test.parquet`

**Option A: Use environment variables** (Recommended)

```bash
export MEMALPHA_PARQUET_PATH="/path/to/your/memalpha/test.parquet"
export MEMORYAGENTBENCH_PARQUET_PATH="/path/to/your/memoryagentbench/test.parquet"
```

**Option B: Copy data files**

Copy data files from original Mem-alpha location:
```bash
mkdir -p data/memalpha data/memoryagentbench
cp /original/path/Mem-alpha/data/memalpha/test.parquet data/memalpha/
cp /original/path/Mem-alpha/data/memoryagentbench/test.parquet data/memoryagentbench/
```

Then set environment variables to point to local copies:
```bash
export MEMALPHA_PARQUET_PATH="./data/memalpha/test.parquet"
export MEMORYAGENTBENCH_PARQUET_PATH="./data/memoryagentbench/test.parquet"
```

### 3. verl Training (Optional)

If you need to train with verl:

```bash
# Activate verl virtual environment
source external/verl/.venv/bin/activate

# verl checkpoints (637GB) are NOT included
# If needed, copy from: /mnt/pfs-guan-ssai/nlu/zhangkehao/verl/checkpoints/
```

### 4. Dependencies

**For evaluation only:**
```bash
pip install -r requirements.txt
```

**For GAM agent:**
```bash
pip install -r external/gam_requirements.txt
```

**For Mem-alpha agent:**
```bash
pip install -r external/memalpha_requirements.txt
```

**For verl training:**
```bash
source external/verl/.venv/bin/activate  # Already has all dependencies
```

## Running Evaluation

```bash
# Example: evaluate concat agent on locomo task
python evaluate_async.py --task locomo --agent concat

# Example: evaluate memalpha agent
python evaluate_async.py --task locomo --agent memalpha

# Example: evaluate gam agent
python evaluate_async.py --task locomo --agent gam
```

## Path Configuration Summary

| Component | Path | Configurable |
|-----------|------|--------------|
| GAM code | `external/gam/` | ✓ (via `agents/__init__.py`) |
| Mem-alpha code | `external/memalpha/` | ✓ (via `agents/__init__.py`) |
| MEM1 code | `external/mem1/` | ✓ (via `agents/__init__.py`) |
| verl code | `external/verl/` | ✓ (via `agents/__init__.py`) |
| Mem-alpha configs | `external/memalpha/config/` | ✓ (via `--agent_config_path`) |
| Data files | See above | ✓ (via environment variables) |
| verl checkpoints | **NOT included** | Copy separately if needed |

## Notes

- All Python import paths have been updated to use `external/`
- verl virtual environment paths have been fixed for new location
- Original source information is documented in `*_SOURCE.txt` files
- Git tracking: `external/verl/.venv/` is excluded via `.gitignore`
