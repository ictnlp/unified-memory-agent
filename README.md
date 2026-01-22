# Unified Memory Agent

A comprehensive evaluation framework for testing memory-based conversational AI agents across multiple datasets and metrics.

## Overview

This system evaluates how well AI agents can answer questions based on accumulated conversational memory over time. It supports 14 benchmark datasets.

## Installation

This project requires setting up two separate Python environments to handle different dependencies.

### Prerequisites

- Python 3.8+
- [uv](https://docs.astral.sh/uv/) package installer (recommended for faster installs)

### Environment 1: Main Project Environment

```bash
# Clone the repository
git clone https://gitlab.chehejia.com/zhangkehao/unified-memory-agent.git
cd unified-memory-agent

# Create virtual environment
uv venv
source .venv/bin/activate  # On Windows: .venv\Scripts\activate

# Install main dependencies
uv pip install -r requirements.txt

# Install VERL in development mode
cd external/verl
uv pip install -e .
cd ../..
```

### Environment 2: Infinity_emb Environment

```bash
# Navigate to infinity_emb directory
cd external/infinity/libs/infinity_emb

# Create separate virtual environment
uv venv
source .venv/bin/activate  # On Windows: .venv\Scripts\activate

# Install infinity_emb with all optional dependencies
uv pip install -e ".[all]"

# Return to project root
cd ../../../..
```

## Quick Start
Deploy
```bash
source .venv/bin/activate
vllm serve $MODEL -dp 2 -tp 4 --gpu-memory-utilization 0.8 --enforce-eager > vllm.log 2>&1 &
source ./external/infinity/.venv/bin/activate
infinity_emb v2 --model-id sentence-transformers/all-MiniLM-L6-v2 --port 8080 > infinity_emb.log 2>&1 &
# Wait for server
until curl -s http://localhost:8080/health > /dev/null 2>&1; do
    sleep 2
    echo "wait for server port 8080..."
done
until curl -s http://localhost:8000/health > /dev/null 2>&1; do
    sleep 2
    echo "wait for server port 8000..."
done
```

Run Demo
```bash
python test_verl_agent.py
```

## Test on Benchmarks

### Download Evaluation Datasets
Follow [Memalpha](https://github.com/wangyu-ustc/Mem-alpha/tree/main), download the memalpha and memagentbench dataset.
```bash
mkdir -p ./data/raw/
# Download Memalpha training/test dataset
bash hfd.sh YuWangX/Memalpha --dataset --tool aria2c -x 10
mv Memalpha ./data/raw/memalpha

# Download MemoryAgentBench evaluation dataset (processed version for this project)
bash hfd.sh YuWangX/Memalpha-Memoryagentbench --dataset --tool aria2c -x 10
mv Memalpha ./data/raw/memoryagentbench
```
Follow [Memagent](https://github.com/BytedTsinghua-SIA/MemAgent), download the hotpotqa dataset.
```bash
bash hfd.sh BytedTsinghua-SIA/hotpotqa --dataset --tool aria2c -x 10
mv hotpotqa ./data/raw/
```
Download the convomem dataset
```bash
bash hfd.sh Salesforce/ConvoMem --dataset --tool aria2c -x 10
mv ConvoMem ./data/raw/
```

### Prepare Ledger-QA

Set your api url in .env and run
```bash
python data/EvalDataset.py
```

Optionally, you can also deploy `Qwen/Qwen3-30B-A3B-Instruct-2507`, correspondingly modify `./data/synthv1_async.py`, and then run `data/EvalDataset.py` to generate Ledger-QA.

### Running UMA

```bash
# Run full evaluation (generation + scoring)
bash run_evaluation.sh
```

### Running Baselines

```bash
# Run concat baseline evaluation
bash baselines/run_concat.sh
# After finish generation
bash run_score.sh
```

### Supported Tasks & Agents

- **Tasks**: `convomem`, `locomo`, `longmemeval`, `memalpha`, `hotpotqa`, `msc`, `squad`, `banking77`, `clinic`, `nlu`, `perltqa`, `pubmed_rct`, `trec_coarse`, `trec_fine`
- **Agents**:`concat`, `memagent`， `memagent_woq`，`mem1`， `memalpha`，`toolmem`

### Output Files

#### Generation Phase
- **responses_*.jsonl**: Real-time JSONL output with questions and responses
- **Location**: `results/{task}/`

#### Evaluation Phase
- **evaluated_*.jsonl**: Complete evaluation results with all metrics
- **Statistics**: Pretty-printed tables (console or text file)

## Training

### Prepare Datasets
1. Memalpha-full
```bash
cd data
python construct_memalpha_verl_dataset.py
python construct_memalpha_verl_dataset.py --val
```

2. hotpotqa
Follow [Memagent](https://github.com/BytedTsinghua-SIA/MemAgent), we construct a hotpotqa containing 8192 items. We further chunk the context of each item for retrieving. You can download the processed data by
```bash
bash hfd.sh dp66/hotpotqa-uma --dataset --tool aria2c -x 10
mv hotpotqa-uma ./data/train
```

3. Ledger-QA
```bash
bash hfd.sh dp66/ledger-qa-train --dataset --tool aria2c -x 10
mv ledger-qa-train ./data/train
```
Optionally, you can also construct ledger-qa by the following code after running `python data/EvalDataset.py`

```bash
cd data
python construct_processed_verl_dataset.py processed_synth-ss10_train.json ./data/train/ledger-qa-train/train.parquet --dataset-name synth --batch-size 40
```

Similarly, you can construct the validation dataset

```bash
cd data
python construct_processed_verl_dataset.py processed_synth-ss10.json ./data/train/ledger-qa-train/dev.parquet --dataset-name synth --batch-size 40
```

### Deploy embedding model

```bash
source ./external/infinity/.venv/bin/activate
infinity_emb v2 --model-id sentence-transformers/all-MiniLM-L6-v2 --port 8080 > infinity_emb.log 2>&1 &
# Wait for server
until curl -s http://localhost:8080/health > /dev/null 2>&1; do
    sleep 2
    echo "wait for server port 8080..."
done
```

### Start Training
Single node
```bash
source .venv/bin/activate
cd external/verl
bash run_1node.sh
```

4 nodes
```bash
source .venv/bin/activate
cd external/verl
bash run_4nodes.sh
```
## Extending the System

### Adding New Agents

1. Create new agent class inheriting from `BaseAgent`
2. Implement required methods: `add_memory_async()` and `QA_batch_async()`
3. Add to agent factory in `evaluate_async.py`

Example:
```python
class CustomAgent(BaseAgent):
    async def add_memory_async(self, chunk: str) -> None:
        # Your implementation
        pass
    
    async def QA_batch_async(self, question: str) -> str:
        # Your implementation
        pass
```
