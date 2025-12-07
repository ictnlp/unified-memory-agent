# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

**verl** is a flexible, efficient, and production-ready reinforcement learning training library for large language models (LLMs). It's the open-source implementation of the HybridFlow framework from the EuroSys 2025 paper.

Key capabilities:
- **RL Algorithms**: PPO, GRPO, DAPO, ReMax, REINFORCE++, RLOO, PRIME, SPPO
- **Training Backends**: FSDP, FSDP2, Megatron-LM
- **Rollout Engines**: vLLM, SGLang, HF Transformers
- **Multi-modal Support**: Vision-language models (Qwen2.5-VL, Kimi-VL)
- **Multi-turn Training**: Tool calling and agent frameworks
- **Scalability**: Up to 671B models across hundreds of GPUs

## Quick Start Commands

### Installation
```bash
# Basic installation
pip install -e .

# With GPU dependencies
pip install -e ".[gpu,vllm,sglang]"

# Development setup
pip install -e ".[test]"
pre-commit install
```

### Running Examples
```bash
# PPO training with Qwen2.5-0.5B on GSM8K
bash examples/ppo_trainer/run_qwen2-7b_math_gsm8k_megatron.sh

# GRPO training with DeepSeek-7B
bash examples/grpo_trainer/run_deepseek7b_llm_math.sh

# Multi-turn training with tools
bash examples/sglang_multiturn/run_qwen2.5-3b_gsm8k_multiturn.sh
```

### Testing
```bash
# Run all CPU tests
pytest tests/ -k "on_cpu"

# Run specific test categories
pytest tests/trainer/ -v
pytest tests/special_sanity/ -v

# Run end-to-end tests
pytest tests/special_e2e/ -v

# Run single test file
pytest tests/trainer/ppo/test_core_algos_on_cpu.py -v
```

### Linting & Formatting
```bash
# Format code with ruff
ruff format .
ruff check . --fix

# Type checking (limited scope)
mypy verl/trainer/config/algorithm.py

# Pre-commit hooks
pre-commit run --all-files
```

## Architecture Overview

### Core Components

**Data Protocol**: `verl/protocol.py` defines `DataProto` for standardized data transfer between components using TensorDict.

**Single Controller**: `verl/single_controller/` provides Ray-based distributed computing abstraction with:
- Worker groups and decorators for distributed execution
- Automatic device placement and resource management
- Fault tolerance and recovery mechanisms

**Trainer Architecture**: `verl/trainer/` contains:
- `main_ppo.py`: Entry point for PPO training with Hydra configuration
- `ppo/ray_trainer.py`: Distributed PPO training orchestration
- `config/`: YAML configurations for different training setups

**Worker System**: `verl/workers/` implements modular workers:
- **Actor**: Policy network for action generation
- **Critic**: Value function estimation
- **Rollout**: Response generation (vLLM/SGLang/HF)
- **Reward Model**: Reward computation and management

### Key Design Patterns

**HybridFlow Pattern**: Decouples computation and data dependencies, enabling:
- Flexible placement of models across devices
- Seamless integration with existing LLM frameworks
- Efficient resource utilization

**3D-HybridEngine**: Optimizes memory and communication:
- Eliminates memory redundancy during training/generation transitions
- Supports expert parallelism for large models
- Enables LoRA fine-tuning with RL

**Configuration Management**: Uses Hydra for hierarchical configuration:
- `trainer/config/`: Base configurations
- `examples/*/config/`: Example-specific overrides
- Runtime parameter adjustment via command line

## Development Workflow

### Project Structure
```
verl/
├── verl/                    # Core library
│   ├── trainer/            # Training algorithms and orchestration
│   ├── workers/            # Distributed workers (actor, critic, rollout, reward)
│   ├── single_controller/   # Ray-based distributed computing
│   ├── models/             # Model implementations (HF, Megatron, custom)
│   ├── tools/              # Tool integration for multi-turn training
│   └── utils/              # Utilities and helpers
├── examples/               # Training examples and scripts
├── recipe/                 # Advanced algorithm implementations
├── tests/                  # Test suites
│   ├── special_e2e/        # End-to-end tests
│   ├── special_distributed/ # Multi-GPU tests
│   └── special_sanity/    # Quick sanity checks
└── docs/                   # Documentation
```

### Adding New Features

**New RL Algorithm**:
1. Create algorithm config in `trainer/config/`
2. Implement core algorithms in `trainer/[algorithm]/`
3. Add example scripts in `examples/[algorithm]_trainer/`

**New Model Support**:
1. Add model implementation in `verl/models/`
2. Create worker configurations in `verl/workers/config/`
3. Update model registry in `verl/models/registry.py`

**New Tools/Environments**:
1. Implement tool in `verl/tools/`
2. Add tool configuration schemas
3. Create example usage in `examples/sglang_multiturn/`

### Testing Strategy

**Test Categories**:
- **Unit Tests**: `tests/*/` - Component-level testing
- **CPU Tests**: `*on_cpu.py` - Logic validation without GPU
- **GPU Tests**: Distributed and memory-intensive tests
- **E2E Tests**: Full training pipeline validation

**Test Execution**:
```bash
# Development testing
pytest tests/special_sanity/ -v

# GPU testing (requires multi-GPU)
pytest tests/special_distributed/ -v

# Algorithm testing
pytest tests/trainer/ -v
```

### Configuration Patterns

**Hydra Configuration**:
- Base configs: `trainer/config/*.yaml`
- Model configs: `trainer/config/model/*.yaml`
- Runtime overrides via command line:
  ```bash
  python -m verl.trainer.main_ppo \
    trainer.n_gpus_per_node=8 \
    actor_rollout_ref.model.path=Qwen/Qwen2.5-7B-Instruct
  ```

**Environment Variables**:
- `VERL_USE_MODELSCOPE=true`: Use ModelScope instead of HuggingFace
- `CUDA_DEVICE_MAX_CONNECTIONS=1`: Optimize Megatron communication
- `RAY_DISABLE_IMPORT_WARNING=1`: Suppress Ray warnings

## Common Issues & Solutions

**Memory Issues**:
- Reduce `actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu`
- Enable gradient checkpointing: `actor_rollout_ref.actor.gradient_checkpointing=True`
- Use LoRA: `actor_rollout_ref.actor.use_lora=True`

**Performance Tuning**:
- Increase `actor_rollout_ref.rollout.tensor_model_parallel_size` for larger models
- Use `actor_rollout_ref.rollout.name=sglang` for better throughput
- Enable sequence packing: `data.use_packed_sequence=True`

**Debugging**:
- Set `trainer.logger=["console","wandb"]` for detailed logs
- Use `trainer.test_freq=1` for frequent validation
- Enable debug mode: `export VERL_DEBUG=1`