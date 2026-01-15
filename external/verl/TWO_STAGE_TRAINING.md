# Two-Stage Training for Memory Agent

This guide explains how to use the two-stage training system for memory agents.

## Overview

The two-stage training approach separates memory formation and QA into two sequential training phases:

- **Stage 1**: Train memory formation model (fix QA model)
- **Stage 2**: Train QA model (fix memory formation model)

This approach allows focused optimization of each component.

## Architecture

### Key Components

1. **agent_loop.py** (Modified)
   - Added `agent_name_override` config to override dataset's agent_name
   - Location: `verl/experimental/agent_loop/agent_loop.py:442-448`

2. **tool_mem_agent_loop_twostage.py** (New)
   - Implements two-stage training logic with built-in FixedModelServerManager
   - Registered as `tool_mem_agent_twostage`
   - Location: `verl/experimental/agent_loop/tool_mem_agent_loop_twostage.py`

3. **Training Scripts** (New)
   - Stage 1: `run_qwen3-4b_memagent_stage1.sh`
   - Stage 2: `run_qwen3-4b_memagent_stage2.sh`

## Training Workflow

### Prerequisites

- 8x H100 GPUs (or adjust `trainer.n_gpus_per_node`)
- vLLM installed for serving fixed models
- Sufficient GPU memory for training + serving

### Step 1: Deploy Fixed QA Model

Before running Stage 1, deploy a QA model as the fixed model:

```bash
# Option A: Use base model
vllm serve Qwen/Qwen3-4B-Instruct \
  --port 8000 \
  --tensor-parallel-size 1 \
  --gpu-memory-utilization 0.9 \
  --max-model-len 16384

# Option B: Use a pre-trained QA model if available
# vllm serve /path/to/pretrained_qa_model --port 8000 ...
```

Verify the endpoint is accessible:
```bash
curl http://localhost:8000/v1/models
```

### Step 2: Run Stage 1 Training (Memory Formation)

```bash
cd /mnt/pfs-guan-ssai/nlu/zhangkehao/unified-memory-agent/external/verl
bash run_qwen3-4b_memagent_stage1.sh
```

**What happens in Stage 1:**
- Memory formation: Uses trainable model (computes gradients)
- QA generation: Calls fixed model at localhost:8000 (no gradients)
- Only conversations with `is_final=0` participate in training
- Checkpoint saved to output directory (check trainer logs for path)

**Monitor training:**
- Check console logs for "Using TRAINABLE/FIXED model" messages
- Verify `is_trainable` flag in trajectory outputs
- SwanLab dashboard: project `tool_memagent_twostage`, experiment `qwen3-4b_stage1_memory`

### Step 3: Deploy Stage 1 Checkpoint as Fixed Model

After Stage 1 completes, deploy the trained memory model:

```bash
# Find the checkpoint path from Stage 1 logs, usually something like:
# /path/to/checkpoints/qwen3-4b_stage1_memory/global_step_XXX

vllm serve /path/to/stage1_checkpoint \
  --port 8001 \
  --tensor-parallel-size 1 \
  --gpu-memory-utilization 0.9 \
  --max-model-len 16384
```

Verify:
```bash
curl http://localhost:8001/v1/models
```

### Step 4: Run Stage 2 Training (QA)

Update `run_qwen3-4b_memagent_stage2.sh` if needed:
- Set `FIXED_MODEL_API` to your Stage 1 model endpoint (default: localhost:8001)
- Optionally adjust `STAGE2_INIT_MODEL` to start from Stage 1 checkpoint

```bash
bash run_qwen3-4b_memagent_stage2.sh
```

**What happens in Stage 2:**
- Memory formation: Calls fixed model at localhost:8001 (no gradients)
- QA generation: Uses trainable model (computes gradients)
- Only conversations with `is_final>0` participate in training
- Final checkpoint is the complete two-stage trained model

**Monitor training:**
- Check console logs for model routing
- SwanLab dashboard: experiment `qwen3-4b_stage2_qa`

## Configuration Reference

### Key Parameters

```yaml
# In training scripts:
actor_rollout_ref:
  rollout:
    agent_name_override: tool_mem_agent_twostage  # Override dataset's agent_name
    training_stage: memory  # or "qa"
    fixed_model_api: http://localhost:8000/v1  # Fixed model endpoint
```

### Training Stages

| Stage | Trains | Fixed | Trainable Conversations |
|-------|--------|-------|------------------------|
| Stage 1 | Memory formation | QA model | `is_final=0` (memory chunks) |
| Stage 2 | QA answering | Memory model | `is_final>0` (final responses) |

## Troubleshooting

### Issue: "agent_name tool_mem_agent_twostage not registered"

**Solution**: Make sure the new agent loop file is imported. Add to `verl/experimental/agent_loop/__init__.py`:
```python
from .tool_mem_agent_loop_twostage import TwoStageMemoryAgentLoop
```

### Issue: "fixed_model_api is required"

**Solution**: Ensure you pass `actor_rollout_ref.rollout.fixed_model_api=...` in training script.

### Issue: Connection error to fixed model API

**Solution**:
1. Verify vLLM server is running: `curl http://localhost:8000/v1/models`
2. Check firewall/network settings
3. Review vLLM server logs for errors

### Issue: "Negative increment detected" in logs

This is a debug message for tool parsing. If training continues normally, it's safe to ignore.

### Issue: Out of memory

**Solutions**:
- Reduce `data.train_batch_size` (default: 32 → 16)
- Reduce `actor_rollout_ref.rollout.n` (sampling size, default: 16 → 8)
- Enable offloading: `actor_rollout_ref.actor.fsdp_config.optimizer_offload=True`
- Reduce model serving: adjust vLLM's `--gpu-memory-utilization`

## Advanced Usage

### Resume from Checkpoint

To resume Stage 1 training:
```bash
bash run_qwen3-4b_memagent_stage1.sh \
  trainer.load_checkpoint=/path/to/checkpoint
```

### Adjust Learning Rate

Different stages may need different learning rates:
```bash
# Stage 1: Memory formation (more complex)
actor_rollout_ref.actor.optim.lr=1e-6

# Stage 2: QA (potentially higher LR)
actor_rollout_ref.actor.optim.lr=5e-6
```

### Three-Stage Training

You can extend to three stages (e.g., alternating):
1. Stage 1: Train memory (fix QA)
2. Stage 2: Train QA (fix memory from Stage 1)
3. Stage 3: Train memory again (fix QA from Stage 2)

Just repeat the process with updated checkpoints.

## Comparison with End-to-End Training

| Aspect | End-to-End | Two-Stage |
|--------|-----------|-----------|
| Training | Single model, both tasks | Separate optimization |
| Complexity | Simpler setup | Requires vLLM deployment |
| GPU Usage | Training only | Training + serving |
| Convergence | May conflict | Focused per stage |
| Flexibility | Less control | Stage-specific tuning |

## Files Modified/Created

### Modified
- `verl/experimental/agent_loop/agent_loop.py` (+7 lines)

### Created
- `verl/experimental/agent_loop/tool_mem_agent_loop_twostage.py` (550+ lines, includes FixedModelServerManager)
- `run_qwen3-4b_memagent_stage1.sh` (92 lines)
- `run_qwen3-4b_memagent_stage2.sh` (104 lines)

### Dataset
- No changes required! The `agent_name_override` parameter handles everything.

## Next Steps

1. ✅ Deploy fixed QA model (vLLM)
2. ✅ Run Stage 1 training
3. ✅ Deploy Stage 1 checkpoint as fixed memory model
4. ✅ Run Stage 2 training
5. 🎯 Evaluate final model
6. 🔄 (Optional) Iterate with Stage 3+

## Questions?

Check the code comments in:
- `tool_mem_agent_loop_twostage.py` - Core implementation (includes FixedModelServerManager)
- Training scripts - Configuration examples
