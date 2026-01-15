#!/bin/bash
# Stage 2: Train QA Model (Fix Memory Formation Model)
# This script trains the QA part while using a fixed memory formation model
set -x

ulimit -n 65535

PROJECT_DIR="$(pwd)"
CONFIG_PATH="$PROJECT_DIR/examples/sglang_multiturn/config"
export VERL_LOGGING_LEVEL=DEBUG
export EMBEDDING_SERVICE_ENDPOINT="http://localhost:8080/embeddings"
export PROMPT_TEMPLATE_PATH="/mnt/pfs-guan-ssai/nlu/zhangkehao/unified-memory-agent/prompt_template.yaml"

# ========================================
# Stage 2 Configuration
# ========================================
# Before running this script:
# 1. Take the checkpoint from Stage 1 training
# 2. Deploy the Stage 1 checkpoint as fixed memory model using vLLM:
#    vllm serve /path/to/stage1_checkpoint \
#      --port 8001 \
#      --tensor-parallel-size 1 \
#      --gpu-memory-utilization 0.9
#
# 3. Make sure the endpoint is accessible at http://localhost:8001/v1
# 4. Update FIXED_MODEL_API below to point to your deployed model
# ========================================

# Fixed memory formation model endpoint (Stage 1 checkpoint via vLLM)
FIXED_MODEL_API="http://localhost:8001/v1"

# Stage 2 starts with the original base model (or optionally the Stage 1 checkpoint)
# If you want to start from Stage 1 checkpoint for QA training, change this path
STAGE2_INIT_MODEL="Qwen/Qwen3-4B-Instruct-2507"
# STAGE2_INIT_MODEL="/path/to/stage1_checkpoint"  # Alternative: continue from Stage 1

python3 -m verl.trainer.main_ppo \
    +ray_kwargs.ray_init.runtime_env.env_vars.HF_HUB_OFFLINE=\"1\" \
    +ray_kwargs.ray_init.runtime_env.env_vars.PROMPT_TEMPLATE_PATH=\"/lpai/volumes/base-agentos-lx-my/zhangkehao/unified-memory-agent/prompt_template.yaml\" \
    +ray_kwargs.ray_init.runtime_env.env_vars.EMBEDDING_SERVICE_ENDPOINT=\"http://localhost:8080/embeddings\" \
    algorithm.adv_estimator=grpo \
    data.train_batch_size=32 \
    data.max_prompt_length=8192 \
    data.max_response_length=8192 \
    data.filter_overlong_prompts=True \
    data.truncation='error' \
    data.return_raw_chat=True \
    actor_rollout_ref.model.path=$STAGE2_INIT_MODEL \
    actor_rollout_ref.actor.optim.lr=1e-6 \
    actor_rollout_ref.model.use_remove_padding=True \
    actor_rollout_ref.actor.use_dynamic_bsz=True \
    actor_rollout_ref.actor.ppo_mini_batch_size=8 \
    actor_rollout_ref.actor.ppo_max_token_len_per_gpu=16384 \
    actor_rollout_ref.actor.use_kl_loss=True \
    actor_rollout_ref.actor.kl_loss_coef=0.001 \
    actor_rollout_ref.actor.kl_loss_type=low_var_kl \
    actor_rollout_ref.actor.entropy_coeff=0 \
    actor_rollout_ref.model.enable_gradient_checkpointing=True \
    actor_rollout_ref.actor.fsdp_config.param_offload=False \
    actor_rollout_ref.actor.fsdp_config.optimizer_offload=False \
    actor_rollout_ref.rollout.log_prob_max_token_len_per_gpu=32768 \
    actor_rollout_ref.rollout.tensor_model_parallel_size=1 \
    actor_rollout_ref.rollout.name=sglang \
    actor_rollout_ref.rollout.gpu_memory_utilization=0.85 \
    actor_rollout_ref.rollout.n=16 \
    actor_rollout_ref.ref.log_prob_max_token_len_per_gpu=32768 \
    actor_rollout_ref.ref.fsdp_config.param_offload=True \
    actor_rollout_ref.rollout.over_sample_rate=0.1 \
    actor_rollout_ref.rollout.mode=async \
    actor_rollout_ref.rollout.multi_turn.tool_config_path=/mnt/pfs-guan-ssai/nlu/zhangkehao/unified-memory-agent/external/verl/memagent/tool_config.yaml \
    actor_rollout_ref.rollout.multi_turn.max_user_turns=10 \
    actor_rollout_ref.rollout.multi_turn.max_assistant_turns=10 \
    actor_rollout_ref.rollout.multi_turn.max_parallel_calls=10 \
    actor_rollout_ref.rollout.multi_turn.max_tool_response_length=3000 \
    actor_rollout_ref.rollout.multi_turn.tool_response_truncate_side=left \
    actor_rollout_ref.rollout.multi_turn.format=hermes \
    actor_rollout_ref.rollout.agent_name_override=tool_mem_agent_twostage \
    actor_rollout_ref.rollout.training_stage=qa \
    actor_rollout_ref.rollout.fixed_model_api=$FIXED_MODEL_API \
    algorithm.use_kl_in_reward=True \
    algorithm.norm_adv_by_std_in_grpo=True \
    trainer.critic_warmup=0 \
    trainer.logger='["console","swanlab"]' \
    trainer.project_name='tool_memagent_twostage' \
    trainer.experiment_name='qwen3-4b_stage2_qa' \
    trainer.n_gpus_per_node=8 \
    trainer.nnodes=1 \
    trainer.save_freq=20 \
    trainer.test_freq=10 \
    trainer.val_before_train=False \
    trainer.total_epochs=1 \
    data.train_files="['/mnt/pfs-guan-ssai/nlu/zhangkehao/unified-memory-agent/data/memalphafull_train_verl.parquet', \
                        '/mnt/pfs-guan-ssai/nlu/zhangkehao/MemAgent_minimal/taskutils/memory_data/hotpotqa_train_mem_agent_loop_chunk2000.parquet', \
                        '/mnt/pfs-guan-ssai/nlu/zhangkehao/unified-memory-agent/data/synth_train_verl.parquet']" \
    data.val_files="['/mnt/pfs-guan-ssai/nlu/zhangkehao/MemAgent_minimal/taskutils/memory_data/hotpotqa_dev_mem_agent_loop_chunk2000.parquet', \
                        '/mnt/pfs-guan-ssai/nlu/zhangkehao/unified-memory-agent/data/memalpha_dev_verl_small.parquet']" \
    custom_reward_function.path=/mnt/pfs-guan-ssai/nlu/zhangkehao/unified-memory-agent/external/verl/memagent/hotpotqa.py \
    custom_reward_function.name=reward_func $@

# ========================================
# Stage 2 Training Notes:
# ========================================
# - This stage trains the QA MODEL
# - The memory formation model is FIXED (called via vLLM API at localhost:8001)
# - Memory conversations (is_final=0) are generated by the fixed model
# - Only final QA responses (is_final>0) will have gradients computed
# - Final checkpoint will be the complete two-stage trained model
# ========================================
