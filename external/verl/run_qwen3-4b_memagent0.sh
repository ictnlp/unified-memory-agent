# run on 8xH100
# make sure your current working directory is the root of the project
set -x

ulimit -n 65535

PROJECT_DIR="$(pwd)"
CONFIG_PATH="$PROJECT_DIR/examples/sglang_multiturn/config"
export VERL_LOGGING_LEVEL=DEBUG
export EMBEDDING_SERVICE_ENDPOINT="http://localhost:8080/v1/embeddings"
export PROMPT_TEMPLATE_PATH="/mnt/pfs-guan-ssai/nlu/zhangkehao/unified-memory-agent/prompt_template.yaml"

python3 -m verl.trainer.main_ppo \
    algorithm.adv_estimator=grpo \
    data.train_batch_size=2 \
    data.max_prompt_length=8192 \
    data.max_response_length=8192 \
    data.filter_overlong_prompts=True \
    data.truncation='error' \
    data.return_raw_chat=True \
    actor_rollout_ref.model.path=Qwen/Qwen3-4B-Instruct-2507 \
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
    actor_rollout_ref.rollout.n=8 \
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
    algorithm.use_kl_in_reward=True \
    trainer.critic_warmup=0 \
    trainer.logger='["console","swanlab"]' \
    trainer.project_name='tool_memagent' \
    trainer.experiment_name='qwen3-4b_GRPO_extend_datav3_debug' \
    trainer.n_gpus_per_node=8 \
    trainer.nnodes=1 \
    trainer.save_freq=20 \
    trainer.test_freq=10 \
    trainer.val_before_train=False \
    trainer.total_epochs=3 \
    data.train_files="['/mnt/pfs-guan-ssai/nlu/zhangkehao/unified-memory-agent/data/memalphafull_train_verl.parquet', \
                    '/mnt/pfs-guan-ssai/nlu/zhangkehao/MemAgent_minimal/taskutils/memory_data/hotpotqa_train_mem_agent_loop_chunk2000.parquet', \
                    '/mnt/pfs-guan-ssai/nlu/zhangkehao/unified-memory-agent/data/locomo_train_verl.parquet', \
                    '/mnt/pfs-guan-ssai/nlu/zhangkehao/unified-memory-agent/data/synth_train_verl.parquet']" \
    data.val_files="['/mnt/pfs-guan-ssai/nlu/zhangkehao/unified-memory-agent/data/memalpha_dev_verl_small_nlu.parquet']" \
    custom_reward_function.path=/mnt/pfs-guan-ssai/nlu/zhangkehao/verl/memagent/hotpotqa.py \
    custom_reward_function.name=reward_func $@

# 在parquet数据里指定用哪个agent loop
