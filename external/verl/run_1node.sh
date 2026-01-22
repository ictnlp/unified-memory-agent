# run on 8xH100
# make sure your current working directory is the root of the project
set -x
ulimit -n 65535

PROJECT_DIR="$(pwd)"
CONFIG_PATH="$PROJECT_DIR/examples/sglang_multiturn/config"
export VERL_LOGGING_LEVEL=DEBUG
export EMBEDDING_SERVICE_ENDPOINT="http://localhost:8080/embeddings"
export PROMPT_TEMPLATE_PATH="../../prompt_template.yaml"

python3 -m verl.trainer.main_ppo \
    +ray_kwargs.ray_init.runtime_env.env_vars.HF_HUB_OFFLINE=\"1\" \
    algorithm.adv_estimator=grpo \
    data.train_batch_size=32 \
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
    actor_rollout_ref.rollout.n=16 \
    actor_rollout_ref.ref.log_prob_max_token_len_per_gpu=32768 \
    actor_rollout_ref.ref.fsdp_config.param_offload=True \
    actor_rollout_ref.rollout.over_sample_rate=0.1 \
    actor_rollout_ref.rollout.mode=async \
    actor_rollout_ref.rollout.multi_turn.tool_config_path=./memagent/tool_config.yaml \
    actor_rollout_ref.rollout.multi_turn.max_user_turns=10 \
    actor_rollout_ref.rollout.multi_turn.max_assistant_turns=10 \
    actor_rollout_ref.rollout.multi_turn.max_parallel_calls=10 \
    actor_rollout_ref.rollout.multi_turn.max_tool_response_length=3000 \
    actor_rollout_ref.rollout.multi_turn.tool_response_truncate_side=left \
    actor_rollout_ref.rollout.multi_turn.format=hermes \
    algorithm.use_kl_in_reward=True \
    algorithm.norm_adv_by_std_in_grpo=True \
    trainer.critic_warmup=0 \
    trainer.logger='["console","swanlab"]' \
    trainer.project_name='my-project' \
    trainer.experiment_name='my-experiment' \
    trainer.n_gpus_per_node=8 \
    trainer.nnodes=1 \
    trainer.save_freq=20 \
    trainer.test_freq=10 \
    trainer.val_before_train=False \
    trainer.total_epochs=1 \
    data.train_files="['../../data/train/hotpotqa-uma/train.parquet', \
                        '../../data/train/ledger-qa-train/train.parquet', \
                        '../../data/train/memalphafull-train/train.parquet']" \
    data.val_files="['../../data/train/ledger-qa-train/dev.parquet', \
                        '../../data/train/memalphafull-train/dev.parquet']" \
    custom_reward_function.path=./memagent/hotpotqa.py \
    custom_reward_function.name=reward_func $@

# 在parquet数据里指定用哪个agent loop
