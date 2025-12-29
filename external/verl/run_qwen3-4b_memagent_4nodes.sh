# run on 8xH200
# make sure your current working directory is the root of the project
source /lpai/volumes/base-agentos-lx-my/zhangkehao/playground/restart.sh
set -x
ulimit -n 65535
# python memagent/embedding_server.py >> server.log 2>&1 &
# VLLM_ATTENTION_BACKEND=FLASH_ATTN vllm serve sentence-transformers/all-MiniLM-L6-v2 --port 8080 --max-model-len 512 --gpu-memory-utilization 0.1 > embed${NODE_IP}.log 2>&1 &
cd /lpai/volumes/base-agentos-lx-my/zhangkehao/unified-memory-agent/external/infinity
source .venv/bin/activate
infinity_emb v2 --model-id sentence-transformers/all-MiniLM-L6-v2 --port 8080 > server.log 2>&1 &
until curl -s http://localhost:8080/health > /dev/null 2>&1; do
    sleep 2
done

cd /lpai/volumes/base-agentos-lx-my/zhangkehao/unified-memory-agent/external/verl
source .venv/bin/activate
PROJECT_DIR="$(pwd)"
CONFIG_PATH="$PROJECT_DIR/examples/sglang_multiturn/config"
export VERL_LOGGING_LEVEL=DEBUG
export EMBEDDING_SERVICE_ENDPOINT="http://localhost:8080/embeddings"
export PROMPT_TEMPLATE_PATH="/lpai/volumes/base-agentos-lx-my/zhangkehao/unified-memory-agent/prompt_template.yaml"

if [ $RANK = "0" ]; then
    ray start --head --dashboard-host=0.0.0.0
    python3 -m verl.trainer.main_ppo \
        +ray_kwargs.ray_init.runtime_env.env_vars.HF_HUB_OFFLINE=\"1\" \
        algorithm.adv_estimator=grpo \
        data.train_batch_size=64 \
        data.max_prompt_length=8192 \
        data.max_response_length=8192 \
        data.filter_overlong_prompts=True \
        data.truncation='error' \
        data.return_raw_chat=True \
        actor_rollout_ref.model.path=Qwen/Qwen3-4B-Instruct-2507 \
        actor_rollout_ref.actor.optim.lr=1e-6 \
        actor_rollout_ref.model.use_remove_padding=True \
        actor_rollout_ref.actor.use_dynamic_bsz=True \
        actor_rollout_ref.actor.ppo_mini_batch_size=64 \
        actor_rollout_ref.actor.ppo_max_token_len_per_gpu=65536 \
        actor_rollout_ref.actor.use_kl_loss=True \
        actor_rollout_ref.actor.kl_loss_coef=0.001 \
        actor_rollout_ref.actor.kl_loss_type=low_var_kl \
        actor_rollout_ref.actor.entropy_coeff=0 \
        actor_rollout_ref.model.enable_gradient_checkpointing=True \
        actor_rollout_ref.actor.fsdp_config.param_offload=False \
        actor_rollout_ref.actor.fsdp_config.optimizer_offload=False \
        actor_rollout_ref.rollout.log_prob_max_token_len_per_gpu=65536 \
        actor_rollout_ref.rollout.tensor_model_parallel_size=1 \
        actor_rollout_ref.rollout.name=sglang \
        actor_rollout_ref.rollout.gpu_memory_utilization=0.5 \
        actor_rollout_ref.rollout.n=16 \
        actor_rollout_ref.ref.log_prob_max_token_len_per_gpu=65536 \
        actor_rollout_ref.ref.fsdp_config.param_offload=False \
        actor_rollout_ref.rollout.over_sample_rate=0.1 \
        actor_rollout_ref.rollout.mode=async \
        actor_rollout_ref.rollout.multi_turn.tool_config_path=./verl/memagent/tool_config.yaml \
        actor_rollout_ref.rollout.multi_turn.max_assistant_turns=100 \
        actor_rollout_ref.rollout.multi_turn.max_parallel_calls=100 \
        actor_rollout_ref.rollout.multi_turn.max_tool_response_length=8192 \
        actor_rollout_ref.rollout.multi_turn.tool_response_truncate_side=left \
        actor_rollout_ref.rollout.multi_turn.format=hermes \
        algorithm.use_kl_in_reward=True \
        trainer.critic_warmup=0 \
        trainer.logger='["console","swanlab"]' \
        trainer.project_name='tool_memagent' \
        trainer.experiment_name='qwen3-4b_GRPO' \
        trainer.n_gpus_per_node=8 \
        trainer.nnodes=4 \
        trainer.save_freq=20 \
        trainer.test_freq=10 \
        trainer.val_before_train=True \
        trainer.total_epochs=2 \
        data.train_files="['/lpai/volumes/base-agentos-lx-my/zhangkehao/unified-memory-agent/data/synth_train_verl_new.parquet',
                        '/mnt/pfs-guan-ssai/nlu/zhangkehao/unified-memory-agent/data/memalphafull_train_verl.parquet']" \
        data.val_files="['/lpai/volumes/base-agentos-lx-my/zhangkehao/MemAgent_minimal/taskutils/memory_data/hotpotqa_dev_agent_loop.parquet', \
                        '/lpai/volumes/base-agentos-lx-my/zhangkehao/unified-memory-agent/data/memalpha_dev_verl_small.parquet', \
                        '/lpai/volumes/base-agentos-lx-my/zhangkehao/unified-memory-agent/data/synth_dev_verl_new.parquet']" \
        custom_reward_function.path=/lpai/volumes/base-agentos-lx-my/zhangkehao/verl/memagent/hotpotqa.py \
        custom_reward_function.name=reward_func $@
else
    sleep 3
    ray start --address=${LPAI_MASTER_0_HOST}:6379 --block
fi

# 在parquet数据里指定用哪个agent loop
