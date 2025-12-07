# source /lpai/volumes/base-agentos-lx-my/zhangkehao/playground/restart.sh
# cd /lpai/volumes/base-agentos-lx-my/zhangkehao/unified-memory-agent
# CUDA_VISIBLE_DEVICES=4,5,6,7 vllm serve Qwen/Qwen3-4B-Instruct-2507 -dp 4 --max-model-len 262144 --port 8001 > /dev/null 2>&1 &
# vllm serve YuWangX/Memalpha-4B -dp 8 --rope-scaling '{"rope_type":"yarn","factor":4.0,"original_max_position_embeddings":32768}' --max-model-len 131072 > vllm${NODE_IP}.log 2>&1 &
# python /lpai/volumes/base-agentos-lx-my/zhangkehao/verl/memagent/embedding_server.py > embed${NODE_IP}.log 2>&1 &
until curl -s http://localhost:8000/health > /dev/null 2>&1; do
    sleep 2
    echo "wait for evaluating"
done


export EMBEDDING_SERVICE_ENDPOINT="http://localhost:8080/embeddings"
export OPENAI_API_BASE="http://127.0.0.1:8000/v1"
export USE_SMALL_VALSETS=1
export PROMPT_TEMPLATE_PATH="/mnt/pfs-guan-ssai/nlu/zhangkehao/unified-memory-agent/prompt_template.yaml"
TASK=synth-s10

# --enable-chunked-prefill --max-num-batched-tokens 512主要是针对concat，序列过长的时候
# vllm serve Qwen/Qwen3-4B-Instruct-2507 -tp 4 -dp 2 --max-model-len 262144 --enable-chunked-prefill --max-num-batched-tokens 512 > vllm.log 2>&1 &
# for AGENT in concat emergence memagent memagent_woq rag
# do
#     python evaluate_async.py \
#         --task $TASK \
#         --agent $AGENT \
#         --concurrency 1 \
#         --output_dir results/qwen3-4b/debug \
#         --generate_only
# done

# CUDA_VISIBLE_DEVICES=0,1,2,3 vllm serve YuWangX/Memalpha-4B -tp 4 --rope-scaling '{"rope_type":"yarn","factor":4.0,"original_max_position_embeddings":32768}' --max-model-len 131072 --enable-chunked-prefill --max-num-batched-tokens 512 > vllm0.log 2>&1 &
# CUDA_VISIBLE_DEVICES=4,5,6,7 vllm serve Qwen/Qwen3-4B-Instruct-2507 -tp 4 --max-model-len 262144 --port 8001 > vllm1.log 2>&1 &
# python evaluate_async.py \
#     --task $TASK \
#     --agent memalpha \
#     --model YuWangX/Memalpha-4B \
#     --concurrency 1 \
#     --output_dir results/qwen3-4b/debug \
#     --generate_only

# CUDA_VISIBLE_DEVICES=0,1,2,3 vllm serve Mem-Lab/Qwen2.5-7B-RL-RAG-Q2-EM-Release -tp 4 --rope-scaling '{"rope_type":"yarn","factor":4.0,"original_max_position_embeddings":32768}' --max-model-len 131072 > vllm0.log 2>&1 &
# CUDA_VISIBLE_DEVICES=4,5,6,7 vllm serve Qwen/Qwen3-4B-Instruct-2507 -tp 4 --max-model-len 262144 --port 8001 > vllm1.log 2>&1 &

python evaluate_async.py \
    --task $TASK \
    --agent mem1 \
    --model Mem-Lab/Qwen2.5-7B-RL-RAG-Q2-EM-Release \
    --concurrency 1 \
    --output_dir results/qwen3-4b/debug \
    --generate_only

# MODEL_PATH=/mnt/pfs-guan-ssai/nlu/zhangkehao/verl/checkpoints/tool_memagent/qwen3-4b_GRPO_extend_data_4nodes/hf171
# vllm serve $MODEL_PATH -tp 8
# python evaluate_async.py \
#     --task $TASK \
#     --agent toolmem \
#     --agent_id base \
#     --concurrency 1 \
#     --output_dir results/qwen3-4b/debug \
#     --generate_only
