# source /lpai/volumes/base-agentos-lx-my/zhangkehao/playground/restart.sh
# cd /lpai/volumes/base-agentos-lx-my/zhangkehao/unified-memory-agent
# CUDA_VISIBLE_DEVICES=4,5,6,7 vllm serve Qwen/Qwen3-4B-Instruct-2507 -dp 4 --max-model-len 262144 --port 8001 > /dev/null 2>&1 &
# vllm serve YuWangX/Memalpha-4B -dp 8 --rope-scaling '{"rope_type":"yarn","factor":4.0,"original_max_position_embeddings":32768}' --max-model-len 131072 > vllm${NODE_IP}.log 2>&1 &
# python /lpai/volumes/base-agentos-lx-my/zhangkehao/verl/memagent/embedding_server.py > embed${NODE_IP}.log 2>&1 &
until curl -s http://localhost:8000/health > /dev/null 2>&1; do
    sleep 2
    echo "wait for evaluating"
done


export EMBEDDING_SERVICE_ENDPOINT="http://localhost:8080/v1/embeddings"
# for TASK in banking77 booksum clinic hotpotqa locomo longmemeval memalpha msc nlu perltqa pubmed_rct trec_coarse trec_fine
export OPENAI_API_BASE="http://127.0.0.1:8000/v1"
export USE_SMALL_VALSETS=1
export PROMPT_TEMPLATE_PATH="/mnt/pfs-guan-ssai/nlu/zhangkehao/unified-memory-agent/prompt_template.yaml"
for TASK in synth-s10
do
    for AGENT in toolmem
    do
        python evaluate_async.py \
            --task $TASK \
            --agent $AGENT \
            --agent_id base-s10 \
            --model Qwen/Qwen3-4B-Instruct-2507 \
            --concurrency 50 \
            --output_dir results/qwen3-4b/debug \
            --generate_only
    done
done