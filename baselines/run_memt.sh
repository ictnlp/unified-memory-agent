BASE_DIR="${BASE_DIR:-/mnt/pfs-guan-ssai/nlu/zhangkehao/unified-memory-agent}"
cd "$BASE_DIR"

unset http_proxy https_proxy HTTP_PROXY HTTPS_PROXY all_proxy

# MODEL="${MODEL:-Qwen/Qwen3-4B-Instruct-2507}"
MODEL="${MODEL:-EdwinYue/Mem-T-4B}"
AGENT_ID="${AGENT_ID:-memt}"
TASKS="${TASKS:-locomo}"
RESULTS_DIR="${RESULTS_DIR:-results/qwen3-4b}"
CONCURRENCY="${CONCURRENCY:-1}"

export OPENAI_API_KEY="${OPENAI_API_KEY:-EMPTY}"
export OPENAI_API_BASE="${OPENAI_API_BASE:-http://127.0.0.1:8000/v1}"
export MEMT_VECTOR_DB_TYPE="${MEMT_VECTOR_DB_TYPE:-persistent}"
export MEMT_EMBEDDING_MODEL="${MEMT_EMBEDDING_MODEL:-BAAI/bge-m3}"
export MEMT_RETRIEVAL_TOPK="${MEMT_RETRIEVAL_TOPK:-5}"
export MEMT_MAX_TOOL_STEPS="${MEMT_MAX_TOOL_STEPS:-6}"
export MEMT_MAX_TOKENS="${MEMT_MAX_TOKENS:-1024}"
export MEMT_TEMPERATURE="${MEMT_TEMPERATURE:-0.7}"
export MEMT_MIN_P="${MEMT_MIN_P:-0.05}"
export MEMT_USE_VLLM_EXTRAS="${MEMT_USE_VLLM_EXTRAS:-1}"

until curl --noproxy '*' -s "${OPENAI_API_BASE%/v1}/health" > /dev/null 2>&1; do
    sleep 2
    echo "wait for server ${OPENAI_API_BASE%/v1}..."
done

for TASK in $TASKS
do
    python evaluate_async.py \
        --task "$TASK" \
        --agent memt \
        --agent-id "$AGENT_ID" \
        --model "$MODEL" \
        --output-dir "$RESULTS_DIR/$TASK" \
        --concurrency "$CONCURRENCY" \
        --generate-only
done
