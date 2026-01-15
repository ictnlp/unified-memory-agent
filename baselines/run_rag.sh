BASE_DIR="path/to/unified-memory-agent"
export EMBEDDING_SERVICE_ENDPOINT="http://localhost:8080/embeddings"

source ${BASE_DIR}/external/infinity/.venv/bin/activate
infinity_emb v2 --model-id sentence-transformers/all-MiniLM-L6-v2 --port 8080 > infinity_emb.log 2>&1 &
source .venv/bin/activate
vllm serve Qwen/Qwen3-4B-Instruct-2507 -tp 4 -dp 2 --max-model-len 262144 --enable-chunked-prefill --max-num-batched-tokens 512 > vllm.log 2>&1 &
until curl -s http://localhost:8080/health > /dev/null 2>&1; do
    sleep 2
    echo "wait for server port 8080..."
done
until curl -s http://localhost:8000/health > /dev/null 2>&1; do
    sleep 2
    echo "wait for server port 8000..."
done

TASKS="synth-ss2 synth-ss5"
RESULTS_DIR="results"
for TASK in $TASKS
do
    python evaluate_async.py \
        --task $TASK \
        --agent rag \
        --output_dir $RESULTS_DIR/$TASK \
        --generate_only
done
