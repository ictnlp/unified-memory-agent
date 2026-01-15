BASE_DIR="path/to/unified-memory-agent"
export EMBEDDING_SERVICE_ENDPOINT="http://localhost:8080/embeddings"

source ${BASE_DIR}/external/infinity/.venv/bin/activate
infinity_emb v2 --model-id sentence-transformers/all-MiniLM-L6-v2 --port 8080 > infinity_emb.log 2>&1 &
source .venv/bin/activate
CUDA_VISIBLE_DEVICES=4,5,6,7 vllm serve Qwen/Qwen3-4B-Instruct-2507 -tp 4 --max-model-len 262144 --port 8001 > vllm1.log 2>&1 &
CUDA_VISIBLE_DEVICES=0,1,2,3 vllm serve YuWangX/Memalpha-4B -tp 4 --rope-scaling '{"rope_type":"yarn","factor":4.0,"original_max_position_embeddings":32768}' --max-model-len 131072 > vllm0.log 2>&1 &
until curl -s http://localhost:8080/health > /dev/null 2>&1; do
    sleep 2
    echo "wait for server port 8080..."
done
until curl -s http://localhost:8000/health > /dev/null 2>&1; do
    sleep 2
    echo "wait for server port 8000..."
done
until curl -s http://localhost:8001/health > /dev/null 2>&1; do
    sleep 2
    echo "wait for server port 8001..."
done

TASKS="synth-ss2 synth-ss5"
RESULTS_DIR="results"
for TASK in $TASKS
do
    python evaluate_async.py \
        --task $TASK \
        --agent memalpha \
        --model YuWangX/Memalpha-4B \
        --output_dir $RESULTS_DIR/$TASK \
        --generate_only
done
