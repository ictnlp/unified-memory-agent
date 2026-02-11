BASE_DIR="path/to/unified-memory-agent"
export EMBEDDING_SERVICE_ENDPOINT="http://localhost:8080/embeddings"

source ${BASE_DIR}/external/infinity/.venv/bin/activate
infinity_emb v2 --model-id sentence-transformers/all-MiniLM-L6-v2 --port 8080 > infinity_emb.log 2>&1 &
source .venv/bin/activate
CUDA_VISIBLE_DEVICES=4,5,6,7 vllm serve Qwen/Qwen3-4B-Instruct-2507 -tp 4 --max-model-len 262144 --port 8001 > vllm1.log 2>&1 &
CUDA_VISIBLE_DEVICES=0,1,2,3 vllm serve Mem-Lab/Qwen2.5-7B-RL-RAG-Q2-EM-Release -tp 4 --rope-scaling '{"rope_type":"yarn","factor":4.0,"original_max_position_embeddings":32768}' --max-model-len 131072 > vllm0.log 2>&1 &
until curl -s http://localhost:8080/health > /dev/null 2>&1; do
    sleep 2
    echo "wait for server port 8080..."
    if ! pgrep -f "infinity_emb" > /dev/null; then
        echo "Error: infinity_emb process died. Check infinity_emb.log for details:"
        cat infinity_emb.log
        exit 1
    fi
done
until curl -s http://localhost:8000/health > /dev/null 2>&1; do
    sleep 2
    echo "wait for server port 8000..."
    if ! pgrep -f "vllm serve" > /dev/null; then
        echo "Error: vllm process died. Check vllm0.log for details:"
        cat vllm0.log
        exit 1
    fi
done
until curl -s http://localhost:8001/health > /dev/null 2>&1; do
    sleep 2
    echo "wait for server port 8001..."
    if ! pgrep -f "vllm serve" > /dev/null; then
        echo "Error: vllm process died. Check vllm1.log for details:"
        cat vllm1.log
        exit 1
    fi
done

TASKS="longmemeval locomo hotpotqa synth-ss2 synth-ss5 synth-ss10 synth-ss20 synth-ss30 synth-ss40 synth-ss50 banking77 booksum clinic msc nlu perltqa pubmed_rct trec_coarse trec_fine squad infbench convomem knowmebench"
RESULTS_DIR="results"
for TASK in $TASKS
do
    python evaluate_async.py \
        --task $TASK \
        --agent mem1 \
        --model Mem-Lab/Qwen2.5-7B-RL-RAG-Q2-EM-Release \
        --output_dir $RESULTS_DIR/$TASK \
        --generate_only
done
