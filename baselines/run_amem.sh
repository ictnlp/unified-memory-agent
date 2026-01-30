# TASKS="synth-ss2 synth-ss5 synth-ss10 synth-ss20 synth-ss30 synth-ss40 synth-ss50 banking77 booksum clinic hotpotqa locomo longmemeval msc nlu perltqa pubmed_rct trec_coarse trec_fine squad infbench convomem"
TASKS="synth-ss2"
RESULTS_DIR="results/qwen3-4b"
# vllm serve Qwen/Qwen3-4B-Instruct-2507 -tp 4 -dp 2 --max-model-len 262144 --enable-chunked-prefill --max-num-batched-tokens 512 > vllm.log 2>&1 &
export OPENAI_API_KEY="EMPTY"
until curl -s http://localhost:8000/health > /dev/null 2>&1; do
    sleep 2
    echo "wait for server port 8000..."
done

for TASK in $TASKS
do
    python evaluate_async.py \
        --task $TASK \
        --agent amem \
        --output_dir $RESULTS_DIR/$TASK \
        --concurrency 1 \
        --generate_only
done
