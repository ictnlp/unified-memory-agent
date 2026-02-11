CUDA_VISIBLE_DEVICES=4,5,6,7 vllm serve Qwen/Qwen3-30B-A3B-Instruct-2507 -tp 4 --max-model-len 262144 --gpu-memory-utilization 0.8 > vllm_judge.log 2>&1 &
until curl -s http://localhost:8000/health > /dev/null 2>&1; do
    sleep 2
    echo "wait for server port 8000..."

    # Check if vllm process is still alive
    if ! pgrep -f "vllm serve" > /dev/null; then
        echo "Error: vllm process died. Check vllm.log for details:"
        cat vllm.log
        exit 1
    fi
done

AGENTS="toolmem2stage"
# TASKS="longmemeval locomo hotpotqa synth-ss2 synth-ss5 synth-ss10 synth-ss20 synth-ss30 synth-ss40 synth-ss50 banking77 booksum clinic msc nlu perltqa pubmed_rct trec_coarse trec_fine squad infbench convomem knowmebench"
TASKS="hotpotqa"
RESULTS_DIR="results/qwen3-4b"

export CUDA_VISIBLE_DEVICES=0,1,2,3
for AGENT in $AGENTS
do
    for TASK in $TASKS
    do
        INPUT_FILE="$RESULTS_DIR/$TASK/responses_${AGENT}.jsonl"
        OUTPUT_DIR="$RESULTS_DIR/$TASK"
        if [ -f "$INPUT_FILE" ]; then
            python evaluate_async.py \
                --task $TASK \
                --concurrency 256 \
                --input_file "$INPUT_FILE" \
                --output_dir "$OUTPUT_DIR"
        fi
    done
done

python generate_stats.py --results_dir $RESULTS_DIR