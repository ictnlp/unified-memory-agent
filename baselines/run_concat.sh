TASKS="longmemeval locomo hotpotqa synth-ss2 synth-ss5 synth-ss10 synth-ss20 synth-ss30 synth-ss40 synth-ss50 banking77 booksum clinic msc nlu perltqa pubmed_rct trec_coarse trec_fine squad infbench convomem knowmebench"
RESULTS_DIR="results"
vllm serve Qwen/Qwen3-4B-Instruct-2507 -tp 4 -dp 2 --max-model-len 262144 --enable-chunked-prefill --max-num-batched-tokens 512 > vllm.log 2>&1 &

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

for TASK in $TASKS
do
    python evaluate_async.py \
        --task $TASK \
        --agent concat \
        --output_dir $RESULTS_DIR/$TASK \
        --generate_only
done
