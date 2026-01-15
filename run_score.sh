CUDA_VISIBLE_DEVICES=4,5,6,7 vllm serve Qwen/Qwen3-30B-A3B-Instruct-2507 -tp 4 --max-model-len 262144 --gpu-memory-utilization 0.8 > vllm.log 2>&1 &
until curl -s http://localhost:8000/health > /dev/null 2>&1; do
    sleep 2
    echo "wait for server port 8000..."
done

AGENTS="concat rag memagent memagent_woq memalpha mem1"
TASKS="synth-ss2 synth-ss5"
RESULTS_DIR="results"

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