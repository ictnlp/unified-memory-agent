TASKS="synth-ss2 synth-ss5"
RESULTS_DIR="results"
vllm serve Qwen/Qwen3-4B-Instruct-2507 -tp 4 -dp 2 --max-model-len 262144 --enable-chunked-prefill --max-num-batched-tokens 512 > vllm.log 2>&1 &

until curl -s http://localhost:8000/health > /dev/null 2>&1; do
    sleep 2
    echo "wait for server port 8000..."
done

for TASK in $TASKS
do
    python evaluate_async.py \
        --task $TASK \
        --agent memagent \
        --output_dir $RESULTS_DIR/$TASK \
        --generate_only
done
