TASKS="longmemeval locomo hotpotqa synth-ss2 synth-ss5 synth-ss10 synth-ss20 synth-ss30 synth-ss40 synth-ss50 banking77 booksum clinic msc nlu perltqa pubmed_rct trec_coarse trec_fine squad infbench convomem knowmebench"
RESULTS_DIR="results"
vllm serve BytedTsinghua-SIA/RL-MemoryAgent-7B -tp 8 --gpu-memory-utilization 0.8 --rope-scaling '{"rope_type":"yarn","factor":4.0,"original_max_position_embeddings":32768}' --max-model-len 131072 > vllm.log 2>&1 &
until curl -s http://localhost:8000/health > /dev/null 2>&1; do
    sleep 2
    echo "wait for server port 8000..."
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
        --agent memagent \
        --agent_id memagent7b \
        --output_dir $RESULTS_DIR/$TASK \
        --model BytedTsinghua-SIA/RL-MemoryAgent-7B \
        --generate_only
done
