# CUDA_VISIBLE_DEVICES=4,5,6,7 vllm serve Qwen/Qwen3-30B-A3B-Instruct-2507 -tp 4 --max-model-len 262144 --gpu-memory-utilization 0.8
CUDA_VISIBLE_DEVICES=0,1,2,3
cd /mnt/pfs-guan-ssai/nlu/zhangkehao/unified-memory-agent
source /mnt/pfs-guan-ssai/nlu/zhangkehao/.venv/bin/activate

until curl -s http://localhost:8000/health > /dev/null 2>&1; do
    sleep 2
    echo "wait for vllm server"
done

for AGENT in evaluated_toolmem90v4continualv2
do
    for TASK in synth-ss2 synth-ss3 synth-ss4 synth-ss5 synth-ss10 synth-ss20 synth-ss30 synth-ss40 synth-ss50 banking77 clinic hotpotqa locomo longmemeval msc nlu perltqa pubmed_rct trec_coarse trec_fine squad convomem
    do
        INPUT_FILE="results/qwen3-4b/$TASK/responses_${AGENT}.jsonl"
        OUTPUT_DIR="results/qwen3-4b/$TASK"
        if [ -f "$INPUT_FILE" ]; then
            python evaluate_async.py \
                --task $TASK \
                --concurrency 256 \
                --input_file "$INPUT_FILE" \
                --output_dir "$OUTPUT_DIR"
        fi
    done
done

# python evaluate_async.py \
#     --task synth-s10 \
#     --concurrency 256 \
#     --input_file "results/qwen3-30b/synth-s10/responses_base.jsonl" \
#     --output_dir "results/qwen3-30b/synth-s10"