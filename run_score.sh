# CUDA_VISIBLE_DEVICES=4,5,6,7 vllm serve Qwen/Qwen3-30B-A3B-Instruct-2507 -tp 4 --max-model-len 262144
CUDA_VISIBLE_DEVICES=0,1,2,3
cd /mnt/pfs-guan-ssai/nlu/zhangkehao/unified-memory-agent
source /mnt/pfs-guan-ssai/nlu/zhangkehao/.venv/bin/activate

until curl -s http://localhost:8000/health > /dev/null 2>&1; do
    sleep 2
    echo "wait for vllm server"
done

# for AGENT in base concat emergence mem1 memagent memagent_woq memalphav1 toolmem513 toolmemv1 rag
for AGENT in toolmem600v3fixfix
do
    for TASK in synth-s10 synth-s1 synth-s3 synth-s50 banking77 booksum clinic hotpotqa locomo longmemeval msc nlu perltqa pubmed_rct trec_coarse trec_fine squad infbench convomem
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