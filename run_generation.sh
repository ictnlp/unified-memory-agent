export EMBEDDING_SERVICE_ENDPOINT="http://localhost:8080/embeddings"
export PROMPT_TEMPLATE_PATH="/mnt/pfs-guan-ssai/nlu/zhangkehao/unified-memory-agent/prompt_template.yaml"
# vllm serve /mnt/pfs-guan-ssai/nlu/zhangkehao/verl/checkpoints/tool_memagent/qwen3-4b_GRPO_extend_datav3/global_step_600/hf -tp 8
# VLLM_ATTENTION_BACKEND=FLASH_ATTN vllm serve sentence-transformers/all-MiniLM-L6-v2 --port 8080 --max-model-len 512
# infinity_emb v2 --model-id sentence-transformers/all-MiniLM-L6-v2 --port 8080


# Test-Time Learning (TTL): trec_coarse, trec_fine, nlu, clinic, banking77, pubmed_rct
# Accurate Retrieval (AR): hotpotqa, locomo, longmemeval, msc, perltqa, squad
# Long Range Understanding (LRU): infbench, booksum

# for AGENT in concat emergence memagent memalpha rag gam
# for TASK in synth-s10 synth-s1 synth-s3 synth-s50 banking77 booksum clinic hotpotqa locomo longmemeval msc nlu perltqa pubmed_rct trec_coarse trec_fine squad infbench convomem

until curl -s http://localhost:8000/health > /dev/null 2>&1; do
    sleep 2
    echo "wait for vllm server port 8000..."
done

# until curl -s http://localhost:8001/health > /dev/null 2>&1; do
#     sleep 2
#     echo "wait for vllm server port 8001..."
# done

# vllm serve Qwen/Qwen3-4B-Instruct-2507 -tp 4 -dp 2 --max-model-len 262144 --enable-chunked-prefill --max-num-batched-tokens 512
# for AGENT in concat
# do
#     for TASK in convomem #synth-s3 synth-s10 synth-s1 synth-s50 banking77 booksum clinic hotpotqa locomo longmemeval msc nlu perltqa pubmed_rct trec_coarse trec_fine squad infbench
#     do
#         python evaluate_async.py \
#             --task $TASK \
#             --agent $AGENT \
#             --concurrency 50 \
#             --output_dir results/qwen3-4b/$TASK \
#             --generate_only
#     done
# done

for TASK in synth-s10 synth-s1 synth-s3 synth-s50 banking77 booksum clinic hotpotqa locomo longmemeval msc nlu perltqa pubmed_rct trec_coarse trec_fine squad infbench convomem
do
    python evaluate_async.py \
        --task $TASK \
        --agent toolmem \
        --agent_id toolmem600v3fixfix \
        --model /mnt/pfs-guan-ssai/nlu/zhangkehao/unified-memory-agent/external/verl/checkpoints/tool_memagent/qwen3-4b_GRPO_extend_datav3/global_step_600/hf \
        --concurrency 50 \
        --output_dir results/qwen3-4b/$TASK \
        --generate_only
done


# CUDA_VISIBLE_DEVICES=4,5,6,7 vllm serve Qwen/Qwen3-4B-Instruct-2507 -tp 4 --max-model-len 262144 --port 8001 > vllm1.log 2>&1 &
# CUDA_VISIBLE_DEVICES=0,1,2,3 vllm serve YuWangX/Memalpha-4B -tp 4 --rope-scaling '{"rope_type":"yarn","factor":4.0,"original_max_position_embeddings":32768}' --max-model-len 131072 > vllm0.log 2>&1 &
# for TASK in convomem
# do
#     python evaluate_async.py \
#         --task $TASK \
#         --agent memalpha \
#         --agent_id memalphav1 \
#         --model YuWangX/Memalpha-4B \
#         --concurrency 50 \
#         --output_dir results/qwen3-4b/$TASK \
#         --generate_only
# done


# CUDA_VISIBLE_DEVICES=0,1,2,3 vllm serve Mem-Lab/Qwen2.5-7B-RL-RAG-Q2-EM-Release -tp 4 --rope-scaling '{"rope_type":"yarn","factor":4.0,"original_max_position_embeddings":32768}' --max-model-len 131072 > vllm0.log 2>&1 &
# CUDA_VISIBLE_DEVICES=4,5,6,7 vllm serve Qwen/Qwen3-4B-Instruct-2507 -tp 4 --max-model-len 262144 --port 8001 > vllm1.log 2>&1 &
# for TASK in synth-s1 synth-s3 synth-s50 banking77 booksum clinic hotpotqa locomo longmemeval msc nlu perltqa pubmed_rct trec_coarse trec_fine squad infbench
# do
#     python evaluate_async.py \
#         --task $TASK \
#         --agent mem1 \
#         --model Mem-Lab/Qwen2.5-7B-RL-RAG-Q2-EM-Release \
#         --concurrency 50 \
#         --output_dir results/qwen3-4b/$TASK \
#         --generate_only
# done