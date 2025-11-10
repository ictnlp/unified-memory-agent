export CUDA_VISIBLE_DEVICES=0
# python evaluate.py --task locomo --agent concat --num_processes 8
# python evaluate.py --task locomo --agent memagent --wo_q --num_processes 8
# python evaluate_async.py --task longmemeval --agent concat --concurrency 256 --output_dir results/qwen3-8b/longmemeval_oracle
# python evaluate.py --task longmemeval --agent memagent --wo_q --num_processes 8
# python evaluate.py --task longmemeval --agent memagent --num_processes 8
# python evaluate.py --task longmemeval --agent filememory --num_processes 8
# python evaluate.py --task longmemeval --agent emergence --num_processes 8

# vllm serve Qwen/Qwen3-4B-Instruct-2507 -dp 8
# for AGENT in concat emergence memagent memalpha
# for TASK in banking77 booksum clinic hotpotqa locomo longmemeval memalpha msc nlu perltqa pubmed_rct trec_coarse trec_fine
for AGENT in memalpha
do
    for TASK in banking77 booksum clinic hotpotqa locomo longmemeval memalpha msc nlu perltqa pubmed_rct trec_coarse trec_fine
    do
        python evaluate_async.py \
            --task $TASK \
            --agent $AGENT \
            --concurrency 128 \
            --output_dir results/qwen3-4b/$TASK \
            --generate_only
    done
done

# vllm serve Qwen/Qwen3-30B-A3B-Instruct-2507 -tp 8

# cd /mnt/pfs-guan-ssai/nlu/zhangkehao/verl
# python reformat.py --input_file /mnt/pfs-guan-ssai/nlu/zhangkehao/verl/checkpoints/tool_memagent/qwen3-4b_GRPO_extend_data_4nodes/results_msc.jsonl --task msc
# python reformat.py --input_file /mnt/pfs-guan-ssai/nlu/zhangkehao/verl/checkpoints/tool_memagent/qwen3-4b_DrGRPO/results_locomo.jsonl --task locomo
# python reformat.py --input_file /mnt/pfs-guan-ssai/nlu/zhangkehao/verl/checkpoints/tool_memagent/qwen3-4b_DrGRPO/results_longmemeval.jsonl --task longmemeval

# cd /mnt/pfs-guan-ssai/nlu/zhangkehao/unified-memory-agent
# source /mnt/pfs-guan-ssai/nlu/zhangkehao/.venv/bin/activate
# for AGENT in concat emergence memagent
# do
#     python evaluate_async.py \
#         --task msc \
#         --concurrency 256 \
#         --input_file /mnt/pfs-guan-ssai/nlu/zhangkehao/unified-memory-agent/results/qwen3-4b/msc/responses_${AGENT}.jsonl \
#         --output_dir results/qwen3-4b/msc
# done


# cd /mnt/pfs-guan-ssai/nlu/zhangkehao/unified-memory-agent
# python evaluate_async.py \
#     --task msc \
#     --concurrency 256 \
#     --input_file /mnt/pfs-guan-ssai/nlu/zhangkehao/verl/checkpoints/tool_memagent/qwen3-4b_GRPO_extend_data_4nodes/results_msc_reformat.jsonl \
#     --output_dir results/qwen3-4b/msc
# mv /mnt/pfs-guan-ssai/nlu/zhangkehao/unified-memory-agent/results/qwen3-4b/msc/evaluated_unknown.jsonl /mnt/pfs-guan-ssai/nlu/zhangkehao/unified-memory-agent/results/qwen3-4b/msc/evaluated_grpo116.jsonl