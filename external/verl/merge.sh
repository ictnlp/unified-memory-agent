python -m verl.model_merger merge \
    --backend fsdp \
    --local_dir /mnt/pfs-guan-ssai/nlu/zhangkehao/unified-memory-agent/external/verl/checkpoints/tool_memagent/qwen3-4b_stage2_qa/global_step_63/actor \
    --target_dir /mnt/pfs-guan-ssai/nlu/zhangkehao/unified-memory-agent/external/verl/checkpoints/tool_memagent/qwen3-4b_stage2_qa/global_step_63/hf