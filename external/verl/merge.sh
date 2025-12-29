python -m verl.model_merger merge \
    --backend fsdp \
    --local_dir /mnt/pfs-guan-ssai/nlu/zhangkehao/unified-memory-agent/external/verl/checkpoints/tool_memagent/qwen3-4b_GRPOv4_continualv2/global_step_90/actor \
    --target_dir /mnt/pfs-guan-ssai/nlu/zhangkehao/unified-memory-agent/external/verl/checkpoints/tool_memagent/qwen3-4b_GRPOv4_continualv2/global_step_90/hf