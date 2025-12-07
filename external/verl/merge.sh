python -m verl.model_merger merge \
    --backend fsdp \
    --local_dir /mnt/pfs-guan-ssai/nlu/zhangkehao/verl/checkpoints/tool_memagent/qwen3-4b_GRPO_extend_datav1/actor \
    --target_dir /mnt/pfs-guan-ssai/nlu/zhangkehao/verl/checkpoints/tool_memagent/qwen3-4b_GRPO_extend_datav1/hf