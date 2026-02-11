export EMBEDDING_SERVICE_ENDPOINT="http://localhost:8080/embeddings"
export PROMPT_TEMPLATE_PATH="/mnt/pfs-guan-ssai/nlu/zhangkehao/unified-memory-agent/prompt_template.yaml"

function kill_vllm_by_port() {
    local port=$1
    local pid=$(ss -lptn "sport = :$port" | grep -oE 'pid=[0-9]+' | cut -d= -f2)

    if [ -z "$pid" ]; then
        echo "No process on port $port."
        return
    fi

    echo "Stopping vLLM (PID: $pid)..."
    kill -2 "$pid" # 发送 SIGINT

    # 等待循环：利用 kill -0 检查进程是否存在
    for i in {1..15}; do
        kill -0 "$pid" 2>/dev/null || break # 进程消失则跳出循环
        sleep 1
    done

    # 兜底强制查杀
    if kill -0 "$pid" 2>/dev/null; then
        echo "Force killing PID $pid and children..."
        pkill -9 -P "$pid"  # 杀子进程 (Ray workers)
        kill -9 "$pid"      # 杀主进程
    fi

    echo "Cleanup finished for port $port."
    echo "Waiting for GPU resources to be released..."
    sleep 15  # 等待GPU资源完全释放（CUDA上下文、显存、Ray workers等）
}
# kill_vllm_by_port 8000
# kill_vllm_by_port 8001
# kill_vllm_by_port 8080
# for TASK in synth-ss500 synth-ss100
# do
#     python evaluate_async.py \
#         --task $TASK \
#         --agent toolmem \
#         --agent_id base_rerun \
#         --concurrency 10 \
#         --output_dir results/qwen3-4b/$TASK \
#         --generate_only
# done

# CUDA_VISIBLE_DEVICES=0,1,2,3 vllm serve /mnt/pfs-guan-ssai/nlu/zhangkehao/unified-memory-agent/external/verl/checkpoints/tool_memagent/qwen3-4b_stage1_memory/global_step_63/hf -tp 4 --gpu-memory-utilization 0.85 > vllm0.log 2>&1 &
# CUDA_VISIBLE_DEVICES=4,5,6,7 vllm serve /mnt/pfs-guan-ssai/nlu/zhangkehao/unified-memory-agent/external/verl/checkpoints/tool_memagent/qwen3-4b_stage2_qa/global_step_63/hf -tp 4 --gpu-memory-utilization 0.85 --port 8001 > vllm1.log 2>&1 &
# source external/infinity/libs/infinity_emb/.venv/bin/activate
# infinity_emb v2 --model-id sentence-transformers/all-MiniLM-L6-v2 --port 8080 > infinity_emb.log 2>&1 &
# source .venv/bin/activate
until curl -s http://localhost:8080/health > /dev/null 2>&1; do
    sleep 2
    echo "wait for server port 8080..."
    if ! pgrep -f "infinity_emb" > /dev/null; then
        echo "Error: infinity_emb process died. Check infinity_emb.log for details:"
        cat infinity_emb.log
        exit 1
    fi
done
until curl -s http://localhost:8000/health > /dev/null 2>&1; do
    sleep 2
    echo "wait for server port 8000..."
    if ! pgrep -f "vllm serve" > /dev/null; then
        echo "Error: vllm process died. Check vllm0.log for details:"
        cat vllm0.log
        exit 1
    fi
done
until curl -s http://localhost:8001/health > /dev/null 2>&1; do
    sleep 2
    echo "wait for server port 8001..."
    if ! pgrep -f "vllm serve" > /dev/null; then
        echo "Error: vllm process died. Check vllm1.log for details:"
        cat vllm1.log
        exit 1
    fi
done
export VERL_QA_MODEL_NAME="/mnt/pfs-guan-ssai/nlu/zhangkehao/unified-memory-agent/external/verl/checkpoints/tool_memagent/qwen3-4b_stage2_qa/global_step_63/hf"
for TASK in hotpotqa
do
    python evaluate_async.py \
        --task $TASK \
        --agent toolmem \
        --agent_id toolmem2stage \
        --model /mnt/pfs-guan-ssai/nlu/zhangkehao/unified-memory-agent/external/verl/checkpoints/tool_memagent/qwen3-4b_stage1_memory/global_step_63/hf \
        --concurrency 10 \
        --output_dir results/qwen3-4b/$TASK \
        --generate_only
done