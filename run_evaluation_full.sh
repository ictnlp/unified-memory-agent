BASE_DIR="/mnt/pfs-guan-ssai/nlu/zhangkehao/unified-memory-agent"
export EMBEDDING_SERVICE_ENDPOINT="http://localhost:8080/embeddings"
export PROMPT_TEMPLATE_PATH="${BASE_DIR}/prompt_template.yaml"
MODEL=${BASE_DIR}/external/verl/checkpoints/tool_memagent/qwen3-4b_GRPO_synth/global_step_20/hf
AGENT_ID=toolmem20synth
# MODEL=Qwen/Qwen3-4B-Instruct-2507
# AGENT_ID=base_onlysearch
TASKS="synth-ss2 synth-ss5 synth-ss10 synth-ss20 synth-ss30 synth-ss40 synth-ss50 synth-ss500 synth-ss100"
# 定义清理函数
function kill_vllm_by_port() {
    local port=$1
    # 优化 PID 提取：利用 ss 的过滤器功能，减少 grep/awk 管道
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

kill_vllm_by_port 8000
vllm serve $MODEL -dp 2 -tp 4 --gpu-memory-utilization 0.8 --enforce-eager > vllm.log 2>&1 &
# source /mnt/pfs-guan-ssai/nlu/zhangkehao/unified-memory-agent/external/infinity/.venv/bin/activate
# cd libs/infinity_emb
# uv pip install -e ".[all]"
# uv pip install click==8.1.8
# cd /mnt/pfs-guan-ssai/nlu/zhangkehao/unified-memory-agent
# infinity_emb v2 --model-id sentence-transformers/all-MiniLM-L6-v2 --port 8080 > infinity_emb.log 2>&1 &
# source /mnt/pfs-guan-ssai/nlu/zhangkehao/.venv/bin/activate

# for AGENT in concat emergence memagent memalpha rag gam
# for TASK in synth-s10 synth-s1 synth-s3 synth-s50 banking77 booksum clinic hotpotqa locomo longmemeval msc nlu perltqa pubmed_rct trec_coarse trec_fine squad infbench convomem
until curl -s http://localhost:8080/health > /dev/null 2>&1; do
    sleep 2
    echo "wait for server port 8080..."
done
until curl -s http://localhost:8000/health > /dev/null 2>&1; do
    sleep 2
    echo "wait for server port 8000..."
done

for TASK in $TASKS
do
    python evaluate_async.py \
        --task $TASK \
        --agent toolmem \
        --model $MODEL \
        --agent_id $AGENT_ID \
        --concurrency 10 \
        --output_dir results/qwen3-4b/$TASK \
        --generate_only
done

kill_vllm_by_port 8000
CUDA_VISIBLE_DEVICES=4,5,6,7 vllm serve Qwen/Qwen3-30B-A3B-Instruct-2507 -tp 4 --max-model-len 262144 --gpu-memory-utilization 0.8 > vllm_judge.log 2>&1 &
until curl -s http://localhost:8000/health > /dev/null 2>&1; do
    sleep 2
    echo "wait for server port 8000..."
done

for TASK in $TASKS
do
    INPUT_FILE="results/qwen3-4b/$TASK/responses_${AGENT_ID}.jsonl"
    OUTPUT_DIR="results/qwen3-4b/$TASK"
    if [ -f "$INPUT_FILE" ]; then
        python evaluate_async.py \
            --task $TASK \
            --concurrency 256 \
            --input_file "$INPUT_FILE" \
            --output_dir "$OUTPUT_DIR" \
            --force-overwrite
    fi
done
kill_vllm_by_port 8000

[ -n "$USE_SMALL_VALSETS" ] && python generate_stats.py --results_dir results/qwen3-4b-small/ || python generate_stats.py