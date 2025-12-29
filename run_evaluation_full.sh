export EMBEDDING_SERVICE_ENDPOINT="http://localhost:8080/embeddings"
export PROMPT_TEMPLATE_PATH="/mnt/pfs-guan-ssai/nlu/zhangkehao/unified-memory-agent/prompt_template.yaml"
# MODEL=/mnt/pfs-guan-ssai/nlu/zhangkehao/unified-memory-agent/external/verl/checkpoints/tool_memagent/qwen3-4b_GRPOv4/global_step_500/global_step_500/hf
MODEL=/mnt/pfs-guan-ssai/nlu/zhangkehao/unified-memory-agent/external/verl/checkpoints/tool_memagent/qwen3-4b_GRPOv4_onlysearch/global_step_766/hf
AGENT_ID=toolmem766onlysearch
# 定义清理函数
function kill_vllm_by_port() {
    local port=\$1
    
    # 1. 使用 ss 替代 lsof 查找 PID
    # grep ":$port " 确保精确匹配端口（避免匹配到 80001 这种）
    local pid=$(ss -lpnt | grep ":$port " | grep -oE 'pid=[0-9]+' | awk -F'=' '{print \$2}')
    
    if [ -n "$pid" ]; then
        echo "Found vLLM process on port $port (PID: $pid)..."
        
        # 2. 发送 SIGINT (信号 2)，完全等同于按下 Ctrl+C
        # 这是让 vLLM 优雅退出、自动释放显存的关键
        echo "Sending SIGINT (Ctrl+C) to allow graceful shutdown..."
        kill -2 $pid 2>/dev/null
        
        # 3. 等待进程退出 (最多等待 15 秒)
        # vLLM 优雅退出通常需要几秒钟来关闭 Ray worker
        for i in {1..15}; do
            if ! kill -0 $pid 2>/dev/null; then
                echo "Process $pid terminated gracefully."
                break
            fi
            sleep 1
            echo "Waiting for vLLM to cleanup (attempt $i/15)..."
        done
        
        # 4. 超时兜底：如果还在运行，只强制杀死该 PID 及其子进程
        if kill -0 $pid 2>/dev/null; then
            echo "Process stuck. Force killing PID $pid and its children..."
            
            # -P 指定父进程PID，只杀这个父进程下的子进程 (Ray worker)
            # 这样不会影响其他无关的 vLLM 服务
            pkill -9 -P $pid 2>/dev/null
            
            # 杀掉主进程
            kill -9 $pid 2>/dev/null
        fi
        
        # 5. 等待端口释放 (使用 ss 检查)
        echo "Waiting for port $port to be free..."
        for i in {1..10}; do
            # 如果 ss 查不到该端口了，说明释放成功
            if ! ss -lnt | grep -q ":$port "; then
                echo "Port $port is free."
                return 0
            fi
            sleep 1
        done
        
        echo "Warning: Port $port might still be in use or in TIME_WAIT."
        
    else
        echo "No process found on port $port."
    fi
}

kill_vllm_by_port 8000
vllm serve $MODEL -dp 2 -tp 4 > vllm.log 2>&1 &
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

for TASK in synth-s10 synth-s1 synth-s3 synth-s50 banking77 booksum clinic hotpotqa locomo longmemeval msc nlu perltqa pubmed_rct trec_coarse trec_fine squad infbench convomem
do
    python evaluate_async.py \
        --task $TASK \
        --agent toolmem \
        --model $MODEL \
        --agent_id $AGENT_ID \
        --concurrency 50 \
        --output_dir results/qwen3-4b/$TASK \
        --generate_only
done

kill_vllm_by_port 8000
kill_vllm_by_port 8080

CUDA_VISIBLE_DEVICES=4,5,6,7 vllm serve Qwen/Qwen3-30B-A3B-Instruct-2507 -tp 4 > vllm_judge.log 2>&1 &
until curl -s http://localhost:8000/health > /dev/null 2>&1; do
    sleep 2
    echo "wait for server port 8000..."
done

for TASK in synth-s10 synth-s1 synth-s3 synth-s50 banking77 booksum clinic hotpotqa locomo longmemeval msc nlu perltqa pubmed_rct trec_coarse trec_fine squad infbench convomem
do
    INPUT_FILE="results/qwen3-4b/$TASK/responses_${AGENT_ID}.jsonl"
    OUTPUT_DIR="results/qwen3-4b/$TASK"
    if [ -f "$INPUT_FILE" ]; then
        python evaluate_async.py \
            --task $TASK \
            --concurrency 256 \
            --input_file "$INPUT_FILE" \
            --output_dir "$OUTPUT_DIR"
    fi
done
kill_vllm_by_port 8000