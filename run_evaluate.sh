export CUDA_VISIBLE_DEVICES=0
# python evaluate.py --task locomo --agent concat --num_processes 8
# python evaluate.py --task locomo --agent memagent --wo_q --num_processes 8
python evaluate_async.py --task longmemeval --agent concat --concurrency 256 --output_dir results/qwen3-8b/longmemeval_oracle
# python evaluate.py --task longmemeval --agent memagent --wo_q --num_processes 8
# python evaluate.py --task longmemeval --agent memagent --num_processes 8
# python evaluate.py --task longmemeval --agent filememory --num_processes 8
# python evaluate.py --task longmemeval --agent emergence --num_processes 8