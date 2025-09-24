from agents import BaseAgent, ConcatAgent, MemAgent
from agents.file_memory_agent import FileMemoryAgent
from agents.emergence_agent import EmergenceAgent
from data.EvalDataset import load_locomo, load_longmemeval, load_hotpotqa
import uuid
from functools import partial

from dotenv import load_dotenv
load_dotenv(".env") # 默认会找项目根目录的 .env
import os

# Configuration
API_CONFIG = {
    "base_url": "http://api-hub.inner.chj.cloud/llm-gateway/v1",
    "api_key": "sk-",
    "default_headers": {
        "BCS-APIHub-RequestId": str(uuid.uuid4()),
        "X-CHJ-GWToken": os.getenv("X-CHJ-GWToken"),
        "X-CHJ-GW-SOURCE": os.getenv("X-CHJ-GW-SOURCE"),
    },
    "max_retries": 100
}
JUDGE_MODEL_NAME = "azure-gpt-4o"
# MODEL_NAME = "azure-gpt-4_1"

API_CONFIG_LOCAL = {
    "base_url": "http://127.0.0.1:8000/v1",
    "api_key": "EMPTY",
    "max_retries": 100
}
MODEL_NAME = "Qwen/Qwen3-8B"

load_hotpotqa_10_3_5 = partial(load_hotpotqa, num_docs=10, num_queries=3, num_samples=5)

_DATASET_LOADERS = {
    'locomo': load_locomo,
    'longmemeval': load_longmemeval, 
    'hotpotqa': load_hotpotqa_10_3_5
}

class DatasetLoaders:
    def __getitem__(self, key):
        if key in _DATASET_LOADERS:
            return _DATASET_LOADERS[key]
        if key.startswith("hotpotqa_"):
            parts = key.split('_')
            return partial(load_hotpotqa,
                         num_docs=int(parts[1]),
                         num_queries=int(parts[2]),
                         num_samples=int(parts[3]))
        return None
    
    def get(self, key, default=None):
        return self.__getitem__(key) or default
    
    def __contains__(self, key):
        return self.__getitem__(key) is not None

DATASET_LOADERS = DatasetLoaders()

AGENT_CLASS = {
    'concat': ConcatAgent,
    'memagent': MemAgent,
    'filememory': FileMemoryAgent,
    'emergence': EmergenceAgent
}