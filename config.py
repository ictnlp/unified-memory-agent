import importlib

from data.EvalDataset import (
    load_locomo,
    load_longmemeval,
    load_hotpotqa,
    load_msc,
    load_memalpha,
    load_trec_coarse,
    load_banking77,
    load_clinic,
    load_nlu,
    load_trec_fine,
    load_booksum,
    load_perltqa,
    load_pubmed_rct,
    load_synth,
    load_squad,
    load_infbench,
    load_convomem,
)
import uuid
from functools import partial

from dotenv import load_dotenv
load_dotenv(".env") # 默认会找项目根目录的 .env
import os
import httpx

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
    "max_retries": 3,  # 降低重试次数，快速失败
    "timeout": 180.0,  # 增加到3分钟，应对长推理
    "http_client": httpx.AsyncClient(
        limits=httpx.Limits(
            max_connections=500,           # 增加连接池
            max_keepalive_connections=200, # 增加保活连接
            keepalive_expiry=2.0           # ✅ 空闲2秒回收，不影响运行中的请求
        ),
        timeout=httpx.Timeout(
            180.0,        # ✅ 总超时3分钟，允许长推理
            connect=5.0,  # ✅ 连接建立5秒超时，快速发现连接问题
            read=180.0,   # ✅ 读取超时3分钟
            write=10.0    # 写入超时10秒
        )
    )
}
MODEL_NAME = "Qwen/Qwen3-8B"

load_hotpotqa_10_3_5 = partial(load_hotpotqa, num_docs=10, num_queries=3, num_samples=5)
load_hotpotqa_200_1_128 = partial(load_hotpotqa, num_docs=200, num_queries=1, num_samples=128)

_DATASET_LOADERS = {
    'locomo': load_locomo,
    'longmemeval': load_longmemeval, 
    'hotpotqa': load_hotpotqa_200_1_128,
    'msc': load_msc,
    'memalpha': load_memalpha,
    'trec_coarse': load_trec_coarse,
    'trec_fine': load_trec_fine,
    'banking77': load_banking77,
    'clinic': load_clinic,
    'nlu': load_nlu,
    'booksum': load_booksum,
    'perltqa': load_perltqa,
    'pubmed_rct': load_pubmed_rct,
    'squad': load_squad,
    'infbench': load_infbench,
    'convomem': load_convomem,
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
        if key.startswith("synth-"):
            parts = key.split('-')
            return partial(load_synth,
                           suf=parts[1])
        raise KeyError(key)
    
    def get(self, key, default=None):
        return self.__getitem__(key) or default
    
    def __contains__(self, key):
        return self.__getitem__(key) is not None

DATASET_LOADERS = DatasetLoaders()

class AgentRegistry:
    _AGENT_SPECS = {
        'concat': ('agents.concat_agent', 'ConcatAgent'),
        'memagent': ('agents.mem_agent', 'MemAgent'),
        'filememory': ('agents.file_memory_agent', 'FileMemoryAgent'),
        'emergence': ('agents.emergence_agent', 'EmergenceAgent'),
        'rag': ('agents.rag_agent', 'RAGAgent'),
        'memalpha': ('agents.mem_alpha_agent', 'MemAlphaUnifiedAgent'),
        'toolmem': ('agents.verl_agent', 'VerlMemoryAgent'),
        'mem1': ('agents.mem1_agent', 'Mem1Agent'),
        'gam': ('agents.gam_agent', 'GAMAgent'),
    }

    def __init__(self) -> None:
        self._cache = {}

    def __getitem__(self, key):
        if key in self._cache:
            return self._cache[key]
        spec = self._AGENT_SPECS.get(key)
        if spec is None:
            raise KeyError(key)
        module_name, class_name = spec
        module = importlib.import_module(module_name)
        agent_cls = getattr(module, class_name)
        self._cache[key] = agent_cls
        return agent_cls

    def get(self, key, default=None):
        try:
            return self.__getitem__(key)
        except KeyError:
            return default

    def __contains__(self, key):
        return key in self._AGENT_SPECS


AGENT_CLASS = AgentRegistry()