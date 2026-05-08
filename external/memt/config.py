from dataclasses import dataclass, field
from typing import Dict, Any, List
import os
from datetime import datetime


@dataclass
class VectorDBConfig:
    backend: str = "chroma"  
    db_type: str = "http"  
    path: str = "./database/" 
    host: str = "localhost"  
    port: int = 8070         
    embedding_model: str = field(default_factory=lambda: os.getenv("MEMT_EMBEDDING_MODEL", "BAAI/bge-m3"))
    index_type: str = "HNSW"
    metric_type: str = "cosine"
    from_scratch: bool = True

@dataclass
class MemoryConfig:
    chunk_size: int = 512
    summary_context_turns: int = 4
    update_retrieval_topk: int = 3 
    retrieval_topk: int = 5
    max_tool_steps: int = 6
    max_facts_per_turn: int = 5
    max_notes_per_turn: int = 3

@dataclass
class LLMConfig:
    strong_model: str = "gpt-4o-mini"
    strong_api_key: str = field(default_factory=lambda: os.getenv("OPENAI_API_KEY", ""))
    strong_api_base: str = field(default_factory=lambda: os.getenv("OPENAI_API_BASE", ""))
    local_model: str = "Qwen/Qwen3-4B-Instruct-2507"
    local_model_path: str = "Qwen/Qwen3-4B-Instruct-2507"
    temperature: float = 0.0
    max_tokens: int = 65536

@dataclass
class SystemConfig:
    vector_db: VectorDBConfig = field(default_factory=VectorDBConfig)
    memory: MemoryConfig = field(default_factory=MemoryConfig)
    llm: LLMConfig = field(default_factory=LLMConfig)
    checkpoint_dir: str = "./checkpoints"    
    algorithm: str = "pipeline"
    data_name: str = "locomo"
    dataset_dir: str = f"./data/{data_name}"
    dataset_path: str = f"./data/{data_name}/locomo10.json"
    mode: str = "test"
    USE_LOCAL_LLM: bool = True
    USE_PARALLEL: bool = True
    NUM_WORKERS: int = 4
    GPU_IDS: List[int] = field(default_factory=lambda: [4, 5, 6, 7])
    seed = 42
    log_path: str = f"./logs/{data_name}_{algorithm}_{USE_LOCAL_LLM}_{mode}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log"
    traj_dir: str = f"./traj/{data_name}_{algorithm}_{USE_LOCAL_LLM}_{mode}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    
