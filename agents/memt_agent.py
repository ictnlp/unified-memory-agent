from __future__ import annotations

import asyncio
import os
import re
import uuid
from pathlib import Path
from typing import Any, Dict, List, Optional

from openai import OpenAI

from .base_agent import BaseAgent
from config import API_CONFIG_LOCAL


PROJECT_ROOT = Path(__file__).resolve().parents[1]


class _OpenAICompatibleMemTClient:
    """Adapter from this repo's OpenAI clients to Mem-T's sync get_completion API."""

    def __init__(self, client: Any, model_name: str, max_tokens: Optional[int] = None):
        self.model_name = model_name
        self.max_tokens = int(max_tokens or os.getenv("MEMT_MAX_TOKENS", "1024"))
        self.temperature = float(os.getenv("MEMT_TEMPERATURE", "0.7"))
        self.min_p = float(os.getenv("MEMT_MIN_P", "0.05"))
        self.use_vllm_extras = os.getenv("MEMT_USE_VLLM_EXTRAS", "1") != "0"

        if client is not None and "Async" not in client.__class__.__name__:
            self.client = client
        else:
            base_url = str(getattr(client, "base_url", None) or API_CONFIG_LOCAL["base_url"])
            api_key = getattr(client, "api_key", None) or API_CONFIG_LOCAL.get("api_key", "EMPTY")
            self.client = OpenAI(base_url=base_url, api_key=api_key)

    def _construct_messages(self, prompt_or_messages, system_prompt=None):
        if isinstance(prompt_or_messages, list):
            messages = list(prompt_or_messages)
            if system_prompt and not any(m.get("role") == "system" for m in messages):
                messages.insert(0, {"role": "system", "content": system_prompt})
            return messages
        if system_prompt:
            return [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": str(prompt_or_messages)},
            ]
        return [{"role": "user", "content": str(prompt_or_messages)}]

    def get_completion(
        self,
        prompt_or_messages,
        system_prompt: Optional[str] = None,
        json_mode: bool = False,
        stop: List[str] = [],
        max_retries: int = 3,
    ) -> str:
        messages = self._construct_messages(prompt_or_messages, system_prompt)
        kwargs = {
            "model": self.model_name,
            "messages": messages,
            "temperature": self.temperature,
            "max_tokens": self.max_tokens,
        }
        if stop:
            kwargs["stop"] = stop
        if self.use_vllm_extras:
            kwargs["extra_body"] = {
                "min_p": self.min_p,
                "chat_template_kwargs": {"enable_thinking": False},
            }
        if json_mode:
            kwargs["response_format"] = {"type": "json_object"}

        last_error = None
        for _ in range(max_retries + 1):
            try:
                response = self.client.chat.completions.create(**kwargs)
                content = response.choices[0].message.content or ""
                if "</think>" in content:
                    content = content.split("</think>")[-1].strip()
                if stop:
                    for token in stop:
                        if token in content:
                            content = content.split(token)[0]
                return content
            except Exception as exc:
                last_error = exc
                if "extra_body" in kwargs:
                    kwargs.pop("extra_body", None)
                    continue
                if json_mode and "response_format" in kwargs:
                    kwargs.pop("response_format", None)
                    continue
        return f"ERROR_API_CALL: {last_error}"


class MemTAgent(BaseAgent):
    """Adapter for the vendored Mem-T pipeline inside the unified evaluation API."""

    def __init__(
        self,
        *,
        client: Any,
        model_name: str = "Qwen/Qwen3-4B-Instruct-2507",
        top_k: Optional[int] = None,
        max_tool_steps: Optional[int] = None,
        db_type: Optional[str] = None,
        **_: Any,
    ):
        super().__init__(client=client, model_name=model_name)
        self.llm = _OpenAICompatibleMemTClient(client, model_name)
        self.top_k = int(top_k or os.getenv("MEMT_RETRIEVAL_TOPK", "5"))
        self.max_tool_steps = int(max_tool_steps or os.getenv("MEMT_MAX_TOOL_STEPS", "6"))
        self.db_type = db_type or os.getenv("MEMT_VECTOR_DB_TYPE", "persistent")
        self._run_id = uuid.uuid4().hex
        self._chunks: List[str] = []
        self._sample = None
        self._sample_id = "sample"
        self._benchmark_name = os.getenv("MEMT_BENCHMARK_NAME", "generic")
        self._question_category: Dict[str, Any] = {}
        self._built = False
        self._retriever = None
        self._builder = None
        self._config = None

    def reset(self) -> None:
        self._run_id = uuid.uuid4().hex
        self._chunks = []
        self._built = False
        self._retriever = None
        self._builder = None
        self._config = None

    def prepare_sample(self, sample) -> None:
        self._sample = sample
        task_id = getattr(sample, "task_id", "") or "sample"
        self._benchmark_name = os.getenv("MEMT_BENCHMARK_NAME", self._infer_benchmark_name(task_id))
        self._sample_id = task_id.removeprefix("locomo_") if self._benchmark_name == "locomo" else task_id
        self._question_category = {
            q.query: q.category
            for q in getattr(sample, "questions", [])
            if getattr(q, "query", None) is not None
        }

    async def add_memory_async(self, chunk: str) -> None:
        if chunk:
            self._chunks.append(chunk)
            self._built = False

    async def QA_batch_async(self, query_list: List[str], save_intermediate: bool = True) -> List[str]:
        if not query_list:
            return []
        await asyncio.to_thread(self._ensure_built)
        tasks = [asyncio.to_thread(self._answer_one, query) for query in query_list]
        return await asyncio.gather(*tasks)

    def _build_config(self):
        from external.memt.config import SystemConfig

        config = SystemConfig()
        config.data_name = self._benchmark_name
        config.USE_LOCAL_LLM = False
        config.USE_PARALLEL = False
        config.memory.retrieval_topk = self.top_k
        config.memory.update_retrieval_topk = int(os.getenv("MEMT_UPDATE_RETRIEVAL_TOPK", "3"))
        config.memory.max_tool_steps = self.max_tool_steps
        config.vector_db.db_type = self.db_type
        config.vector_db.embedding_model = os.getenv("MEMT_EMBEDDING_MODEL", config.vector_db.embedding_model)
        config.vector_db.from_scratch = True
        config.vector_db.path = str(PROJECT_ROOT / "tmp" / "memt_agent" / self._run_id / "db")
        config.traj_dir = str(PROJECT_ROOT / "tmp" / "memt_agent" / self._run_id / "traj")
        config.log_path = str(PROJECT_ROOT / "tmp" / "memt_agent" / self._run_id / "memt.log")
        Path(config.vector_db.path).mkdir(parents=True, exist_ok=True)
        Path(config.traj_dir).mkdir(parents=True, exist_ok=True)
        return config

    def _ensure_built(self) -> None:
        if self._built:
            return
        from external.memt.vector_db import VectorDBFactory
        from external.memt.memory_formation import MemoryFormation
        from external.memt.memory_update import MemoryUpdate
        from external.memt.memory_retrieval import MemoryRetriever
        from external.memt.memory_builder import MemoryBuilder

        self._config = self._build_config()
        vector_db = VectorDBFactory.create_db(self._config.vector_db)
        formation = MemoryFormation(llm_executor=self.llm)
        update = MemoryUpdate(llm_executor=self.llm, vector_db=vector_db)
        self._retriever = MemoryRetriever(vector_db=vector_db, llm_executor=self.llm, config=self._config)
        self._builder = MemoryBuilder(
            vector_db=vector_db,
            formation=formation,
            update=update,
            config=self._config,
        )
        memt_sample = self._to_memt_sample()
        self._builder.build_from_sample(memt_sample)
        self._built = True

    def _answer_one(self, query: str) -> str:
        category = self._question_category.get(query, "")
        result = self._retriever.retrieve_and_answer(
            query,
            sample_id=self._sample_id,
            category=str(category) if category is not None else "",
        )
        return str(result.get("answer", "")).strip()

    def _to_memt_sample(self) -> Dict[str, Any]:
        if self._benchmark_name == "locomo" or any(self._is_locomo_chunk(chunk) for chunk in self._chunks):
            return self._to_locomo_memt_sample()
        return self._to_generic_memt_sample()

    def _to_locomo_memt_sample(self) -> Dict[str, Any]:
        speaker_a, speaker_b = self._infer_speakers()
        conversation = []
        for idx, chunk in enumerate(self._chunks, start=1):
            session_time = self._extract_session_time(chunk)
            turns = self._parse_chunk_turns(chunk, idx)
            conversation.append({
                "sample_id": self._sample_id,
                "session_id": f"{self._sample_id}_conv_session_{idx}",
                "session_turns": turns,
                "metadata": {
                    "sample_id": self._sample_id,
                    "conversation_id": f"{self._sample_id}_conv",
                    "session_id": f"{self._sample_id}_conv_session_{idx}",
                    "session_time": session_time,
                    "speaker_a": speaker_a,
                    "speaker_b": speaker_b,
                },
            })
        return {"qa": [], "conversation": conversation}

    def _to_generic_memt_sample(self) -> Dict[str, Any]:
        conversation = []
        speaker_a = "context"
        speaker_b = "assistant"
        for idx, chunk in enumerate(self._chunks, start=1):
            text = str(chunk).strip()
            if not text:
                continue
            session_id = f"{self._sample_id}_session_{idx}"
            conversation.append({
                "sample_id": self._sample_id,
                "session_id": session_id,
                "session_turns": [{
                    "turn_id": f"{session_id}_turn_1",
                    "speaker": speaker_a,
                    "text": text,
                }],
                "metadata": {
                    "sample_id": self._sample_id,
                    "conversation_id": f"{self._sample_id}_conv",
                    "session_id": session_id,
                    "session_time": "",
                    "speaker_a": speaker_a,
                    "speaker_b": speaker_b,
                },
            })
        return {"qa": [], "conversation": conversation}

    def _infer_benchmark_name(self, task_id: str) -> str:
        if task_id.startswith("locomo_"):
            return "locomo"
        if task_id.startswith("longmemeval_"):
            return "longmemeval"
        if task_id.startswith("hotpotqa_"):
            return "hotpotqa"
        return "generic"

    def _is_locomo_chunk(self, chunk: str) -> bool:
        return (
            "Below is a conversation between " in chunk
            and "CONVERSATION:" in chunk
            and re.search(r'.+? said, ".*"', chunk) is not None
        )

    def _infer_speakers(self) -> tuple[str, str]:
        for chunk in self._chunks:
            match = re.search(r"Below is a conversation between (.*?) and (.*?)\.", chunk)
            if match:
                return match.group(1).strip(), match.group(2).strip()
        return "SpeakerA", "SpeakerB"

    def _extract_session_time(self, chunk: str) -> str:
        match = re.search(r"DATE:\s*(.*)", chunk)
        return match.group(1).strip() if match else ""

    def _parse_chunk_turns(self, chunk: str, session_idx: int) -> List[Dict[str, str]]:
        turns = []
        for line in chunk.splitlines():
            line = line.strip()
            match = re.match(r'(.+?) said, "(.*?)"(?: and shared (.*))?$', line)
            if not match:
                continue
            speaker, text = match.group(1).strip(), match.group(2).strip()
            caption = match.group(3)
            if caption:
                text = f"{text} Shared image caption: {caption.strip()}"
            turns.append({
                "turn_id": f"{self._sample_id}_conv_session_{session_idx}_D{session_idx}:{len(turns) + 1}",
                "speaker": speaker,
                "text": text,
            })
        return turns
