from __future__ import annotations

import asyncio
import importlib
import os
import sys
from pathlib import Path
from typing import List, Optional

import yaml

# Import Mem-alpha components (path managed by agents/__init__.py)
from agent import MemoryAgent as RawMemoryAgent  # type: ignore
from memory import Memory  # type: ignore

DEFAULT_AGENT_CONFIG = {
    "agent_name": "memalpha_qwen_agent",
    "model_name": "Qwen/Qwen3-4B-Instruct",
    "enable_thinking": False,
    "vllm": False,
    "thinking_budget": 1024,
    "max_new_tokens": 2048,
    "infer_with_full_memory": False,
    "external_model_url": None,
    "api_key": None,
    "include_conversation_history": True,
}

class MemAlphaUnifiedAgent:
    """Adapter that exposes Mem-alpha's MemoryAgent through the unified-memory-agent API."""

    def __init__(
        self,
        model_name: str = "YuWangX/Memalpha-4B",
        client=None,
        agent_config_path: Optional[str] = None,
        prompts_config_path: Optional[str] = None,
        agent_config: Optional[dict] = None,
    ) -> None:
        self.client = client

        config, resolved_config_path = self._load_agent_config(agent_config_path, agent_config)
        self._agent_overrides = dict(agent_config or {})
        self._agent_config_path = resolved_config_path
        resolved_prompts_path = prompts_config_path or os.environ.get("MEM_ALPHA_PROMPTS_CONFIG")
        if not resolved_prompts_path:
            raise ValueError(
                "Mem-alpha prompt config path not provided. Pass prompts_config_path or set "
                "MEM_ALPHA_PROMPTS_CONFIG."
            )

        prompts_path = Path(resolved_prompts_path)
        self._prompts_config_path = str(prompts_path)

        if not prompts_path.exists():
            raise FileNotFoundError(
                f"Mem-alpha prompt config not found: {prompts_path}. "
                "Set MEM_ALPHA_PROMPTS_CONFIG or provide prompts_config_path."
            )

        with open(prompts_path, "r", encoding="utf-8") as f:
            prompts = yaml.safe_load(f)

        self._unified_prompt: str = prompts.get("unified_prompt", "{context}")
        self._prompt_metadata: dict = {k: v for k, v in prompts.items() if k != "unified_prompt"}

        # Instantiate the underlying Mem-alpha agent (loads tokenizer/model once)
        self._agent = RawMemoryAgent(agent_config=config, save_process=False, client=client, model_name=model_name)
        self._max_new_tokens = config.get("max_new_tokens", self._agent.MAX_NEW_TOKENS)

        # Runtime state
        self.current_memory: Memory = Memory(including_core=True)
        self._agent.memory = self.current_memory
        self.current_data_source: Optional[str] = None
        self.current_query_prompt: Optional[str] = None
        self._chunk_counter = 0

    # ------------------------------------------------------------------
    # Framework hooks
    # ------------------------------------------------------------------
    def reset(self) -> None:
        """Reset agent state between samples."""
        self.current_memory = Memory(including_core=True)
        self._agent.memory = self.current_memory
        self._reset_conversation_history()
        self._agent.step = 0
        self.current_data_source = None
        self.current_query_prompt = None
        self._chunk_counter = 0

    def prepare_sample(self, sample) -> None:
        """Configure agent for a new sample using dataset metadata."""
        data_source = self._infer_data_source(sample)
        including_core = self._should_include_core(data_source)
        self.current_query_prompt = self._prompt_metadata.get(data_source, {}).get("query_prompt")

        self.current_data_source = data_source
        self.current_memory = Memory(including_core=including_core)
        self._agent.memory = self.current_memory
        self._reset_conversation_history()
        self._agent.step = 0
        self._chunk_counter = 0

    # ------------------------------------------------------------------
    # Memory ingestion
    # ------------------------------------------------------------------
    async def add_memory_async(self, chunk: str) -> None:
        if not chunk:
            return
        formatted_chunk = self._format_chunk(chunk)
        self._agent.memory = self.current_memory
        self._reset_conversation_history()

        try:
            await self._agent.chat(user_msg=formatted_chunk, status="memorie")
        except Exception as exc:  # pragma: no cover - runtime dependent
            raise RuntimeError(f"Mem-alpha memory ingestion failed: {exc}") from exc

        self.current_memory = self._agent.memory
        self._chunk_counter += 1

    # ------------------------------------------------------------------
    # Question answering
    # ------------------------------------------------------------------

    async def QA_batch_async(self, query_list: List[str]) -> List[str]:
        responses: List[str] = []
        for query in query_list:
            formatted_query = self._format_query(query)
            self._agent.memory = self.current_memory
            self._reset_conversation_history()

            try:
                answer = await self._agent.chat(user_msg=formatted_query, status="chat")
            except Exception as exc:  # pragma: no cover - runtime dependent
                answer = self._handle_api_error(exc, query)

            if isinstance(answer, tuple):  # chat may return (content, step_info)
                answer = answer[0]
            responses.append(str(answer).strip())
        responses = [resp.split("</think>")[-1].strip() for resp in responses]
        return responses

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------
    def _load_agent_config(
        self,
        agent_config_path: Optional[str],
        agent_config: Optional[dict],
    ) -> tuple[dict, Optional[str]]:
        config = DEFAULT_AGENT_CONFIG.copy()
        resolved_path: Optional[str] = None

        candidate_path = agent_config_path or os.environ.get("MEM_ALPHA_AGENT_CONFIG")
        if candidate_path:
            path = Path(candidate_path)
            if not path.exists():
                raise FileNotFoundError(
                    f"Mem-alpha agent config not found: {candidate_path}"
                )
            with open(path, "r", encoding="utf-8") as f:
                loaded = yaml.safe_load(f) or {}
            config.update(loaded)
            resolved_path = str(path)

        if agent_config:
            config.update(agent_config)

        config.setdefault("max_new_tokens", DEFAULT_AGENT_CONFIG["max_new_tokens"])
        return config, resolved_path

    def _format_chunk(self, chunk: str) -> str:
        prompt = self._unified_prompt
        return prompt.format(context=chunk, max_new_tokens=self._max_new_tokens)

    def _format_query(self, query: str) -> str:
        if self.current_query_prompt:
            return f"{self.current_query_prompt}\n\n{query}"
        return query

    def _reset_conversation_history(self) -> None:
        if hasattr(self._agent, "conversation_history"):
            self._agent.conversation_history = []

    def _infer_data_source(self, sample) -> str:
        if hasattr(sample, "questions"):
            for question in sample.questions:
                category = getattr(question, "category", None)
                if category:
                    return str(category)
        return "memalpha"

    def _should_include_core(self, data_source: str) -> bool:
        metadata = self._prompt_metadata.get(data_source)
        if isinstance(metadata, dict):
            return bool(metadata.get("including_core", True))
        return True