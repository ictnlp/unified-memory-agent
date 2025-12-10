from __future__ import annotations

import asyncio
import re
import sys
from pathlib import Path
from typing import Any, List, Optional, Sequence
from uuid import uuid4

from omegaconf import OmegaConf
from transformers import AutoTokenizer

from .base_agent import BaseAgent, MODEL_NAME_MAP

# Path constants
PROJECT_ROOT = Path(__file__).resolve().parents[1]  # unified-memory-agent root
VERL_ROOT = PROJECT_ROOT / "external" / "verl"      # verl in external/

# Import verl components (path managed by agents/__init__.py)
from verl.experimental.agent_loop.agent_loop import _DummyConfig  # type: ignore
from verl.experimental.agent_loop.tool_mem_agent_loop import ToolMemoryAgentLoop  # type: ignore
from verl.workers.rollout.replica import TokenOutput  # type: ignore


class ClientCompletionsServerManager:
    """Adapter that lets ToolMemoryAgentLoop use an OpenAI-compatible client."""

    def __init__(
        self,
        client: Any,
        model_name: str,
        tokenizer,
        *,
        strip_think_tag: bool = True,
    ) -> None:
        self._client = client
        self._model_name = model_name
        self._tokenizer = tokenizer
        self._strip_think_tag = strip_think_tag
        self._is_async = hasattr(client, "__class__") and "Async" in client.__class__.__name__
        self.calls = 0

    async def generate(
        self,
        request_id: str,
        prompt_ids: List[int],
        sampling_params: dict[str, Any],
        image_data: Any = None,
    ) -> TokenOutput:
        prompt_text = self._tokenizer.decode(prompt_ids, skip_special_tokens=False)
        max_tokens = int(sampling_params.get("max_tokens", 4096))
        temperature = float(sampling_params.get("temperature", 0.0))
        stop = sampling_params.get("stop", None)
        response = await self._create_completion(prompt_text, max_tokens, temperature, stop=stop)
        self.calls += 1

        choice = response[0]
        text = choice.get("text", "")
        if self._strip_think_tag and "</think>" in text:
            text = text.split("</think>")[-1].strip()

        token_ids = self._tokenizer.encode(text, add_special_tokens=False)
        supplied_logprobs = choice.get("logprobs") or []
        if not supplied_logprobs:
            supplied_logprobs = [0.0] * len(token_ids)
        else:
            supplied_logprobs = supplied_logprobs[: len(token_ids)]
            if len(supplied_logprobs) < len(token_ids):
                supplied_logprobs.extend([0.0] * (len(token_ids) - len(supplied_logprobs)))

        return TokenOutput(token_ids=token_ids, log_probs=supplied_logprobs)

    async def _create_completion(self, prompt: str, max_tokens: int, temperature: float, stop: list[str] | None = None) -> List[dict[str, Any]]:
        model = self._model_name
        response = await self._client.completions.create(
            model=model,
            prompt=prompt,
            max_tokens=max_tokens,
            temperature=temperature,
            stop=stop,
            logprobs=1,
            extra_body={"skip_special_tokens": False, "include_stop_str_in_output": True},
        )
        choice = response.choices[0]
        token_logprobs = []
        if getattr(choice, "logprobs", None) and getattr(choice.logprobs, "token_logprobs", None):
            token_logprobs = list(choice.logprobs.token_logprobs)
        return [
            {
                "text": choice.text,
                "logprobs": token_logprobs,
            }
        ]


class VerlMemoryAgent(BaseAgent):
    """Agent that wraps Verl's ToolMemoryAgentLoop for unified-memory-agent."""

    def __init__(
        self,
        *,
        client: Any = None,
        model_name: str = "gpt4.1",
        agent_config: Optional[dict[str, Any]] = None,
        data_source: str | None = None,
    ) -> None:
        if client is None:
            super().__init__(model_name=model_name)
        else:
            super().__init__(client=client, model_name=model_name)
        self._agent_config = agent_config or {}
        self._is_async = hasattr(self.client, "__class__") and "Async" in self.client.__class__.__name__
        self._resolved_model = MODEL_NAME_MAP.get(self.model_name, self.model_name)
        self._tokenizer = AutoTokenizer.from_pretrained(self._resolved_model, fix_mistral_regex=True)
        self._config = self._build_config(self._agent_config)
        ToolMemoryAgentLoop.init_class(self._config, self._tokenizer, None)
        self._memory_dir = Path(self._agent_config.get("memory_dir", PROJECT_ROOT / "tmp" / "verl_agent"))
        self._memory_dir.mkdir(parents=True, exist_ok=True)
        self._max_response_tokens = int(self._agent_config.get("response_length", self._config.actor_rollout_ref.rollout.response_length))
        self._max_chunk_size = int(
            self._agent_config.get(
                "max_chunk_size",
                self._config.actor_rollout_ref.rollout.multi_turn.get("max_chunk_size", 15000),
            )
        )
        self._context_chunks: List[str] = []
        self._context: str = ""
        self.data_source = data_source

    def _build_config(self, overrides: dict[str, Any]):
        tool_config_default = VERL_ROOT / "memagent" / "tool_config.yaml"
        tool_config_path = Path(overrides.get("tool_config_path", tool_config_default))
        config_dict = {
            "actor_rollout_ref": {
                "rollout": {
                    "prompt_length": overrides.get("prompt_length", 8192),
                    "response_length": overrides.get("response_length", 8192),
                    "multi_turn": {
                        "max_user_turns": overrides.get("max_user_turns", 10),
                        "max_assistant_turns": overrides.get("max_assistant_turns", 10),
                        "max_parallel_calls": overrides.get("max_parallel_calls", 10),
                        "max_tool_response_length": overrides.get("max_tool_response_length", 3000),
                        "tool_response_truncate_side": overrides.get("tool_response_truncate_side", "left"),
                        "tool_config_path": str(tool_config_path),
                        "format": overrides.get("format", "hermes"),
                        "max_chunk_size": overrides.get("max_chunk_size", 15000),
                    },
                },
            },
            "data": {},
        }
        return OmegaConf.create(config_dict)

    def reset(self) -> None:
        self._context_chunks = []
        self._context = ""

    def prepare_sample(self, sample) -> None:
        self.reset()

    def add_memory(self, chunk: str) -> None:
        if not chunk:
            return
        self._context_chunks.append(chunk)
        # Filter out empty strings from splitlines to avoid warnings in EmbeddingRetrievalTool
        # lines = [line for line in chunk.splitlines() if line.strip()]
        # self._context_chunks.extend(lines)
        self._context = "\n\n".join(self._context_chunks)

    async def add_memory_async(self, chunk: str) -> None:
        self.add_memory(chunk)

    async def QA_async(self, query: str) -> str:
        results = await self._run_agent_loop([query])
        return results[0] if results else ""

    def QA(self, query: str) -> str:
        if self._is_async:
            raise RuntimeError("Use QA_async() when the client is asynchronous")
        return asyncio.run(self.QA_async(query))

    def QA_batch(self, query_list: List[str], batch_size: int = 32) -> List[str]:
        if self._is_async:
            raise RuntimeError("Use QA_batch_async() when the client is asynchronous")
        return asyncio.run(self.QA_batch_async(query_list))

    async def QA_batch_async(self, query_list: List[str], save_intermediate: bool = False) -> List[str]:
        if not query_list:
            return []
        raw_result, outputs = await self._run_agent_loop(query_list)
        results = [str(resp).split("</tool_response>\nassistant")[-1].strip() for resp in raw_result]
        def try_remove_boxed(r: str) -> str:
            if not isinstance(r, str):
                return r
            # handle patterns like "\\boxed answer" (space) used occasionally
            if "\\boxed " in r:
                return r.split("\\boxed ")[-1].split("$")[0].strip()
            # fallback to the canonical \\boxed{...} or \box{...} braces
            def _extract_balanced(start_token: str, text: str) -> Optional[str]:
                idx = text.rfind(start_token)
                if idx == -1:
                    return None
                start = idx + len(start_token)
                depth = 1
                i = start
                while i < len(text):
                    ch = text[i]
                    if ch == "{":
                        depth += 1
                    elif ch == "}":
                        depth -= 1
                        if depth == 0:
                            return text[start:i].strip()
                    i += 1
                return None

            extracted = _extract_balanced("\\boxed{", r)
            if extracted is None:
                extracted = _extract_balanced("\\box{", r)
            if extracted:
                # unwrap simple LaTeX wrappers like \text{...}
                text_match = re.match(r"\\text\{(.+)\}$", extracted)
                if text_match:
                    return text_match.group(1).strip()
                return extracted.strip()
            return r[-2000:]  # return last 2000 chars as fallback
        results = [try_remove_boxed(r) for r in results]
        if save_intermediate:
            intermediate_path = Path(f"./tmp/intermediate_verl_outputs/{uuid4().hex}")
            intermediate_path.mkdir(parents=True, exist_ok=True)
            for i, output in enumerate(outputs):
                filename = intermediate_path / f"output_{i}.txt"
                with open(filename, "w") as f:
                    f.write(self._tokenizer.decode(output.prompt_ids))
        return results

    async def _run_agent_loop(self, queries: Sequence[str]) -> List[str]:
        server_manager = ClientCompletionsServerManager(
            self.client,
            self._resolved_model,
            self._tokenizer,
        )
        agent_loop = ToolMemoryAgentLoop(
            trainer_config=_DummyConfig(self._config),
            server_manager=server_manager,
            tokenizer=self._tokenizer,
            processor=None,
        )

        # memory_store_path = self._memory_dir / f"memory_store_{uuid4().hex}.jsonl"
        memory_store_path = self._memory_dir / f"memory_store.jsonl"
        sampling_params = {
            "temperature": 0.0,
            "max_tokens": self._max_response_tokens,
            "repetition_penalty": 1.05,
            # "stop": ["</tool_call>"],
        }

        try:
            outputs = await agent_loop.run(
                sampling_params=sampling_params,
                raw_prompt=[{"role": "user", "content": queries}],
                context=self._context,
                memory_kwargs={
                    "initial_memory": "",
                    "chunk_size": self._max_chunk_size,
                },
                tools_kwargs=self._build_tool_kwargs(memory_store_path),
                verbose=True,
                data_source=self.data_source,
            )
        finally:
            if memory_store_path.exists():
                try:
                    memory_store_path.unlink()
                except OSError:
                    pass

        response_list: List[str] = []
        for output in outputs:
            if getattr(output, "extra_fields", None) and output.extra_fields.get("is_final"):
                if getattr(output, "response_ids", None):
                    decoded = self._tokenizer.decode(output.response_ids, skip_special_tokens=True)
                    response_list.append(decoded)

        return response_list, outputs

    def _build_tool_kwargs(self, memory_store_path: Path) -> dict[str, Any]:
        filename = str(memory_store_path)
        return {
            "memory_add": {"create_kwargs": {"filename": filename}},
            "memory_update": {"create_kwargs": {"filename": filename}},
            "memory_delete": {"create_kwargs": {"filename": filename}},
            "memory_key_retrieve": {"create_kwargs": {"filename": filename}},
            "memory_list": {"create_kwargs": {"filename": filename}},
            "memory_bm25_retrieve": {"create_kwargs": {"chunks": list(self._context_chunks)}},
            "memory_embedding_retrieve": {"create_kwargs": {"chunks": list(self._context_chunks)}},
        }
