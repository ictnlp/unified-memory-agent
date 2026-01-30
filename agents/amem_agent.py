"""
A-mem Agent: Lightweight wrapper around external/A-mem evaluation code.

This integrates the A-mem AgenticMemorySystem with the local evaluation
framework and OpenAI/AsyncOpenAI client, reusing the original A-mem logic.
"""

from __future__ import annotations

import asyncio
import json
import logging
import sys
from pathlib import Path
from typing import List, Optional, Any, Dict

from .base_agent import BaseAgent, MODEL_NAME_MAP

# Ensure external/A-mem is importable
A_MEM_PATH = Path(__file__).parents[1] / "external" / "A-mem"
if A_MEM_PATH.exists() and str(A_MEM_PATH) not in sys.path:
    sys.path.insert(0, str(A_MEM_PATH))

# Import A-mem core
import memory_layer as amem_layer
from memory_layer import BaseLLMController
from test_advanced import advancedMemAgent

# Monkey patch missing dependency in A-mem (re is used but not imported)
if not hasattr(amem_layer, "re"):
    import re
    amem_layer.re = re

logger = logging.getLogger(__name__)


class CustomOpenAIController(BaseLLMController):
    """Adapter to use the caller-provided OpenAI/AsyncOpenAI client."""

    def __init__(self, client, model: str):
        self.client = client
        self.model = model
        self._is_async = hasattr(client, "__class__") and "Async" in client.__class__.__name__

    def _generate_empty_value(self, schema_type: str, schema_items: dict = None) -> Any:
        if schema_type == "array":
            return []
        if schema_type == "string":
            return ""
        if schema_type == "object":
            return {}
        if schema_type == "number" or schema_type == "integer":
            return 0
        if schema_type == "boolean":
            return False
        return None

    def _generate_empty_response(self, response_format: dict) -> dict:
        if not response_format or "json_schema" not in response_format:
            return {}
        schema = response_format["json_schema"].get("schema", {})
        result: Dict[str, Any] = {}
        for prop_name, prop_schema in schema.get("properties", {}).items():
            result[prop_name] = self._generate_empty_value(prop_schema.get("type"), prop_schema.get("items"))
        return result

    def _normalize_json_response(self, content: str, response_format: Optional[dict]) -> str:
        if not response_format or "json_schema" not in response_format:
            return content
        try:
            cleaned = content.strip()
            cleaned = cleaned.strip("`")
            if not cleaned.startswith("{"):
                start_idx = cleaned.find("{")
                if start_idx != -1:
                    cleaned = cleaned[start_idx:]
            if not cleaned.endswith("}"):
                end_idx = cleaned.rfind("}")
                if end_idx != -1:
                    cleaned = cleaned[: end_idx + 1]
            parsed = json.loads(cleaned)
            return json.dumps(parsed, ensure_ascii=False)
        except Exception:
            empty = self._generate_empty_response(response_format)
            return json.dumps(empty, ensure_ascii=False)

    def get_completion(self, prompt: str, response_format: dict = None,
                       temperature: float = 0.7, max_tokens: int = None) -> str:
        try:
            kwargs = {
                "model": self.model,
                "messages": [
                    {"role": "system", "content": "You must respond with a JSON object."},
                    {"role": "user", "content": prompt},
                ],
                "temperature": temperature,
            }
            if response_format:
                kwargs["response_format"] = response_format
            if max_tokens is not None:
                kwargs["max_tokens"] = max_tokens

            if self._is_async:
                import asyncio as _asyncio
                try:
                    loop = _asyncio.get_running_loop()
                except RuntimeError:
                    loop = _asyncio.new_event_loop()
                    _asyncio.set_event_loop(loop)
                    try:
                        response = loop.run_until_complete(
                            self.client.chat.completions.create(**kwargs)
                        )
                    finally:
                        loop.close()
                else:
                    raise RuntimeError("Unexpected running loop in thread context")
            else:
                response = self.client.chat.completions.create(**kwargs)

            content = response.choices[0].message.content
            return self._normalize_json_response(content, response_format)

        except Exception as exc:
            logger.error(f"CustomOpenAIController.get_completion error: {exc}")
            empty = self._generate_empty_response(response_format)
            return json.dumps(empty, ensure_ascii=False)


class AmemAgent(BaseAgent):
    """
    A-mem agent that directly reuses advancedMemAgent from the evaluation script.

    This delegates memory insertion and QA to advancedMemAgent.answer_question
    to stay consistent with the original A-mem evaluation behavior.
    """

    def __init__(
        self,
        client,
        model_name: str = "gpt4.1",
        backend: str = "openai",
        retrieve_k: int = 10,
        temperature_c5: float = 0.5,
        sglang_host: str = "http://localhost",
        sglang_port: int = 30000,
    ):
        super().__init__(client, model_name)
        self.llm_model = MODEL_NAME_MAP.get(model_name, model_name)
        self.backend = backend
        self.retrieve_k = retrieve_k
        self.temperature_c5 = temperature_c5
        self.sglang_host = sglang_host
        self.sglang_port = sglang_port
        self._init_agent()

    def _init_agent(self) -> None:
        self._amem = advancedMemAgent(
            self.llm_model,
            self.backend,
            self.retrieve_k,
            self.temperature_c5,
            self.sglang_host,
            self.sglang_port,
        )

    async def add_memory_async(self, chunk: str) -> None:
        try:
            await asyncio.to_thread(self._amem.add_memory, chunk, None)
        except Exception as exc:
            logger.error(f"AmemAgent add_memory_async error: {exc}")

    async def QA_batch_async(self, query_list: List[str]) -> List[str]:
        answers: List[str] = []
        for query in query_list:
            try:
                if isinstance(query, dict):
                    question = query.get("question", "")
                    category = int(query.get("category", 1))
                    answer = query.get("answer", "")
                else:
                    question = str(query)
                    category = 1
                    answer = ""
                response, _, _ = await asyncio.to_thread(
                    self._amem.answer_question,
                    question,
                    category,
                    answer,
                )
                try:
                    parsed = json.loads(response)
                    answers.append(parsed.get("answer", "").strip())
                except Exception:
                    answers.append(str(response).strip())
            except Exception as exc:
                logger.error(f"Error answering query: {exc}")
                answers.append(self._handle_api_error(exc, str(query)))
        return answers

    def reset(self) -> None:
        try:
            self._init_agent()
        except Exception as exc:
            logger.error(f"AmemAgent reset error: {exc}")

    def prepare_sample(self, sample) -> None:
        return None
