"""
A-mem Agent: Lightweight wrapper around external/A-mem evaluation code.

This integrates the A-mem AgenticMemorySystem with the local evaluation
framework and AsyncOpenAI client, using the async version of advancedMemAgent.
"""

from __future__ import annotations

import asyncio
import json
import logging
import sys
from pathlib import Path
from typing import List

from .base_agent import BaseAgent, MODEL_NAME_MAP

# Ensure external/A-mem is importable
A_MEM_PATH = Path(__file__).parents[1] / "external" / "A-mem"
if A_MEM_PATH.exists() and str(A_MEM_PATH) not in sys.path:
    sys.path.insert(0, str(A_MEM_PATH))

# Import A-mem core
import memory_layer as amem_layer
from test_advanced import advancedMemAgent

# Monkey patch missing dependency in A-mem (re is used but not imported)
if not hasattr(amem_layer, "re"):
    import re
    amem_layer.re = re

logger = logging.getLogger(__name__)


class AmemAgent(BaseAgent):
    """
    A-mem agent that wraps async advancedMemAgent.

    This delegates memory insertion and QA to advancedMemAgent using async methods.
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
        max_concurrent: int = 10,
    ):
        super().__init__(client, model_name)
        self.llm_model = MODEL_NAME_MAP.get(model_name, model_name)
        self.backend = backend
        self.retrieve_k = retrieve_k
        self.temperature_c5 = temperature_c5
        self.sglang_host = sglang_host
        self.sglang_port = sglang_port
        self.max_concurrent = max_concurrent
        self._semaphore = asyncio.Semaphore(max_concurrent)
        self._init_agent()

    def _init_agent(self) -> None:
        """Initialize advancedMemAgent and inject async client."""
        self._amem = advancedMemAgent(
            self.llm_model,
            self.backend,
            self.retrieve_k,
            self.temperature_c5,
            self.sglang_host,
            self.sglang_port,
        )
        # Inject the async client
        self._amem.retriever_llm.llm.client = self.client
        self._amem.memory_system.llm_controller.llm.client = self.client

    async def add_memory_async(self, chunk: str) -> None:
        """Add memory chunk to the system."""
        try:
            await self._amem.add_memory(chunk, None)
        except Exception as exc:
            logger.error(f"AmemAgent add_memory_async error: {exc}")

    async def QA_batch_async(self, query_list: List[str]) -> List[str]:
        """Answer a batch of questions using concurrent async calls with concurrency limit."""

        async def answer_single(query: str) -> str:
            """Answer a single question with semaphore control."""
            async with self._semaphore:  # Limit concurrent requests
                try:
                    response, _, _ = await self._amem.answer_question(
                        question=query,
                        category=1,  # Default category
                        answer="",
                    )

                    try:
                        parsed = json.loads(response)
                        return parsed.get("answer", "").strip()
                    except Exception:
                        return str(response).strip()

                except Exception as exc:
                    logger.error(f"Error answering query '{query}': {exc}")
                    return self._handle_api_error(exc, query)

        # Concurrently execute all queries with max_concurrent limit
        answers = await asyncio.gather(*[answer_single(query) for query in query_list])
        return list(answers)

    def reset(self) -> None:
        """Reset the agent by reinitializing."""
        try:
            self._init_agent()
            # Recreate semaphore after reset
            self._semaphore = asyncio.Semaphore(self.max_concurrent)
        except Exception as exc:
            logger.error(f"AmemAgent reset error: {exc}")

    def prepare_sample(self, sample) -> None:
        """Prepare sample for evaluation."""
        return None
