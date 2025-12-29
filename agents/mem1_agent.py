"""
MEM1 Agent wrapper for unified-memory-agent evaluation framework.

This agent wraps the MEM1 system to work with the unified-memory-agent interface.
MEM1 uses a VLLM server for inference and implements a chunked memory construction
approach with internal state tracking via <think> tags.
"""

from __future__ import annotations

import asyncio
import importlib
import re
import sys
from pathlib import Path
from typing import TYPE_CHECKING, Any, List, Optional

from .base_agent import BaseAgent

# Import MEM1 components (path managed by agents/__init__.py)
if TYPE_CHECKING:  # pragma: no cover - static typing only
    from inference.data_pipelines import Mem1Pipeline  # type: ignore
else:
    try:
        pipelines_module = importlib.import_module("inference.data_pipelines")
        Mem1Pipeline = pipelines_module.Mem1Pipeline
    except ImportError as exc:  # pragma: no cover - import guard
        raise ImportError(
            "Failed to import MEM1 modules. Ensure external/mem1 is available "
            "and its dependencies are installed."
        ) from exc

DEFAULT_QA_TASK = "Remember all the information and be ready to answer questions."


class ClientWrapper:
    """Wrapper to make AsyncOpenAI client compatible with VLLMOpenAIClient interface."""

    def __init__(self, client: Any, tokenizer: Any, model_name: str):
        self.client = client
        self.tokenizer = tokenizer
        self.model_name = model_name

    def reset(self):
        """Reset method for compatibility."""
        pass

    def make_completion(self, initial_prompt: str, content: str, model: str = None,
                       temperature: float = 0.01, force_json: bool = False, is_last_turn: bool = False) -> str:
        """
        Synchronous completion compatible with VLLMOpenAIClient.make_completion.

        This mimics the behavior of VLLMOpenAIClient by:
        1. Building messages with initial_prompt and content
        2. Using tokenizer.apply_chat_template
        3. Calling completions API
        """
        return asyncio.run(self._make_completion_async(initial_prompt, content, model, temperature, is_last_turn))

    async def _make_completion_async(self, initial_prompt: str, content: str, model: str,
                                    temperature: float, is_last_turn: bool) -> str:
        """Async implementation."""
        # Build messages
        messages = [
            {"role": "user", "content": initial_prompt},
            {"role": "assistant", "content": content}
        ]

        # Apply chat template
        prompt = self.tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)

        # Remove trailing marker
        if prompt.endswith("<|im_end|>\n"):
            prompt = prompt[:-len("<|im_end|>\n")]

        # Prepare stop tokens
        stop = ["</search>", "</answer>"] if not is_last_turn else ["</answer>"]

        # Call completions API
        response = await self.client.completions.create(
            model=model or self.model_name,
            prompt=prompt,
            temperature=temperature,
            max_tokens=1024,
            stop=stop
        )

        content = response.choices[0].text.strip()

        # Add back stop token if needed
        if hasattr(response.choices[0], 'finish_reason'):
            finish_reason = response.choices[0].finish_reason
            if finish_reason == "stop" and hasattr(response.choices[0], 'stop_reason'):
                stop_reason = response.choices[0].stop_reason
                if stop_reason == "</search>":
                    content += "</search>"
                elif stop_reason == "</answer>":
                    content += "</answer>"

        return content


class Mem1Agent(BaseAgent):
    """
    MEM1 Agent wrapper that integrates MEM1's memory construction approach
    with the unified-memory-agent evaluation framework.

    MEM1 constructs memory by:
    1. Accumulating chunks
    2. Processing all chunks through MEM1 pipeline on first QA call
    3. Extracting internal state from <think> tags to maintain cumulative memory
    4. Using the accumulated memory to answer questions
    """

    def __init__(
        self,
        client: Optional[Any] = None,
        model_name: str = "Qwen/Qwen3-4B-Instruct-2507",
        qa_task: str = DEFAULT_QA_TASK,
        temperature: float = 0.01,
    ) -> None:
        """
        Initialize MEM1 agent.

        Args:
            client: AsyncOpenAI-compatible client for memory construction (required)
            model_name: Model name for API calls
            qa_task: Task description for QA
            temperature: Sampling temperature
        """
        super().__init__(client, model_name)

        self.qa_task = qa_task
        self.temperature = temperature
        self.qa_model_name = "Qwen/Qwen3-4B-Instruct-2507"

        # Load tokenizer (needed for wrapper)
        try:
            from transformers import AutoTokenizer
            self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        except Exception as e:
            breakpoint()
            raise RuntimeError(f"Failed to load tokenizer for {model_name}: {e}")

        # Wrap the provided client to match VLLMOpenAIClient interface
        if not client:
            raise ValueError("client parameter is required for Mem1Agent")

        print(f"Using provided client wrapped for MEM1 memory construction")
        self.mem1_client = ClientWrapper(client, self.tokenizer, model_name)

        # Initialize MEM1 pipeline
        self.pipeline = Mem1Pipeline(self.mem1_client, inference_type="mem1")

        # Create QA client for question answering (port 8001)
        from openai import AsyncOpenAI
        self.qa_client = AsyncOpenAI(
            base_url="http://localhost:8001/v1",
            api_key="EMPTY"
        )
        print(f"Using QA client on port 8001 for question answering")

        # Runtime state
        self._chunks: List[str] = []
        self._current_memory: str = ""
        self._memory_built: bool = False

    def reset(self) -> None:
        """Reset agent state between samples."""
        self._chunks = []
        self._current_memory = ""
        self._memory_built = False
        if hasattr(self.mem1_client, 'reset'):
            self.mem1_client.reset()

    async def add_memory_async(self, chunk: str) -> None:
        """
        Add a memory chunk to the buffer.

        Chunks are accumulated and processed together when QA is first called.
        This follows MEM1's original design of batch processing all chunks.

        Args:
            chunk: Text chunk to add to memory
        """
        if not chunk:
            return
        self._chunks.append(chunk)
        # Mark that memory needs to be rebuilt
        self._memory_built = False

    async def QA_batch_async(self, query_list: List[str]) -> List[str]:
        """
        Answer multiple questions based on accumulated memory.

        Args:
            query_list: List of questions

        Returns:
            List of answers
        """
        # Build memory once at the beginning for all queries
        if not self._memory_built and self._chunks:
            await self._build_memory_async()

        # Process all queries (memory already built, won't rebuild)
        results = []
        for query in query_list:
            try:
                # Extract text from <think> tags if present
                memory_text = self._current_memory
                if "<think>" in memory_text:
                    memory_text = memory_text.replace("<think>", "").replace("</think>", "").strip()

                if not memory_text:
                    print("Warning: No memory built, cannot answer question")

                # MEM1 style prompt: task + memory + question
                qa_prompt = f"""{self.qa_task}

{memory_text}

{query}"""

                # Call qa_client
                response = await self.qa_client.chat.completions.create(
                    model=self.qa_model_name,
                    messages=[{"role": "user", "content": qa_prompt}],
                    temperature=self.temperature,
                    max_tokens=2048
                )
                answer = response.choices[0].message.content

                # Clean up response
                answer = answer.strip()
                if answer.startswith("Error:"):
                    answer = f"ERROR_MEM1_QA: {answer}"
                # Handle thinking tags if present in answer
                elif "</think>" in answer:
                    answer = answer.split("</think>")[-1].strip()
                elif "<think>" in answer and "</think>" not in answer:
                    answer = f"ERROR_THINK_LENGTH_EXCEEDED: The think is too long. Think: {answer[:100]}..."

                # Extract answer from tags if present
                if "<answer>" in answer and "</answer>" in answer:
                    extracted = re.findall(r'<answer>(.*?)</answer>', answer, re.DOTALL)
                    if extracted:
                        answer = extracted[0].strip()

                results.append(answer)
            except Exception as e:
                results.append(self._handle_api_error(e, query))
        return results

    async def _build_memory_async(self) -> None:
        """Build memory using MEM1 pipeline (runs in thread pool)."""
        if self._memory_built or not self._chunks:
            return

        # Run pipeline in thread pool since it's synchronous
        self._current_memory = await asyncio.to_thread(
            self.pipeline.run_llm_loop,
            self.qa_task,
            self._chunks,
            self.model_name
        )
        self._memory_built = True

