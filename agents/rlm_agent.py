from typing import List, Optional
from .base_agent import BaseAgent, MODEL_NAME_MAP
from pathlib import Path
from uuid import uuid4
import json
import asyncio
from rlm.rlm_repl import RLM_REPL


class RLMAgent(BaseAgent):
    """
    RLM (Recursive Language Model) agent with REPL environment for long context processing.

    This agent uses RLM's recursive approach to handle long contexts by building
    an interactive REPL environment where the model can explore the context incrementally.
    """

    def __init__(
        self,
        client,
        model_name: str = "gpt4.1",
        max_iterations: int = 10,
        enable_logging: bool = False,
    ):
        super().__init__(client, model_name)

        # Accumulate memory chunks for RLM context construction
        self.memory_chunks: List[str] = []
        self._rlm: Optional[RLM_REPL] = None
        self.model_name = MODEL_NAME_MAP.get(model_name, model_name)
        self.recursive_model = self.model_name
        self.max_iterations = max_iterations
        self.enable_logging = enable_logging

    async def add_memory_async(self, chunk: str) -> None:
        """
        Add a memory chunk to the accumulated context.

        Unlike other agents, RLM needs the complete context at QA time,
        so we store chunks and build context when answering.
        """
        self.memory_chunks.append(chunk)

    def _build_context(self) -> str:
        """
        Build the complete context string from accumulated memory chunks.
        """
        return "\n".join(self.memory_chunks)

    def _get_rlm(self) -> RLM_REPL:
        """
        Get or create the RLM instance.

        Uses singleton pattern within the agent's lifetime to maintain state.
        """
        if self._rlm is None:
            self._rlm = RLM_REPL(
                model=self.model_name,
                recursive_model=self.recursive_model,
                max_iterations=self.max_iterations,
                enable_logging=self.enable_logging,
                client=self.client,  # Pass external client
            )
        return self._rlm

    async def QA_batch_async(self, query_list: List[str], save_intermediate: bool=False) -> List[str]:
        """
        Answer a batch of questions using RLM with true async concurrency.

        Since RLM.completion is synchronous, we use asyncio.to_thread() to run
        each query in a separate thread, allowing true concurrent execution.

        Args:
            query_list: List of questions to answer.
            save_intermediate: Whether to save intermediate outputs.

        Returns:
            List of answers (and optional paths) in the same order as input queries.
        """
        context = self._build_context()

        # Create a separate RLM instance for each query to avoid state conflicts
        # Each query runs in its own thread with its own RLM instance
        async def process_single_query(query: str):
            """Process a single query in a thread"""
            # Create independent RLM instance for thread safety
            rlm = RLM_REPL(
                model=self.model_name,
                recursive_model=self.recursive_model,
                max_iterations=self.max_iterations,
                enable_logging=self.enable_logging,
                client=self.client,
            )
            # Run synchronous completion in thread pool
            return await asyncio.to_thread(rlm.completion, context=context, query=query)

        # Execute all queries concurrently using asyncio.gather
        results = await asyncio.gather(*[process_single_query(q) for q in query_list])

        # Process results and save intermediate outputs if requested
        responses = []
        paths = []
        intermediate_path = Path(f"./tmp/intermediate_rlm_outputs/{uuid4().hex}")
        intermediate_path.mkdir(parents=True, exist_ok=True)

        for i, output in enumerate(results):
            responses.append(output[0])
            filename = f"final_{i}.json"
            with open(intermediate_path / filename, "w") as f:
                json.dump(output[1], f, indent=4)
            paths.append(intermediate_path / filename)

        if save_intermediate:
            return responses, paths
        return responses

    def reset(self) -> None:
        """
        Reset the agent's internal state.

        Clears accumulated memory chunks and resets the RLM instance
        for a new evaluation sample.
        """
        self.memory_chunks = []
        if self._rlm is not None:
            self._rlm.reset()
        self._rlm = None
