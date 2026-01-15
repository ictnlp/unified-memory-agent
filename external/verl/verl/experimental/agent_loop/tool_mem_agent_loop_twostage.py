# Copyright 2025 Bytedance Ltd. and/or its affiliates
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""
Two-Stage Tool Memory Agent Loop

This module implements a two-stage training approach for memory agents:
- Stage 1 (training_stage="memory"): Train memory formation, fix QA model
- Stage 2 (training_stage="qa"): Fix memory formation, train QA model

The fixed model is called via an external vLLM API endpoint.
"""

import asyncio
import logging
import os
from typing import Any, List, Optional

from openai import AsyncOpenAI

from verl.experimental.agent_loop.agent_loop import AgentLoopOutput, register
from verl.experimental.agent_loop.tool_mem_agent_loop import (
    ToolMemoryAgentLoop,
    AgentData,
    AgentState,
    TrajectoryConversation,
)
from verl.utils.profiler import simple_timer

logger = logging.getLogger(__file__)
logger.setLevel(os.getenv("VERL_LOGGING_LEVEL", "WARN"))


# ============================================================================
# Fixed Model Server Manager (Internal to Two-Stage Training)
# ============================================================================

class FixedModelOutput:
    """Output format compatible with VERL's server manager interface"""

    def __init__(self, token_ids: List[int], log_probs: Optional[List[float]] = None):
        self.token_ids = token_ids
        self.log_probs = log_probs


class FixedModelServerManager:
    """
    Wrapper for fixed model API using OpenAI AsyncClient

    This manager provides the same interface as VERL's server_manager but calls
    an external API instead of managing a local model. It's used for two-stage
    training where one model is fixed.

    Args:
        api_base: Base URL for the API (e.g., "http://localhost:8000/v1" or "https://api.openai.com/v1")
        model_name: Model name to pass in API requests
        tokenizer: Tokenizer for encoding/decoding (required)
        timeout: Request timeout in seconds (default: 300)
        api_key: API key for authentication (optional, use "EMPTY" for vLLM)
    """

    def __init__(
        self,
        api_base: str,
        model_name: str = "default",
        tokenizer=None,
        timeout: float = 300.0,
        api_key: str = None,
    ):
        self.api_base = api_base.rstrip("/")
        self.model_name = model_name
        self.tokenizer = tokenizer
        self.timeout = timeout

        if not self.tokenizer:
            raise ValueError("Tokenizer is required for FixedModelServerManager")

        # Initialize OpenAI AsyncClient
        # For vLLM, use api_key="EMPTY" or None
        self.client = AsyncOpenAI(
            base_url=self.api_base,
            api_key=api_key or "EMPTY",
            timeout=timeout,
            max_retries=3,  # Built-in retry mechanism
        )

        logger.info(f"Initialized FixedModelServerManager with AsyncOpenAI: api_base={api_base}, model={model_name}")

    async def generate(
        self,
        request_id: str,
        prompt_ids: List[int],
        sampling_params: dict[str, Any],
    ) -> FixedModelOutput:
        """
        Generate response from fixed model via OpenAI API

        Args:
            request_id: Unique identifier for this request
            prompt_ids: List of token IDs as prompt
            sampling_params: Sampling parameters (temperature, top_p, etc.)

        Returns:
            FixedModelOutput with token_ids and log_probs=None
        """
        # Convert token IDs to text using tokenizer
        prompt_text = self.tokenizer.decode(prompt_ids, skip_special_tokens=False)

        try:
            # Call OpenAI chat completions API
            # This works with both OpenAI and vLLM (vLLM supports chat completions)
            response = await self.client.chat.completions.create(
                model=self.model_name,
                messages=[{"role": "user", "content": prompt_text}],
                temperature=sampling_params.get("temperature", 0.7),
                top_p=sampling_params.get("top_p", 1.0),
                max_tokens=sampling_params.get("max_tokens", 2048),
            )

            # Extract generated text
            generated_text = response.choices[0].message.content

            # Handle empty or None response
            if not generated_text:
                logger.warning(f"Fixed model returned empty response for request {request_id}, using fallback")
                # Use a space as fallback to ensure at least one token
                generated_text = " "

            # Encode back to token IDs
            response_ids = self.tokenizer.encode(generated_text, add_special_tokens=False)

            # Ensure response_ids is a list and not empty
            if not isinstance(response_ids, list):
                response_ids = list(response_ids)

            # Final safety check: if still empty after encoding, use pad token
            if len(response_ids) == 0:
                logger.error(f"Fixed model encode resulted in empty token list for request {request_id}")
                response_ids = [self.tokenizer.pad_token_id if self.tokenizer.pad_token_id else 0]

            logger.debug(f"Fixed model generated {len(response_ids)} tokens for request {request_id}")

            # Return output with log_probs=None (not used for fixed model)
            # IMPORTANT: Return tuple instead of list to match SGLang server behavior
            return FixedModelOutput(token_ids=tuple(response_ids), log_probs=None)

        except Exception as e:
            logger.error(f"Error calling fixed model API (request_id={request_id}): {type(e).__name__}: {e}")
            raise

    async def close(self):
        """Close the OpenAI client"""
        await self.client.close()
        logger.info("Closed FixedModelServerManager OpenAI client")

    def __del__(self):
        """Cleanup on deletion"""
        try:
            import asyncio
            loop = asyncio.get_event_loop()
            if loop.is_running():
                loop.create_task(self.close())
            else:
                loop.run_until_complete(self.close())
        except Exception:
            pass  # Best effort cleanup


# ============================================================================
# Two-Stage Training Components
# ============================================================================


class TwoStageTrajectoryConversation(TrajectoryConversation):
    """Extended TrajectoryConversation with is_trainable flag"""

    def __init__(self, *args, is_trainable: bool = True, **kwargs):
        super().__init__(*args, **kwargs)
        self.is_trainable = is_trainable


class TwoStageAgentData(AgentData):
    """Extended AgentData with stage tracking"""

    def add_conversation(self, is_final: int = 0, is_trainable: bool = True):
        """Add a conversation with trainability flag"""
        response_token_count = len(self.current_response_mask)
        prompt_token_count = max(len(self.current_prompt_ids) - response_token_count, 0)
        prompt_ids = self.current_prompt_ids[:prompt_token_count]

        conversation = TwoStageTrajectoryConversation(
            prompt_ids=prompt_ids,
            response_ids=self.current_response_ids.copy(),
            response_mask=self.current_response_mask.copy(),
            response_logprobs=self.current_response_logprobs.copy(),
            is_final=is_final,
            user_turns=self.user_turns,
            assistant_turns=self.assistant_turns,
            tool_rewards=self.current_tool_rewards,
            num_tools=self.current_num_tools,
            is_trainable=is_trainable,
        )
        self.conversations.append(conversation)

        # Reset current state for next conversation
        self.current_prompt_ids = []
        self.current_response_ids = []
        self.current_response_mask = []
        self.current_response_logprobs = []
        self.user_turns = 0
        self.assistant_turns = 0
        self.current_tool_rewards = []
        self.current_num_tools = 0


@register("tool_mem_agent_twostage")
class TwoStageMemoryAgentLoop(ToolMemoryAgentLoop):
    """
    Two-stage training variant of ToolMemoryAgentLoop

    Configuration:
        training_stage: "memory" or "qa"
            - "memory": Train memory formation, use fixed QA model
            - "qa": Use fixed memory formation, train QA model

        fixed_model_api: URL of the fixed model vLLM endpoint
            e.g., "http://localhost:8000/v1"

    Example config:
        actor_rollout_ref:
          rollout:
            agent_name_override: tool_mem_agent_twostage
            training_stage: memory
            fixed_model_api: http://localhost:8000/v1
    """

    @classmethod
    def init_class(cls, config, tokenizer, processor, **kwargs):
        # Call parent initialization first
        super().init_class(config, tokenizer, processor, **kwargs)

        # Two-stage specific configuration
        rollout_config = config.actor_rollout_ref.rollout
        cls.training_stage = rollout_config.get("training_stage", "memory")
        cls.fixed_model_api = rollout_config.get("fixed_model_api", None)

        # Validate configuration
        if cls.training_stage not in ["memory", "qa"]:
            raise ValueError(f"training_stage must be 'memory' or 'qa', got: {cls.training_stage}")

        if not cls.fixed_model_api:
            raise ValueError("fixed_model_api is required for two-stage training")

        # Initialize fixed model server manager
        cls.fixed_server_manager = FixedModelServerManager(
            api_base=cls.fixed_model_api,
            model_name="Qwen/Qwen3-4B-Instruct-2507",
            tokenizer=tokenizer,
        )

        logger.info(f"✅ TwoStageMemoryAgentLoop initialized:")
        logger.info(f"   - Training stage: {cls.training_stage}")
        logger.info(f"   - Fixed model API: {cls.fixed_model_api}")
        logger.info(f"   - Trainable: {'Memory formation' if cls.training_stage == 'memory' else 'QA answering'}")

    async def _process_memory_chunk(self, agent_data: AgentData, sampling_params: dict[str, Any], verbose: bool, trajectory_id: str, data_source: str):
        """Process a single memory chunk - mark trainability based on stage"""
        # Prepare memory update prompt
        current_chunk = agent_data.context_chunks[agent_data.current_chunk_idx]
        agent_data.current_prompt_ids = await self._build_memory_prompt_ids(
            agent_data.messages, agent_data.memory_content, current_chunk, data_source
        )

        if len(agent_data.current_prompt_ids) > self.prompt_length + self.response_length:
            print(f"WARNING: Memory prompt length {len(agent_data.current_prompt_ids)} exceeds max prompt length {self.prompt_length + self.response_length}")

        # Multi-turn generation loop for this chunk
        state = AgentState.GENERATING
        while state != AgentState.TERMINATED:
            if state == AgentState.GENERATING:
                # Pass stage="memory" to use appropriate model
                state = await self._handle_generating_state_with_stage(
                    agent_data, sampling_params, stage="memory"
                )
            elif state == AgentState.PROCESSING_TOOLS:
                state = await self._handle_processing_tools_state(agent_data, trajectory_id)

            # Check termination conditions for this conversation
            if len(agent_data.current_response_mask) >= self.response_length:
                state = AgentState.TERMINATED

        await self._update_memory_content(agent_data, verbose, trajectory_id)

        # Mark trainability: memory stage conversations are trainable only in "memory" training stage
        is_trainable = (self.training_stage == "memory")
        agent_data.add_conversation(is_final=0, is_trainable=is_trainable)
        agent_data.memory_turns += 1

    async def _process_final_response(self, agent_data: AgentData, sampling_params: dict[str, Any], verbose: bool, trajectory_id: str, data_source: str):
        """Generate final response - mark trainability based on stage"""
        final_prompts = await self._build_final_prompt_ids(agent_data.messages, agent_data.memory_content, data_source)
        sem = asyncio.Semaphore(10)

        async def process_one_final_prompt(idx, final_prompt):
            async with sem:
                local_agent_data = TwoStageAgentData(
                    messages=agent_data.messages,
                    metrics=agent_data.metrics,
                    request_id=f"{agent_data.request_id}_final_{idx}",
                    memory_kwargs=agent_data.memory_kwargs,
                )
                local_agent_data.memory_content = agent_data.memory_content
                local_agent_data.current_prompt_ids = final_prompt

                if len(local_agent_data.current_prompt_ids) > self.prompt_length + self.response_length:
                    print(f"WARNING: Final prompt length {len(local_agent_data.current_prompt_ids)} exceeds max prompt length {self.prompt_length + self.response_length}")

                # Multi-turn generation loop for final response
                state = AgentState.GENERATING
                while state != AgentState.TERMINATED:
                    if state == AgentState.GENERATING:
                        # Pass stage="qa" to use appropriate model
                        state = await self._handle_generating_state_with_stage(
                            local_agent_data, sampling_params, stage="qa"
                        )
                    elif state == AgentState.PROCESSING_TOOLS:
                        state = await self._handle_processing_tools_state(local_agent_data, trajectory_id)

                    # Check termination conditions
                    if len(local_agent_data.current_response_mask) >= self.response_length:
                        state = AgentState.TERMINATED

                if verbose:
                    response_text = await asyncio.to_thread(
                        self.tokenizer.decode, local_agent_data.current_response_ids, skip_special_tokens=False
                    )
                    print(f"Final Answer:\n{response_text}")

                # Mark trainability: final response is trainable only in "qa" training stage
                is_trainable = (self.training_stage == "qa")
                local_agent_data.add_conversation(is_final=idx + 1, is_trainable=is_trainable)
                return local_agent_data.conversations

        tasks = []
        for idx, final_prompt in enumerate(final_prompts):
            tasks.append(process_one_final_prompt(idx, final_prompt))

        results_lists = await asyncio.gather(*tasks)
        for result_conversations in results_lists:
            agent_data.conversations.extend(result_conversations)

    async def _handle_generating_state_with_stage(
        self, agent_data: AgentData, sampling_params: dict[str, Any], stage: str
    ) -> AgentState:
        """
        Handle generating state with stage-aware model selection

        Args:
            agent_data: Agent data container
            sampling_params: Sampling parameters
            stage: "memory" or "qa" - indicates which stage this generation is for

        Returns:
            Next agent state
        """
        # Determine which model to use based on training stage
        use_trainable_model = (stage == self.training_stage)

        max_model_len = self.prompt_length + self.response_length
        max_new_tokens = min(self.response_length, max_model_len - len(agent_data.current_prompt_ids) - 1)

        if max_new_tokens <= 0:
            print(f"ERROR: Current len(agent_data.current_prompt_ids)={len(agent_data.current_prompt_ids)} exceeds model max length {max_model_len}")

        with simple_timer("generate_sequences", agent_data.metrics):
            if use_trainable_model:
                # Use trainable model (requires log_probs for training)
                logger.debug(f"Using TRAINABLE model for {stage} stage")
                output = await self.server_manager.generate(
                    request_id=agent_data.request_id,
                    prompt_ids=agent_data.current_prompt_ids,
                    sampling_params=sampling_params,
                )
                response_ids = output.token_ids
                response_logprobs = output.log_probs if output.log_probs else [0.0] * len(response_ids)
            else:
                # Use fixed model (no log_probs needed)
                logger.debug(f"Using FIXED model for {stage} stage")
                output = await self.fixed_server_manager.generate(
                    request_id=agent_data.request_id,
                    prompt_ids=agent_data.current_prompt_ids,
                    sampling_params=sampling_params,
                )
                response_ids = output.token_ids
                response_logprobs = [0.0] * len(response_ids)  # Dummy log_probs

        # Extract tool calls (same logic as parent class)
        agent_data.tool_calls = []
        agent_data.failed_tool_calls = []

        if self.tool_parser:
            response_text, agent_data.tool_calls, num_tools, agent_data.failed_tool_calls = \
                await self.tool_parser.extract_tool_calls(response_ids, return_failed_and_num_tools=True)

            actual_start_count = response_text.count(self.tool_parser.tool_call_start_token)
            actual_end_count = response_text.count(self.tool_parser.tool_call_end_token)
            increment = num_tools + actual_start_count + actual_end_count

            if increment < 0:
                logger.error("=" * 80)
                logger.error(f"❌ NEGATIVE INCREMENT DETECTED: {increment}")
                logger.error(f"   num_tools: {num_tools}, start_count: {actual_start_count}, end_count: {actual_end_count}")
                logger.error("=" * 80)

            agent_data.current_num_tools += increment

            # Handle tool call truncation if needed (same as parent)
            if num_tools:
                tool_start_ids = self.tokenizer.encode(self.tool_parser.tool_call_start_token, add_special_tokens=False)
                tool_end_ids = self.tokenizer.encode(self.tool_parser.tool_call_end_token, add_special_tokens=False)

                def remove_last_tool_call():
                    def find_last_sequence(response, sequence):
                        seq_len = len(sequence)
                        for i in range(len(response) - seq_len, -1, -1):
                            if response[i:i + seq_len] == sequence:
                                return i
                        return -1

                    end_idx = find_last_sequence(response_ids, tool_end_ids)
                    if end_idx == -1:
                        return
                    start_idx = find_last_sequence(response_ids[:end_idx], tool_start_ids)
                    if start_idx != -1 and start_idx < end_idx:
                        del response_ids[start_idx:end_idx + len(tool_end_ids)]

                while len(agent_data.current_response_ids) + len(response_ids) > self.response_length - 500 and num_tools > 0:
                    remove_last_tool_call()
                    num_tools -= 1
                    if agent_data.failed_tool_calls and agent_data.failed_tool_calls[-1][0] == num_tools:
                        agent_data.failed_tool_calls.pop()
                    else:
                        agent_data.tool_calls.pop()

                if len(agent_data.current_response_ids) + len(response_ids) > self.response_length - 500:
                    response_ids = response_ids[:self.response_length - len(agent_data.current_response_ids) - 500 - 1] + response_ids[-1:]

        agent_data.assistant_turns += 1
        agent_data.current_prompt_ids += response_ids
        agent_data.current_response_ids += response_ids
        agent_data.current_response_mask += [1] * len(response_ids)
        agent_data.current_response_logprobs += response_logprobs

        # Determine next state
        if agent_data.tool_calls or agent_data.failed_tool_calls:
            return AgentState.PROCESSING_TOOLS
        else:
            return AgentState.TERMINATED

    async def run(self, sampling_params: dict[str, Any], **kwargs) -> List[AgentLoopOutput]:
        """Override run to use TwoStageAgentData"""
        messages = list(kwargs["raw_prompt"])
        data_source = kwargs["data_source"]
        metrics = {}
        request_id = f"twostage_{self.training_stage}_{id(self)}"
        memory_kwargs = kwargs.get("memory_kwargs", {})
        self.retrieve_chunks = kwargs['tools_kwargs']['memory_bm25_retrieve']['create_kwargs']['chunks']

        # Initialize context chunks
        context_text = kwargs.get("context", "")
        if context_text:
            if kwargs.get('raw_chunks') is not None:
                context_chunks = kwargs['raw_chunks']
            elif data_source == 'synth':
                context_chunks = self.retrieve_chunks
            else:
                chunk_size = self.max_chunk_size
                from verl.experimental.agent_loop.tool_mem_agent_loop import get_chunks
                context_chunks = get_chunks(context_text, chunk_size)
            memory_kwargs["context_chunks"] = context_chunks
        else:
            memory_kwargs["context_chunks"] = []

        # Use TwoStageAgentData instead of AgentData
        agent_data = TwoStageAgentData(
            messages=messages,
            metrics=metrics,
            request_id=request_id,
            memory_kwargs=memory_kwargs,
        )

        from uuid import uuid4
        trajectory_id = str(uuid4())

        # Process each chunk
        for chunk_idx in range(len(agent_data.context_chunks)):
            agent_data.current_chunk_idx = chunk_idx
            await self._process_memory_chunk(agent_data, sampling_params, kwargs.get('verbose'), trajectory_id, data_source)

        # Generate final response
        await self._process_final_response(agent_data, sampling_params, kwargs.get('verbose'), trajectory_id, data_source)

        # Release tools
        for tool in self.tools.values():
            await tool.release(trajectory_id)

        # Create outputs with is_trainable flag
        outputs = []
        for i, conversation in enumerate(agent_data.conversations):
            output = AgentLoopOutput(
                prompt_ids=conversation.prompt_ids,
                response_ids=conversation.response_ids[:self.response_length],
                response_mask=conversation.response_mask[:self.response_length],
                multi_modal_data={},
                response_logprobs=conversation.response_logprobs[:self.response_length]
                if conversation.response_logprobs
                else None,
                num_turns=conversation.user_turns + conversation.assistant_turns + 1,
                metrics=agent_data.metrics,
                extra_fields={
                    'trajectory_id': trajectory_id,
                    'is_final': conversation.is_final,
                    'tool_rewards': conversation.tool_rewards,
                    'num_tools': conversation.num_tools,
                    'is_trainable': conversation.is_trainable,  # ⭐ Key field for loss filtering
                    'training_stage': self.training_stage,
                },
            )
            outputs.append(output)

        return outputs
