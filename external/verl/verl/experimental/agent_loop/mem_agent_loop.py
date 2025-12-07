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
import asyncio
import copy
import json
import logging
import os
from enum import Enum
from typing import Any, List, Optional
from uuid import uuid4

from verl.experimental.agent_loop.agent_loop import AgentLoopBase, AgentLoopOutput, register
from verl.utils.profiler import simple_timer
from verl.utils.rollout_trace import rollout_trace_op

logger = logging.getLogger(__file__)
logger.setLevel(os.getenv("VERL_LOGGING_LEVEL", "WARN"))


class AgentState(Enum):
    PENDING = "pending"
    GENERATING = "generating"
    PROCESSING_MEMORY = "processing_memory"
    TERMINATED = "terminated"


class TrajectoryConversation:
    """Represents a single conversation within a trajectory."""
    
    def __init__(self, prompt_ids: list[int], response_ids: list[int], response_mask: list[int], 
                 response_logprobs: list[float], is_final: bool = False):
        self.prompt_ids = prompt_ids
        self.response_ids = response_ids
        self.response_mask = response_mask
        self.response_logprobs = response_logprobs
        self.is_final = is_final


class AgentData:
    """Encapsulates all state variables for the memory agent loop with trajectory support."""

    def __init__(
        self,
        messages: list[dict[str, Any]],
        metrics: dict[str, Any],
        request_id: str,
        memory_kwargs: dict[str, Any],
    ):
        self.messages = messages
        self.metrics = metrics
        self.request_id = request_id
        self.memory_kwargs = memory_kwargs

        # Trajectory tracking
        self.conversations: list[TrajectoryConversation] = []
        
        # State variables for current conversation
        self.current_prompt_ids: list[int] = []
        self.current_response_ids: list[int] = []
        self.current_response_mask: list[int] = []
        self.current_response_logprobs: list[float] = []
        
        # Memory management
        self.memory_turns = 0
        self.memory_content = memory_kwargs.get("initial_memory", "")
        self.context_chunks = memory_kwargs.get("context_chunks", [])
        self.current_chunk_idx = 0
        
        # Final trajectory data
        self.final_reward = None
        self.final_memory_content = ""
    
    def add_conversation(self, prompt_ids: list[int], response_ids: list[int], 
                        response_mask: list[int], response_logprobs: list[float],
                        is_final: bool = False):
        """Add a conversation to the trajectory."""
        conversation = TrajectoryConversation(
            prompt_ids=prompt_ids.copy(),
            response_ids=response_ids.copy(),
            response_mask=response_mask.copy(),
            response_logprobs=response_logprobs.copy(),
            is_final=is_final,
        )
        self.conversations.append(conversation)


@register("mem_agent")
class MemoryAgentLoop(AgentLoopBase):
    @classmethod
    def init_class(cls, config, tokenizer, processor, **kwargs):
        if cls._class_initialized:
            return
        cls._class_initialized = True
        print("Performing class-level MemoryAgentLoop initialization")

        cls.tokenizer = tokenizer
        cls.processor = processor
        cls.max_user_turns = config.actor_rollout_ref.rollout.multi_turn.max_user_turns
        cls.max_assistant_turns = config.actor_rollout_ref.rollout.multi_turn.max_assistant_turns
        
        # Memory-specific configuration
        cls.max_chunk_size = config.actor_rollout_ref.rollout.multi_turn.get("max_chunk_size", 15000)
        cls.max_memorization_length = config.actor_rollout_ref.rollout.multi_turn.get("max_memorization_length", 256)
        cls.max_final_response_length = config.actor_rollout_ref.rollout.multi_turn.get("max_final_response_length", 256)

        cls.apply_chat_template_kwargs = config.data.get("apply_chat_template_kwargs", {})
        cls.prompt_length = config.actor_rollout_ref.rollout.prompt_length
        cls.response_length = config.actor_rollout_ref.rollout.response_length

    @rollout_trace_op
    async def run(self, sampling_params: dict[str, Any], **kwargs) -> List[AgentLoopOutput]:
        messages = list(kwargs["raw_prompt"])
        metrics = {}
        request_id = uuid4().hex
        memory_kwargs = kwargs.get("memory_kwargs", {})

        # Initialize context chunks from the prompt
        context_text = kwargs.get("context", "")
        if context_text:
            # Split context into chunks
            chunk_size = memory_kwargs.get("chunk_size", self.max_chunk_size)
            context_chunks = [context_text[i:i+chunk_size] for i in range(0, len(context_text), chunk_size)]
            memory_kwargs["context_chunks"] = context_chunks
        else:
            memory_kwargs["context_chunks"] = []

        # Create AgentData instance to encapsulate all state
        agent_data = AgentData(
            messages=messages,
            metrics=metrics,
            request_id=request_id,
            memory_kwargs=memory_kwargs,
        )
        trajectory_id = str(uuid4())
        # State machine loop
        state = AgentState.PENDING
        while state != AgentState.TERMINATED:
            if state == AgentState.PENDING:
                state = await self._handle_pending_state(agent_data, sampling_params)
            elif state == AgentState.GENERATING:
                state = await self._handle_generating_state(agent_data, sampling_params, kwargs.get('verbose'))
                agent_data.memory_turns += 1
            elif state == AgentState.PROCESSING_MEMORY: # 输出中间记忆过程；更新memory_content；存储每一步的memory，但是存下来好像也没啥用
                state = await self._handle_processing_memory_state(agent_data, kwargs.get('verbose'))
            else:
                logger.error(f"Invalid state: {state}")
                state = AgentState.TERMINATED

        # Create a list of AgentLoopOutput objects, one for each conversation
        outputs = []
        
        for i, conversation in enumerate(agent_data.conversations):
            is_final = conversation.is_final
            
            # Create individual output for each conversation
            output = AgentLoopOutput(
                prompt_ids=conversation.prompt_ids,
                response_ids=conversation.response_ids[: self.response_length],
                response_mask=conversation.response_mask[: self.response_length],
                multi_modal_data={},
                response_logprobs=conversation.response_logprobs[: self.response_length]
                if conversation.response_logprobs
                else None,
                num_turns=2,  # Each conversation represents one turn
                metrics=agent_data.metrics,
                extra_fields={'trajectory_id': trajectory_id, 'is_final': is_final},
            )
            outputs.append(output)
        
        return outputs

    async def _handle_pending_state(self, agent_data: AgentData, sampling_params: dict[str, Any]) -> AgentState:
        """Handle the pending state: prepare the prompt and start generation."""
        if agent_data.current_chunk_idx < len(agent_data.context_chunks):
            # Prepare memory update prompt
            current_chunk = agent_data.context_chunks[agent_data.current_chunk_idx]
            memory_prompt = self._build_memory_prompt(
                agent_data.messages, agent_data.memory_content, current_chunk
            )
            
            agent_data.current_prompt_ids = await self.loop.run_in_executor(
                None,
                lambda: self.tokenizer.apply_chat_template(
                    [{"role": "user", "content": memory_prompt}],
                    add_generation_prompt=True,
                    tokenize=True,
                    **self.apply_chat_template_kwargs,
                ),
            )
        else:
            # Final response generation
            final_prompt = self._build_final_prompt(agent_data.messages, agent_data.memory_content)
            
            agent_data.current_prompt_ids = await self.loop.run_in_executor(
                None,
                lambda: self.tokenizer.apply_chat_template(
                    [{"role": "user", "content": final_prompt}],
                    add_generation_prompt=True,
                    tokenize=True,
                    **self.apply_chat_template_kwargs,
                ),
            )
        return AgentState.GENERATING

    async def _handle_generating_state(self, agent_data: AgentData, sampling_params: dict[str, Any], verbose: bool) -> AgentState:
        """Handle the generating state: generate model response."""
        with simple_timer("generate_sequences", agent_data.metrics):
            output = await self.server_manager.generate(
                request_id=agent_data.request_id,
                prompt_ids=agent_data.current_prompt_ids,
                sampling_params=sampling_params,
            )

        agent_data.current_response_ids = output.token_ids
        agent_data.current_response_mask = [1] * len(output.token_ids)
        if output.log_probs:
            agent_data.current_response_logprobs = output.log_probs
        else:
            agent_data.current_response_logprobs = []

        # Save this conversation to trajectory
        is_final = agent_data.current_chunk_idx >= len(agent_data.context_chunks)
        agent_data.add_conversation(
            prompt_ids=agent_data.current_prompt_ids,
            response_ids=agent_data.current_response_ids,
            response_mask=agent_data.current_response_mask,
            response_logprobs=agent_data.current_response_logprobs,
            is_final=is_final
        )
        if verbose and is_final:
            response_text = await self.loop.run_in_executor(
                None,
                lambda: self.tokenizer.decode(agent_data.current_response_ids, skip_special_tokens=True),
            )
            print(f"Final Answer:\n{response_text}")
        # Check termination conditions
        if agent_data.current_chunk_idx >= len(agent_data.context_chunks):
            return AgentState.TERMINATED

        return AgentState.PROCESSING_MEMORY

    async def _handle_processing_memory_state(self, agent_data: AgentData, verbose: bool) -> AgentState:
        """Handle the processing memory state: update memory content."""
        # Decode the response to get updated memory
        response_text = await self.loop.run_in_executor(
            None,
            lambda: self.tokenizer.decode(agent_data.current_response_ids, skip_special_tokens=True),
        )
        
        # Update memory content
        if '</think>' in response_text:
            response_text = response_text.split('</think>')[-1].strip()
        agent_data.memory_content = response_text
        agent_data.current_chunk_idx += 1

        # Add memory update to conversation
        memory_message = {
            "role": "assistant",
            "content": f"Updated memory:\n{response_text}"
        }
        if verbose:
            print(f"memory_turns:\n{agent_data.memory_turns}")
            print(memory_message['content'])
        agent_data.messages.append(memory_message)
        return AgentState.PENDING

    def _build_memory_prompt(self, messages: list[dict[str, Any]], memory: str, chunk: str) -> str:
        """Build prompt for memory update."""
        prompt_text = messages[0]["content"] if messages else ""
        
        memory_prompt = f"""You are presented with a problem, a section of context that may contain relevant information, and a previous memory. Please read the provided section carefully and update the memory with the new information that helps to answer the problem. Be sure to retain all relevant details from the previous memory while adding any new, useful information.

<problem>
{prompt_text}
</problem>

<memory>
{memory if memory else "No previous memory"}
</memory>

<section>
{chunk}
</section>

Updated memory:"""
        return memory_prompt

    def _build_final_prompt(self, messages: list[dict[str, Any]], memory: str) -> str:
        """Build prompt for final response."""
        prompt_text = messages[0]["content"] if messages else ""
        
        final_prompt = f"""You are presented with a problem and a previous memory. Please answer the problem based on the previous memory and put the answer in \boxed{{}}.

<problem>
{prompt_text}
</problem>

<memory>
{memory if memory else "No previous memory"}
</memory>

Your answer:"""
        return final_prompt