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
import json
import logging
import os
from enum import Enum
from typing import Any, List, Optional
from uuid import uuid4
from transformers import PreTrainedTokenizer
import numpy as np
import re
import torch
import yaml
from pathlib import Path

from verl.experimental.agent_loop.agent_loop import AgentLoopBase, AgentLoopOutput, register
from verl.experimental.agent_loop.tool_parser import FunctionCall, ToolParser
from verl.tools.schemas import ToolResponse
from verl.tools.utils.tool_registry import initialize_tools_from_config
from verl.utils.profiler import simple_timer
from verl.utils.rollout_trace import rollout_trace_op

logger = logging.getLogger(__file__)
logger.setLevel(os.getenv("VERL_LOGGING_LEVEL", "WARN"))

yaml_path = Path(os.environ['PROMPT_TEMPLATE_PATH'])
with open(yaml_path, 'r') as f:
    PROMPT_TEMPLATE = yaml.safe_load(f) 

class AgentState(Enum):
    GENERATING = "generating"
    PROCESSING_TOOLS = "processing_tools"
    TERMINATED = "terminated"

class TokenTemplate:
    """
    format string, but in token_ids, use torch.LongTensor as data type.
    Input value can also be nunpy.ndarray or list[int].

    usage:
    ```
    TEMPLATE = "Here is a problem: {problem}"
    "Given this section: {section}"
    "Please answer it."

    processor = TokenTemplate(TEMPLATE, tokenizer)

    kwarg_text = dict(
        problem="What is the capital of France?",
        section="Here is a introduction to France. France is a country in Western Europe. Its capital is Paris.",
    )
    kwargs_token_ids = {
        k: tokenizer.encode(v, add_special_tokens=False) for k, v in kwarg_text.items()
    }

    print(tokenizer.decode(processor.format(**kwargs_token_ids)))

    # just as a text format string.
    assert TEMPLATE.format(**kwarg_text) == tokenizer.decode(processor.format(**kwargs_token_ids))
    ```
    """

    def __init__(self, template: str, tokenizer: PreTrainedTokenizer=None):
        self.template = template
        self.initialized = False
        if tokenizer:
            self.init(tokenizer)
        
    def init(self, tokenizer):
        self.keywords: list[str] = []  # Store extracted {keywords}
        self.token_sections: list[torch.LongTensor] = []  # Store tokenized text sections as LongTensors
        self.last_section: torch.LongTensor = None  # Last section as LongTensor

        # Match only lowercase keywords with optional underscores (memory, chunk, prompttext)
        pattern = r'\{([a-z_]+)\}'
        parts = re.split(pattern, self.template) # 只匹配小写字母和下划线组合
        
        # Split text: even indices are non-{} parts, odd indices are {} keywords
        for i, part in enumerate(parts[:-1]):
            if i % 2 == 0:  # Even index, non-{} part
                tokens = tokenizer.encode(part, add_special_tokens=False)
                self.token_sections.append(torch.tensor(tokens, dtype=torch.long))
            else:  # Odd index, {} keyword
                self.keywords.append(part)
        self.last_section = torch.tensor(tokenizer.encode(parts[-1], add_special_tokens=False), dtype=torch.long)
        
        assert len(self.keywords) == len(self.token_sections), \
            f"{self.keywords} and {self.token_sections} should have the same length"
        self.initialized = type(tokenizer)

    def __len__(self) -> int:
        """
        Length of the template in token numbers
        """
        total = sum(section.numel() for section in self.token_sections)
        total += self.last_section.numel()
        return total

    def format(self, **kwargs: dict[str, torch.LongTensor | list[int] | np.ndarray]) -> torch.LongTensor:
        """
        Format the template with provided token ids
        
        Args:
            **kwargs: Dictionary of keyword to token ids (as LongTensor)
            
        Returns:
            Concatenated token ids as LongTensor
        """
        # Initialize with first section if exists
        formatted_parts = []
        
        # Reconstruct template by interleaving sections and keyword tokens
        for i, k in enumerate(self.keywords):
            if isinstance(kwargs[k], list):
                kwargs[k] = torch.tensor(kwargs[k], dtype=torch.long)
            elif isinstance(kwargs[k], np.ndarray):
                kwargs[k] = torch.from_numpy(kwargs[k]).to(torch.long)
            formatted_parts.append(self.token_sections[i])
            formatted_parts.append(kwargs[k])
        formatted_parts.append(self.last_section)
        
        return torch.cat(formatted_parts).tolist()

class TrajectoryConversation:
    """Represents a single conversation within a trajectory."""

    def __init__(self, prompt_ids: list[int], response_ids: list[int], response_mask: list[int],
                 response_logprobs: list[float], is_final: int = 0, user_turns: int = 0, assistant_turns: int = 0, tool_rewards: list[float] = [], num_tools: int = 0):
        self.prompt_ids = prompt_ids
        self.response_ids = response_ids
        self.response_mask = response_mask
        self.response_logprobs = response_logprobs
        self.is_final = is_final
        self.user_turns = user_turns
        self.assistant_turns = assistant_turns
        self.tool_rewards = tool_rewards
        self.num_tools = num_tools


class AgentData:
    """Encapsulates all state variables for the memory agent loop with tool support."""

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
        
        # State variables for current conversation (accumulative within a conversation)
        self.current_prompt_ids: list[int] = []
        self.current_response_ids: list[int] = []
        self.current_response_mask: list[int] = []
        self.current_response_logprobs: list[float] = []
        self.current_num_tools: int = 0
        
        # Memory management
        self.memory_turns = 0
        self.memory_content = memory_kwargs.get("initial_memory", "")
        self.context_chunks = memory_kwargs.get("context_chunks", [])
        self.current_chunk_idx = 0
        
        # Tool-related state
        self.tool_calls: list[FunctionCall] = []
        self.failed_tool_calls: list[tuple[int, str]] = []  # (index, error_message)
        self.user_turns = 0
        self.assistant_turns = 0
        self.current_tool_rewards: list[float] = []
    
    def add_conversation(self, is_final: int = 0):
        """Add a conversation to the trajectory using current accumulated state."""
        response_token_count = len(self.current_response_mask)
        prompt_token_count = max(len(self.current_prompt_ids) - response_token_count, 0)
        prompt_ids = self.current_prompt_ids[:prompt_token_count]
        conversation = TrajectoryConversation(
            prompt_ids=prompt_ids,
            response_ids=self.current_response_ids.copy(),
            response_mask=self.current_response_mask.copy(),
            response_logprobs=self.current_response_logprobs.copy(),
            is_final=is_final,
            user_turns=self.user_turns,
            assistant_turns=self.assistant_turns,
            tool_rewards=self.current_tool_rewards,
            num_tools=self.current_num_tools,
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

def get_chunks(context_text, chunk_size):
    separator = "\n"
    small_chunks = context_text.split(separator)
    chunks = []
    to_append = ""
    for chunk in small_chunks:
        if len(to_append) + len(chunk) + len(separator) <= chunk_size:
            if to_append:
                to_append += separator + chunk
            else:
                to_append = chunk
        else:
            if to_append:
                chunks.append(to_append)
            to_append = chunk
    if to_append:
        chunks.append(to_append)
    return chunks

@register("tool_mem_agent")
class ToolMemoryAgentLoop(AgentLoopBase):
    @classmethod
    def init_class(cls, config, tokenizer, processor, **kwargs):
        if cls._class_initialized:
            return
        cls._class_initialized = True
        print("Performing class-level ToolMemoryAgentLoop initialization")

        cls.tokenizer = tokenizer
        cls.processor = processor
        cls.max_user_turns = config.actor_rollout_ref.rollout.multi_turn.max_user_turns
        cls.max_assistant_turns = config.actor_rollout_ref.rollout.multi_turn.max_assistant_turns
        
        # Memory-specific configuration
        cls.max_chunk_size = config.actor_rollout_ref.rollout.multi_turn.get("max_chunk_size", 15000)
        
        # Tool-specific configuration
        cls.max_parallel_calls = config.actor_rollout_ref.rollout.multi_turn.get("max_parallel_calls", 5)
        cls.max_tool_response_length = config.actor_rollout_ref.rollout.multi_turn.get("max_tool_response_length", 1000)
        cls.tool_response_truncate_side = config.actor_rollout_ref.rollout.multi_turn.get("tool_response_truncate_side", "right")
        tool_config_path = config.actor_rollout_ref.rollout.multi_turn.get("tool_config_path", None)
        
        if tool_config_path:
            tool_list = initialize_tools_from_config(tool_config_path)
            cls.tools = {tool.name: tool for tool in tool_list}
            cls.tool_schemas = [tool.tool_schema.model_dump(exclude_unset=True, exclude_none=True) for tool in tool_list]
            cls.tool_parser = ToolParser.get_tool_parser(
                config.actor_rollout_ref.rollout.multi_turn.get("format", "default"), 
                cls.tokenizer
            )
            print(f"Initialized tools: {list(cls.tools.keys())}")
        else:
            cls.tools = {}
            cls.tool_schemas = []
            cls.tool_parser = None
            print("No tools configured")

        cls.apply_chat_template_kwargs = config.data.get("apply_chat_template_kwargs", {})
        cls.prompt_length = config.actor_rollout_ref.rollout.prompt_length
        cls.response_length = config.actor_rollout_ref.rollout.response_length
        cls.system_prompt = tokenizer.apply_chat_template(
            [{}], add_generation_prompt=False, tokenize=True, **cls.apply_chat_template_kwargs
        )

    @rollout_trace_op
    async def run(self, sampling_params: dict[str, Any], **kwargs) -> List[AgentLoopOutput]:
        messages = list(kwargs["raw_prompt"])
        data_source = kwargs["data_source"]
        # {'locomo',
        # 'memalpha_booksum',
        # 'memalpha_hotpotqa',
        # 'memalpha_icl_nlu_8296shot_balance',
        # 'memalpha_icl_trec_coarse_6600shot_balance',
        # 'memalpha_perltqa',
        # 'memalpha_pubmed-rct',
        # 'memalpha_squad',
        # 'synth'}
        metrics = {}
        request_id = uuid4().hex
        memory_kwargs = kwargs.get("memory_kwargs", {})
        self.retrieve_chunks = kwargs['tools_kwargs']['memory_bm25_retrieve']['create_kwargs']['chunks']
        # Initialize context chunks from the prompt
        context_text = kwargs.get("context", "")
        if context_text:
            # Split context into chunks
            chunk_size = memory_kwargs.get("chunk_size", self.max_chunk_size)
            context_chunks = get_chunks(context_text, chunk_size)
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
        # Process each chunk as a separate conversation with multi-turn tool calling
        for chunk_idx in range(len(agent_data.context_chunks)):
            agent_data.current_chunk_idx = chunk_idx
            await self._process_memory_chunk(agent_data, sampling_params, kwargs.get('verbose'), trajectory_id, data_source)
        # Generate final response with tool support
        await self._process_final_response(agent_data, sampling_params, kwargs.get('verbose'), trajectory_id, data_source)
        for tool in self.tools.values():
            await tool.release(trajectory_id)
        # Create a list of AgentLoopOutput objects, one for each conversation
        outputs = []
        for i, conversation in enumerate(agent_data.conversations):
            output = AgentLoopOutput(
                prompt_ids=conversation.prompt_ids,
                response_ids=conversation.response_ids[: self.response_length],
                response_mask=conversation.response_mask[: self.response_length],
                multi_modal_data={},
                response_logprobs=conversation.response_logprobs[: self.response_length]
                if conversation.response_logprobs
                else None,
                num_turns=conversation.user_turns + conversation.assistant_turns + 1,
                metrics=agent_data.metrics,
                extra_fields={'trajectory_id': trajectory_id, 'is_final': conversation.is_final, 'tool_rewards': conversation.tool_rewards, 
                              'num_tools': conversation.num_tools},
            )
            outputs.append(output)
        
        return outputs

    async def _process_memory_chunk(self, agent_data: AgentData, sampling_params: dict[str, Any], verbose: bool, trajectory_id: str, data_source: str):
        """Process a single memory chunk with multi-turn tool calling."""
        # Prepare memory update prompt
        current_chunk = agent_data.context_chunks[agent_data.current_chunk_idx]
        agent_data.current_prompt_ids = await self._build_memory_prompt_ids(agent_data.messages, agent_data.memory_content, current_chunk, data_source)
        if len(agent_data.current_prompt_ids) > self.prompt_length + self.response_length:
            print(f"WARNING: Memory prompt length {len(agent_data.current_prompt_ids)} exceeds max prompt length {self.prompt_length + self.response_length}")
        # Multi-turn generation loop for this chunk
        state = AgentState.GENERATING
        while state != AgentState.TERMINATED:
            if state == AgentState.GENERATING:
                state = await self._handle_generating_state(agent_data, sampling_params)
            elif state == AgentState.PROCESSING_TOOLS:
                state = await self._handle_processing_tools_state(agent_data, trajectory_id)
            
            # Check termination conditions for this conversation
            if len(agent_data.current_response_mask) >= self.response_length:
                state = AgentState.TERMINATED
            if self.max_assistant_turns and agent_data.assistant_turns >= self.max_assistant_turns:
                state = AgentState.TERMINATED

        await self._update_memory_content(agent_data, verbose, trajectory_id)
        agent_data.add_conversation(is_final=0)
        agent_data.memory_turns += 1

    async def _process_final_response(self, agent_data: AgentData, sampling_params: dict[str, Any], verbose: bool, trajectory_id: str, data_source: str):
        """Generate final response with tool support."""
        # Prepare final prompt
        final_prompts = await self._build_final_prompt_ids(agent_data.messages, agent_data.memory_content, data_source)
        for idx, final_prompt in enumerate(final_prompts):
            agent_data.current_prompt_ids = final_prompt
            if len(agent_data.current_prompt_ids) > self.prompt_length + self.response_length:
                print(f"WARNING: Final prompt length {len(agent_data.current_prompt_ids)} exceeds max prompt length {self.prompt_length + self.response_length}")
            # Multi-turn generation loop for final response
            state = AgentState.GENERATING
            while state != AgentState.TERMINATED:
                if state == AgentState.GENERATING:
                    state = await self._handle_generating_state(agent_data, sampling_params)
                elif state == AgentState.PROCESSING_TOOLS:
                    state = await self._handle_processing_tools_state(agent_data, trajectory_id)
                
                # Check termination conditions
                if len(agent_data.current_response_mask) >= self.response_length:
                    state = AgentState.TERMINATED
                if self.max_assistant_turns and agent_data.assistant_turns >= self.max_assistant_turns:
                    state = AgentState.TERMINATED
            
            if verbose:
                response_text = await asyncio.to_thread(
                    self.tokenizer.decode, agent_data.current_response_ids, skip_special_tokens=True
                )
                print(f"Final Answer:\n{response_text}")
            
            agent_data.add_conversation(is_final=idx+1)

    async def _handle_generating_state(self, agent_data: AgentData, sampling_params: dict[str, Any]) -> AgentState:
        """Handle the generating state: generate model response and check for tool calls."""
        max_model_len = self.prompt_length + self.response_length
        max_new_tokens = min(self.response_length, max_model_len - len(agent_data.current_prompt_ids) - 1)
        if max_new_tokens <= 0:
            print(f"ERROR: Current len(agent_data.current_prompt_ids)={len(agent_data.current_prompt_ids)} exceeds model max length {max_model_len}")
        with simple_timer("generate_sequences", agent_data.metrics):
            # 走的async_sglang_server.py
            output = await self.server_manager.generate(
                request_id=agent_data.request_id,
                prompt_ids=agent_data.current_prompt_ids,
                sampling_params=sampling_params,
            )
        agent_data.assistant_turns += 1
        response_ids = output.token_ids
        # if "<tool_call>" in self.tokenizer.decode(response_ids, skip_special_tokens=True):
        #     breakpoint()
        agent_data.current_prompt_ids += response_ids
        agent_data.current_response_ids += response_ids
        agent_data.current_response_mask += [1] * len(response_ids)
        if output.log_probs:
            agent_data.current_response_logprobs += output.log_probs
        else:
            agent_data.current_response_logprobs += [0.0] * len(response_ids)

        # Extract tool calls if tool parser is available
        agent_data.tool_calls = []
        agent_data.failed_tool_calls = []
        if self.tool_parser:
            _, agent_data.tool_calls, num_tools, agent_data.failed_tool_calls = await self.tool_parser.extract_tool_calls(response_ids, return_failed_and_num_tools=True)
            agent_data.current_num_tools += num_tools
        
        # Determine next state
        if agent_data.tool_calls or agent_data.failed_tool_calls:
            return AgentState.PROCESSING_TOOLS
        else:
            return AgentState.TERMINATED

    async def _handle_processing_tools_state(self, agent_data: AgentData, trajectory_id: str) -> AgentState:
        """Handle the processing tools state: execute tool calls and prepare tool responses."""
        # Build a list of (original_index, call_type, data) for all tool calls in original order
        all_calls = []
        success_idx = 0
        failed_dict = {idx: msg for idx, msg in agent_data.failed_tool_calls}
        
        # Merge successful and failed calls in original order
        total_calls = len(agent_data.tool_calls) + len(agent_data.failed_tool_calls)
        for i in range(total_calls):
            if i in failed_dict:
                all_calls.append(('failed', failed_dict[i]))
            else:
                if success_idx < len(agent_data.tool_calls):
                    all_calls.append(('success', agent_data.tool_calls[success_idx]))
                    success_idx += 1
        
        # Limit to max_parallel_calls
        all_calls = all_calls[:self.max_parallel_calls]
        
        # Prepare tasks for parallel execution (only for successful calls)
        tasks = []
        for call_type, call_data in all_calls:
            if call_type == 'success':
                tasks.append(self._call_tool(call_data, trajectory_id=trajectory_id))
        
        # Execute all successful tools in parallel
        with simple_timer("tool_calls", agent_data.metrics):
            if tasks:
                results = await asyncio.gather(*tasks)
            else:
                results = []
        
        # Process responses in original order
        add_messages: list[dict[str, Any]] = []
        result_idx = 0
        for call_type, call_data in all_calls:
            if call_type == 'failed':
                message = {"role": "tool", "content": f"Error: {call_data}"}
                add_messages.append(message)
            else:
                tool_response, tool_reward, _ = results[result_idx]
                message = {"role": "tool", "content": tool_response.text or ""}
                add_messages.append(message)
                if tool_reward is not None and tool_reward != 0:
                    agent_data.current_tool_rewards.append(tool_reward)
                result_idx += 1
        
        # Update prompt with tool responses
        response_ids = await asyncio.to_thread(
            self.tokenizer.apply_chat_template,
            add_messages,
            add_generation_prompt=True,
            tokenize=True,
            **self.apply_chat_template_kwargs
        )
        response_ids = response_ids[len(self.system_prompt):]
        
        # Check if we would exceed response length
        if len(agent_data.current_response_ids) + len(response_ids) >= self.response_length:
            return AgentState.TERMINATED
        if len(agent_data.current_response_ids) + len(response_ids) >= self.response_length - 500:
            add_prompt = "No more tools can be called due to response length limit."
            add_prompt_ids = await asyncio.to_thread(
                self.tokenizer.encode, add_prompt, add_special_tokens=False
            )
            if len(agent_data.current_response_ids) + len(response_ids) + len(add_prompt_ids) < self.response_length:
                response_ids += add_prompt_ids
        # Update prompt_ids and response_mask
        agent_data.current_prompt_ids += response_ids
        agent_data.current_response_ids += response_ids
        agent_data.current_response_mask += [0] * len(response_ids)
        agent_data.current_response_logprobs += [0.0] * len(response_ids)
        
        # Clear tool calls for next round
        agent_data.tool_calls = []
        agent_data.failed_tool_calls = []
        agent_data.user_turns += 1
        return AgentState.GENERATING

    async def _update_memory_content(self, agent_data: AgentData, verbose: bool, trajectory_id: str):
        """Update memory content from the generated response."""
        # Decode the response to get updated memory
        response_text = await asyncio.to_thread(
            self.tokenizer.decode, agent_data.current_response_ids, skip_special_tokens=True
        )
        # Extract memory content (remove thinking tags if present)
        for sep in ['</tool_response>\nassistant', '</think>']:
            if sep in response_text:
                response_text = response_text.split(sep)[-1].strip()

        if verbose:
            print(f"Updated memory:\n{response_text}")
            print(f"TRAJECTORY_ID: {trajectory_id}")
            # print(f"Memory turn: {agent_data.memory_turns}")
            print(f"Current chunk index: {agent_data.current_chunk_idx + 1} / {len(agent_data.context_chunks)}")
            current_chunk_idx = agent_data.current_chunk_idx
            current_full_chunk = agent_data.context_chunks[current_chunk_idx]
            show_len = 100
            if len(current_full_chunk) <= show_len:
                print(f"Current chunk:\n{current_full_chunk}")
            else:
                print(fr"Current chunk (truncated):\n{current_full_chunk[:show_len // 2]}...(truncated)...{current_full_chunk[-show_len // 2:]}")
        
        agent_data.memory_content = response_text

    async def _call_tool(self, tool_call: FunctionCall, trajectory_id: str) -> ToolResponse:
        """Call tool and return tool response."""
        tool, instance_id = None, None
        try:
            tool_name = tool_call.name
            tool_args = json.loads(tool_call.arguments)
            tool = self.tools[tool_name]
            create_kwargs = {
                'trajectory_id': trajectory_id,
                'filename': './tmp/verl_agent/memory_store.jsonl'
            }
            if tool_name == "memory_bm25_retrieve" or tool_name == "memory_embedding_retrieve":
                create_kwargs['chunks'] = self.retrieve_chunks
            instance_id, _ = await tool.create(create_kwargs=create_kwargs)
            tool_execution_response, tool_reward, res = await tool.execute(instance_id, tool_args)
        except Exception as e:
            logger.warning(f"Error when executing {tool_name} tool: {e}")
            return ToolResponse(text=f"Error when executing {tool_name} tool: {e}"), 0.0, {}

        tool_response_text = tool_execution_response.text
        if tool_response_text and len(tool_response_text) > self.max_tool_response_length:
            if self.tool_response_truncate_side == "left":
                tool_response_text = tool_response_text[: self.max_tool_response_length] + "...(truncated)"
            elif self.tool_response_truncate_side == "right":
                tool_response_text = "(truncated)..." + tool_response_text[-self.max_tool_response_length:]
            else:
                length = self.max_tool_response_length // 2
                tool_response_text = tool_response_text[:length] + "...(truncated)..." + tool_response_text[-length:]

        return ToolResponse(text=tool_response_text), tool_reward, res

    async def _build_memory_prompt_ids(self, messages: list[dict[str, Any]], memory: str, chunk: str, data_source: str) -> str:
        """Build prompt for memory update."""
        prompt_key = f"memory_prompt_{data_source}"
        MEMORY_PROMPT_TEMPLATE = PROMPT_TEMPLATE.get(prompt_key, PROMPT_TEMPLATE["memory_prompt_default"])
        FULL_MEMORY_PROMPT_TEMPLATE = self.tokenizer.apply_chat_template(
            [{"role": "user", "content": MEMORY_PROMPT_TEMPLATE}],
            tools=self.tool_schemas if self.tools else None,
            add_generation_prompt=True,
            tokenize=False,
            **self.apply_chat_template_kwargs,
        )
        memory_token_template = TokenTemplate(FULL_MEMORY_PROMPT_TEMPLATE, self.tokenizer)
        memory_ids = self.tokenizer.encode(memory, add_special_tokens=False)
        chunk_ids = self.tokenizer.encode(chunk, add_special_tokens=False)
        if len(memory_token_template) + len(memory_ids) + len(chunk_ids) > self.prompt_length:
            print(f"WARNING: Memory prompt length {len(memory_token_template)} + {len(memory_ids)} + {len(chunk_ids)} exceeds max prompt length {self.prompt_length}, truncating memory.")
            memory_ids = memory_ids[-(self.prompt_length - 200 - len(memory_token_template) - len(chunk_ids)):]
            # 留200个token给response
        memory_prompt = memory_token_template.format(
            memory=memory_ids,
            chunk=chunk_ids,
        )
        # print("#" * 50 + f"Memory prompt: {self.tokenizer.decode(memory_prompt, skip_special_tokens=False)}")
        return memory_prompt

    async def _build_final_prompt_ids(self, messages: list[dict[str, Any]], memory: str, data_source: str) -> str:
        """Build prompt for final response."""
        prompt_key = f"final_prompt_{data_source}"
        FINAL_PROMPT_TEMPLATE = PROMPT_TEMPLATE.get(prompt_key, PROMPT_TEMPLATE["final_prompt_default"])
        prompt_text = messages[0]["content"] if messages[0]["role"] == "user" else messages[1]["content"]
        FULL_FINAL_PROMPT_TEMPLATE = self.tokenizer.apply_chat_template(
            [{"role": "user", "content": FINAL_PROMPT_TEMPLATE}],
            tools=self.tool_schemas if self.tools else None,
            add_generation_prompt=True,
            tokenize=False,
            **self.apply_chat_template_kwargs,
        )
        final_token_template = TokenTemplate(FULL_FINAL_PROMPT_TEMPLATE, self.tokenizer)
        memory_ids = self.tokenizer.encode(memory if memory else "No previous memory", add_special_tokens=False)
        final_prompts = []
        if isinstance(prompt_text, str):
            prompt_text = [prompt_text]
        for p in prompt_text:
            prompt_ids = self.tokenizer.encode(p, add_special_tokens=False)
            if len(final_token_template) + len(prompt_ids) + len(memory_ids) > self.prompt_length:
                print(f"WARNING: Final prompt length {len(final_token_template)} + {len(memory_ids)} + {len(prompt_ids)} exceeds max prompt length {self.prompt_length}, truncating memory.")
                memory_ids = memory_ids[-(self.prompt_length - 200 - len(final_token_template) - len(prompt_ids)):]
                # 留200个token给response
            final_prompt = final_token_template.format(
                prompttext=prompt_ids,
                memory=memory_ids,
            )
            final_prompts.append(final_prompt)
            # print("#" * 50 + f"Final prompt: {self.tokenizer.decode(final_prompt, skip_special_tokens=False)}")
        return final_prompts