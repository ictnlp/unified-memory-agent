import json
import backoff
from openai import RateLimitError
from .base_agent import BaseAgent
from typing import List, Union
import tqdm


class MemAgent(BaseAgent):
    """
    Memory Agent implementation with two memory versions:
    1. Concatenated memory - direct chunk concatenation as string
    2. Updated memory - MemAgent-style memory updates as string
    
    Supports wo_q parameter for query-aware vs query-agnostic memory building.
    """
    
    # Templates for memory updates
    MEMORY_UPDATE_TEMPLATE = """You are presented with a problem, a section of an article that may contain the answer to the problem, and a previous memory. Please read the provided section carefully and update the memory with the new information that helps to answer the problem. Be sure to retain all relevant details from the previous memory while adding any new, useful information.

<problem> 
{prompt}
</problem>

<memory>
{memory}
</memory>

<section>
{chunk}
</section>

Updated memory:
"""

    MEMORY_UPDATE_TEMPLATE_WO_Q = """You are presented with a previous memory and a section of an article that may contain important information. Please read the provided section carefully and update the memory with the new information. Be sure to retain all important details from the previous memory while adding any new, useful information.

<memory>
{memory}
</memory>

<section>
{chunk}
</section>

Updated memory:
"""

    FINAL_QA_TEMPLATE = """You are presented with a problem and a previous memory. Please answer the problem based on the previous memory.

<problem> 
{prompt}
</problem>

<memory>
{memory}
</memory>

Your answer:
"""

    def __init__(self, chunk_size: int = 5000, max_chunks: int = None, wo_q: bool = False, **kwargs):
        super().__init__(**kwargs)
        self.chunk_size = chunk_size  # Size of each chunk in characters
        self.max_chunks = max_chunks  # Maximum number of chunks to process
        self.wo_q = wo_q  # Whether to use wo_q template
        # Two versions of memory (both as strings)
        self.concat_memory = ""  # Version 1: Direct concatenation
        self.updated_memory = "No previous memory"  # Version 2: MemAgent-style updates
    
    def add_memory(self, chunk: str):
        """
        Add memory chunk and maintain both versions:
        1. Concatenated memory (direct append)
        2. Updated memory (MemAgent-style processing using wo_q template)
        """
        # Version 1: Direct concatenation
        if self.concat_memory:
            self.concat_memory += f"\n\n{chunk}"
        else:
            self.concat_memory = chunk
        
        # Version 2: MemAgent-style memory update using wo_q template
        if self.wo_q:
            try:
                prompt_text = self.MEMORY_UPDATE_TEMPLATE_WO_Q.format(
                    memory=self.updated_memory,
                    chunk=chunk
                )
                
                messages = [{"role": "user", "content": prompt_text}]
                response = self._make_request(messages, max_tokens=2048)
                self.updated_memory = response.strip()
            except Exception as e:
                print(f"Error updating memory in add_memory: {e}")
                # Handle content filter and other API errors gracefully
                error_message = self._handle_api_error(e, chunk[:100])
                print(f"Memory update failed: {error_message}")
                # Fallback: just append the chunk to updated memory
                if self.updated_memory == "No previous memory":
                    self.updated_memory = f"[Memory update failed due to API error] {chunk}"
                else:
                    self.updated_memory += f"\n\n[Memory update failed due to API error] {chunk}"

    def _chunk_text(self, text: str) -> List[str]:
        """
        Split text into chunks of approximately chunk_size characters.
        """
        if len(text) <= self.chunk_size:
            return [text]
        
        chunks = []
        start = 0
        
        while start < len(text):
            end = start + self.chunk_size
            
            # If this is not the last chunk, try to find a good break point
            if end < len(text):
                # Look for word boundary within the last 200 characters
                search_start = max(end - 200, start)
                good_break = text.rfind(' ', search_start, end)
                
                if good_break > start:
                    end = good_break
            
            chunks.append(text[start:end].strip())
            start = end
            
            # Skip any leading whitespace for the next chunk
            while start < len(text) and text[start].isspace():
                start += 1
        
        return chunks
    
    def _process_memory_with_query(self, query: str) -> str:
        """
        Process concatenated memory using MemAgent approach with given query.
        """
        if not self.concat_memory:
            return "No previous memory"
        
        # Split concatenated memory into processing chunks
        processing_chunks = self._chunk_text(self.concat_memory)
        
        # Limit chunks if max_chunks is set
        if self.max_chunks and len(processing_chunks) > self.max_chunks:
            # Take first half and last half to stay within limit
            half_limit = self.max_chunks // 2
            processing_chunks = processing_chunks[:half_limit] + processing_chunks[-half_limit:]
        
        memory = "No previous memory"
        
        for chunk in tqdm.tqdm(processing_chunks, desc="Processing chunks"):
            try:
                prompt_text = self.MEMORY_UPDATE_TEMPLATE.format(
                    prompt=query,
                    memory=memory,
                    chunk=chunk
                )
                
                messages = [{"role": "user", "content": prompt_text}]
                
                # Update memory using LLM
                response = self._make_request(messages, max_tokens=2048)
                memory = response.strip()
                
            except Exception as e:
                print(f"Error updating memory in _process_memory_with_query: {e}")
                # Handle content filter and other API errors gracefully
                error_message = self._handle_api_error(e, f"Query: {query[:50]}... Chunk: {chunk[:50]}...")
                print(f"Memory processing failed: {error_message}")
                # If memory update fails, fall back to concatenating chunk
                if memory == "No previous memory":
                    memory = f"[Memory processing failed due to API error] {chunk}"
                else:
                    memory += f"\n\n[Memory processing failed due to API error] {chunk}"
        
        return memory
    
    @backoff.on_exception(backoff.expo, RateLimitError, max_tries=16, max_time=300)
    def _make_request(self, messages: List[dict], max_tokens: int = 1024, temperature: float = 0.0):
        """Make API request with retry logic"""
        response = self.client.chat.completions.create(
            model=self.model_name,
            messages=messages,
            max_tokens=max_tokens,
            temperature=temperature
        )
        return response.choices[0].message.content
    
    def QA(self, query: str, wo_q: bool = False) -> str:
        """
        Answer query using memory. 
        
        Args:
            query: The question to answer
            wo_q: If False (default), use query-aware memory processing from concat_memory
                  If True, use pre-built updated_memory directly
        """
        try:
            if wo_q:
                # Use version 2: pre-built updated memory (without query awareness)
                memory_to_use = self.updated_memory
            else:
                # Use version 1: Process concatenated memory with query awareness
                memory_to_use = self._process_memory_with_query(query)
            
            # Use FINAL_QA_TEMPLATE for both cases
            prompt_text = self.FINAL_QA_TEMPLATE.format(
                prompt=query,
                memory=memory_to_use
            )
            
            messages = [{"role": "user", "content": prompt_text}]
            response = self._make_request(messages, max_tokens=1024)
            
            return response.strip()
            
        except Exception as e:
            return self._handle_api_error(e, query)
    
    def QA_batch(self, queries: List[str], wo_q: bool = False) -> List[str]:
        """
        Answer multiple queries using memory.
        
        Args:
            queries: List of questions to answer
            wo_q: If False, use query-aware memory processing
                  If True, use direct updated memory
        """
        try:
            if wo_q:
                # For wo_q=True, use pre-built updated memory for all queries
                memory_to_use = self.updated_memory
                
                # Process all queries with the same memory
                responses = []
                for query in queries:
                    prompt_text = self.FINAL_QA_TEMPLATE.format(
                        prompt=query,
                        memory=memory_to_use
                    )
                    messages = [{"role": "user", "content": prompt_text}]
                    response = self._make_request(messages, max_tokens=1024)
                    responses.append(response.strip())
                
                return responses
            else:
                # For query-aware processing, we can optimize by processing memory once
                # for all queries (assuming they're related)
                if not queries:
                    return []
                
                responses = []
                for query in queries:
                    processed_memory = self._process_memory_with_query(query)
                    prompt_text = self.FINAL_QA_TEMPLATE.format(
                        prompt=query,
                        memory=processed_memory
                    )
                    messages = [{"role": "user", "content": prompt_text}]
                    response = self._make_request(messages, max_tokens=1024)
                    responses.append(response.strip())
                
                return responses
                
        except Exception as e:
            error_msg = self._handle_api_error(e, f"Batch queries: {queries}")
            return [error_msg] * len(queries)
    
    def get_memory_stats(self) -> dict:
        """Get statistics about current memory state"""
        concat_length = len(self.concat_memory.split()) if self.concat_memory else 0
        updated_length = len(self.updated_memory.split()) if self.updated_memory != "No previous memory" else 0
        
        return {
            "concat_memory_words": concat_length,
            "updated_memory_words": updated_length,
        }