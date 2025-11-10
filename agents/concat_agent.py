from typing import List
import json
import inspect
from .base_agent import BaseAgent, MODEL_NAME_MAP

QA_PROMPT = """
Based on your memory, write an answer in the form of a short phrase or a lowercase option for the following question. If it is a multiple-choice question, please respond with the lowercase option only (e.g. 'a'). Otherwise, answer with exact words from the context whenever possible.

Question: {} Short answer:
"""

QA_PROMPT_BATCH = """
Based on the above conversations, write short answers for each of the following questions in a few words. 
Write the answers in the form of a json dictionary where each entry contains the question number as "key" and the short answer as "value". 
Use single-quote characters for named entities and double-quote characters for enclosing json elements. For each question, if it is a multiple-choice question, please respond with the lowercase option only (e.g. 'a'). Otherwise, answer with exact words from the context whenever possible.

"""
class ConcatAgent(BaseAgent):
    def __init__(
        self,
        client = None,
        model_name: str = "gpt4.1"
    ):
        super().__init__(client, model_name)
        self.memory: List[str] = []
        self._is_async = self._check_if_async_client()
    
    def _check_if_async_client(self):
        """Check if client is AsyncOpenAI"""
        return hasattr(self.client, '__class__') and 'Async' in self.client.__class__.__name__
    
    def add_memory(
        self,
        chunk: str
    ):
        self.memory.append(chunk)

    def reset(self) -> None:
        self.memory = []
    
    async def QA_async(
        self,
        query: str
    ) -> str:
        try:
            context = "\n".join(self.memory)
            prompt = f"Your memory:\n{context}\n\n{QA_PROMPT.format(query)}"
            
            model = MODEL_NAME_MAP.get(self.model_name, self.model_name)
            response = await self.client.chat.completions.create(
                model=model,
                messages=[
                    {"role": "user", "content": prompt}
                ],
                max_tokens=2048,
                temperature=0.7
            )
            res = response.choices[0].message.content
            if "</think>" in res:
                res = res.split("</think>")[-1].strip()
                return res
            elif "<think>" in res:
                return f"ERROR_THINK_LENGTH_EXCEEDED: The think is too long for the model to process. Think: {res[:100]}..."
            return res.strip()
        except Exception as e:
            return self._handle_api_error(e, query)
    
    def QA(
        self,
        query: str
    ) -> str:
        if self._is_async:
            raise RuntimeError("Use QA_async() for async client")
        
        try:
            context = "\n".join(self.memory)
            prompt = f"Your memory:\n{context}\n\n{QA_PROMPT.format(query)}"
            
            model = MODEL_NAME_MAP.get(self.model_name, self.model_name)
            response = self.client.chat.completions.create(
                model=model,
                messages=[
                    {"role": "user", "content": prompt}
                ],
                max_tokens=2048,
                temperature=0.7
            )
            res = response.choices[0].message.content
            if "</think>" in res:
                res = res.split("</think>")[-1].strip()
                return res
            elif "<think>" in res:
                return f"ERROR_THINK_LENGTH_EXCEEDED: The think is too long for the model to process. Think: {res[:100]}..."
            return res.strip()
        except Exception as e:
            return self._handle_api_error(e, query)
    
    async def QA_batch_async(
        self,
        query_list: list[str],
        batch_size: int=32
    ) -> list[str]:
        if len(query_list) == 1:
            return [await self.QA_async(query_list[0])]
        context = "\n".join(self.memory)
        res = []
        for batch_idx in range(0, len(query_list), batch_size):
            query_batch = query_list[batch_idx : batch_idx+batch_size]
            prompt = f"Your memory:\n{context}\n\n{QA_PROMPT_BATCH}{"\n".join(["%s: %s" % (k, q) for k, q in enumerate(query_batch)])}"

            model = MODEL_NAME_MAP.get(self.model_name, self.model_name)
            retries = 0
            reformat_response = None  # Initialize to None
            while retries < 3:
                try:
                    response = await self.client.chat.completions.create(
                        model=model,
                        messages=[
                            {"role": "user", "content": prompt}
                        ],
                        max_tokens=2048 * batch_size,
                        temperature=0.7
                    )
                    response_str = response.choices[0].message.content
                    if "</think>" in response_str:
                        response_str = response_str.split("</think>")[-1].strip()
                    text = response_str.replace('\\"', "'").replace('json','').replace('`','').strip().replace("\\'", "").strip()
                    if text.count("'") > text.count('"'):
                        text = text.replace('"', "")
                        text = text.replace("'", '"')
                    reformat_response = json.loads(text)
                    break
                except json.decoder.JSONDecodeError as e:
                    print(f'JsonError: {e}\nText: {text}\nRetry: {retries}')
                    retries += 1
                    continue
                except Exception as e:
                    error_msg = self._handle_api_error(e, f"Batch queries: {query_batch}")
                    print(f"API Error in batch {batch_idx//batch_size + 1}: {error_msg}")
                    retries += 1
                    continue
            
            # Parse results for this batch
            if reformat_response is None:
                # All retries failed, add error messages for this batch
                print(f"Batch {batch_idx//batch_size + 1} failed after {retries} retries, adding error responses")
                for _ in range(len(query_batch)):
                    res.append("ERROR_BATCH_FAILED: Failed to get response after retries")
            else:
                # Use local index i (0, 1, 2, ...) for each batch
                for i in range(len(query_batch)):
                    try:
                        try:
                            res.append(str(reformat_response[str(i)]).replace('(a)', '').replace('(b)', '').strip())
                        except:
                            res.append(', '.join([str(n) for n in list(reformat_response[str(i)].values())]))
                    except:
                        res.append("ERROR_PARSING: Failed to parse response")
        return res
    
    def QA_batch(
        self,
        query_list: list[str],
        batch_size: int=32
    ) -> list[str]:
        if self._is_async:
            raise RuntimeError("Use QA_batch_async() for async client")
        context = "\n".join(self.memory)
        res = []
        for batch_idx in range(0, len(query_list), batch_size):
            query_batch = query_list[batch_idx : batch_idx+batch_size]
            prompt = f"Your memory:\n{context}\n\n{QA_PROMPT_BATCH}{"\n".join(["%s: %s" % (k, q) for k, q in enumerate(query_batch)])}"

            model = MODEL_NAME_MAP.get(self.model_name, self.model_name)
            retries = 0
            reformat_response = None  # Initialize to None
            while retries < 3:
                try:
                    response = self.client.chat.completions.create(
                        model=model,
                        messages=[
                            {"role": "user", "content": prompt}
                        ],
                        max_tokens=2048 * batch_size,
                        temperature=0.7
                    )
                    response_str = response.choices[0].message.content
                    if "</think>" in response_str:
                        response_str = response_str.split("</think>")[-1].strip()
                    text = response_str.replace('\\"', "'").replace('json','').replace('`','').strip().replace("\\'", "").strip()
                    if text.count("'") > text.count('"'):
                        text = text.replace('"', "")
                        text = text.replace("'", '"')
                    reformat_response = json.loads(text)
                    break
                except json.decoder.JSONDecodeError as e:
                    print(f'JsonError: {e}\nText: {text}\nRetry: {retries}')
                    retries += 1
                    continue
                except Exception as e:
                    error_msg = self._handle_api_error(e, f"Batch queries: {query_batch}")
                    print(f"API Error in batch {batch_idx//batch_size + 1}: {error_msg}")
                    retries += 1
                    continue
            
            # Parse results for this batch
            if reformat_response is None:
                # All retries failed, add error messages for this batch
                print(f"Batch {batch_idx//batch_size + 1} failed after {retries} retries, adding error responses")
                for _ in range(len(query_batch)):
                    res.append("ERROR_BATCH_FAILED: Failed to get response after retries")
            else:
                # Use local index i (0, 1, 2, ...) for each batch
                for i in range(len(query_batch)):
                    try:
                        try:
                            res.append(str(reformat_response[str(i)]).replace('(a)', '').replace('(b)', '').strip())
                        except:
                            res.append(', '.join([str(n) for n in list(reformat_response[str(i)].values())]))
                    except:
                        res.append("ERROR_PARSING: Failed to parse response")
        return res