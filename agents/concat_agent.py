from typing import List
import json
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
    
    def add_memory(
        self,
        chunk: str
    ):
        self.memory.append(chunk)
    
    def QA(
        self,
        query: str
    ) -> str:
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
                res = res.split("</think>")[1]
                return res.strip()
            else:
                return f"ERROR_THINK_LENGTH_EXCEEDED: The think is too long for the model to process. Think: {res[:100]}..."
        except Exception as e:
            return self._handle_api_error(e, query)
    
    def QA_batch(
        self,
        query_list: list[str],
        batch_size: int=32
    ) -> list[str]:
        if len(query_list) == 1:
            return [self.QA(query_list[0])]
        context = "\n".join(self.memory)
        res = []
        for batch_idx in range(0, len(query_list), batch_size):
            query_batch = query_list[batch_idx : batch_idx+batch_size]
            prompt = f"Your memory:\n{context}\n\n{QA_PROMPT_BATCH}{"\n".join(["%s: %s" % (k, q) for k, q in enumerate(query_batch)])}"

            model = MODEL_NAME_MAP.get(self.model_name, self.model_name)
            retries = 0
            while retries < 3:
                try:
                    response = self.client.chat.completions.create(
                        model=model,
                        messages=[
                            {"role": "user", "content": prompt}
                        ],
                        max_tokens=50 * batch_size,
                        temperature=0.7
                    )
                    text = response.choices[0].message.content.replace('\\"', "'").replace('json','').replace('`','').strip().replace("\\'", "").strip()
                    if text.count("'") > text.count('"'):
                        text = text.replace('"', "")
                        text = text.replace("'", '"')
                    reformat_response = json.loads(text)
                    break
                except json.decoder.JSONDecodeError as e:
                    print(f'JsonError: {e}\nRetry: {retries}')
                    retries += 1
                    continue
                except Exception as e:
                    # Handle API errors by returning error messages for all queries in this batch
                    error_msg = self._handle_api_error(e, f"Batch queries: {query_batch}")
                    return [error_msg] * len(query_list)  # Return error for all queries
            for k in range(batch_idx, batch_idx + batch_size):
                if k >= len(query_batch):
                    break
                try:
                    try:
                        res.append(str(reformat_response[str(k)]).replace('(a)', '').replace('(b)', '').strip())
                    except:
                        res.append(', '.join([str(n) for n in list(reformat_response[str(k)].values())]))
                except:
                    res.append("ERROR_PARSING: Failed to parse response")
        return res