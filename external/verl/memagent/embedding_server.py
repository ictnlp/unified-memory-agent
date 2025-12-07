# embedding_server.py
from fastapi import FastAPI
from pydantic import BaseModel
from typing import List, Union
from sentence_transformers import SentenceTransformer
import time

app = FastAPI()
model = SentenceTransformer('all-MiniLM-L6-v2')

class EmbeddingRequest(BaseModel):
    input: Union[str, List[str]]
    model: str = "all-MiniLM-L6-v2"

class EmbeddingResponse(BaseModel):
    object: str = "list"
    data: List[dict]
    model: str
    usage: dict

@app.post("/v1/embeddings")
async def create_embeddings(request: EmbeddingRequest):
    # 处理输入
    if isinstance(request.input, str):
        texts = [request.input]
    else:
        texts = request.input
    
    # 生成 embeddings
    embeddings = model.encode(texts)
    
    # 构造 OpenAI 格式的响应
    data = []
    for i, embedding in enumerate(embeddings):
        data.append({
            "object": "embedding",
            "embedding": embedding.tolist(),
            "index": i
        })
    
    return EmbeddingResponse(
        data=data,
        model=request.model,
        usage={
            "prompt_tokens": sum(len(text.split()) for text in texts),
            "total_tokens": sum(len(text.split()) for text in texts)
        }
    )

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8080)