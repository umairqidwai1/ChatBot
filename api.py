import os
from pydantic import BaseModel
from fastapi.responses import JSONResponse
from fastapi.security.api_key import APIKeyHeader
from search import answer_query
from typing import List
from fastapi import FastAPI, HTTPException, Depends, Security
import traceback

#Authentication setup
API_KEY = os.getenv("CHAT_API_KEY")
api_key_header = APIKeyHeader(name="X-API-Key", auto_error=False)

async def get_api_key(api_key: str = Security(api_key_header)):
    if api_key == API_KEY:
        return api_key
    raise HTTPException(status_code=403, detail="Could not validate credentials")

app = FastAPI(dependencies=[Depends(get_api_key)])

# === OpenAI-compatible chat/completions endpoint ===
class ChatMessage(BaseModel):
    role: str
    content: str

class ChatCompletionRequest(BaseModel):
    model: str
    messages: List[ChatMessage]
    temperature: float = 0.3

# === OpenAI-compatible chat/completions endpoint ===
@app.post("/v1/chat/completions", tags=["OpenAI Compatibility"])
async def chat_completion(request_data: ChatCompletionRequest):
    messages = [m.dict() for m in request_data.messages]
    try:
        answer = answer_query(messages)
    except Exception as e:
        traceback.print_exc()
        raise HTTPException(500, f"LLM error: {str(e)}")

    return {
        "id": "chatcmpl-local-001",
        "object": "chat.completion",
        "created": 0,
        "model": request_data.model,
        "choices": [
            {
                "index": 0,
                "message": {
                    "role": "assistant",
                    "content": answer
                },
                "finish_reason": "stop"
            }
        ],
        "usage": {
            "prompt_tokens": 0,
            "completion_tokens": 0,
            "total_tokens": 0
        }
    }

# === OpenAI-compatible models endpoint ===
@app.get("/v1/models", tags=["OpenAI Compatibility"])
async def list_models():
    return JSONResponse(content={
        "object": "list",
        "data": [
            {
                "id": "YQ Answers",
                "object": "model",
                "created": 1715200000,
                "owned_by": "local",
                "permission": [],
            },
        ]
    })
