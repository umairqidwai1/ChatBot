import os
from fastapi import FastAPI, HTTPException, Depends, Security
from fastapi.security.api_key import APIKeyHeader
from pydantic import BaseModel
from search import answer_query

API_KEY = os.getenv("CHAT_API_KEY")
api_key_header = APIKeyHeader(name="X-API-Key", auto_error=False)

async def get_api_key(api_key: str = Security(api_key_header)):
    if api_key == API_KEY:
        return api_key
    raise HTTPException(
        status_code=403,
        detail="Could not validate credentials"
    )

app = FastAPI()

class QueryRequest(BaseModel):
    query: str

class AnswerResponse(BaseModel):
    answer: str

@app.post("/chat", response_model=AnswerResponse, dependencies=[Depends(get_api_key)])
async def chat(user_query: QueryRequest):
    if not user_query.query.strip():
        raise HTTPException(400, "Query cannot be empty")
    try:
        answer = answer_query(user_query.query)
    except RuntimeError as err:
        raise HTTPException(500, str(err))
    return AnswerResponse(answer=answer)
