import os
from fastapi import FastAPI, HTTPException, Depends, Security, Request, Header
from fastapi.security.api_key import APIKeyHeader
from pydantic import BaseModel
from search import answer_query
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles

app = FastAPI()

API_KEY = os.getenv("CHAT_API_KEY")
api_key_header = APIKeyHeader(name="X-API-Key", auto_error=False)

async def get_api_key(api_key: str = Security(api_key_header)):
    if api_key == API_KEY:
        return api_key
    raise HTTPException(status_code=403, detail="Could not validate credentials")


class QueryRequest(BaseModel):
    query: str

class AnswerResponse(BaseModel):
    answer: str

"""
Expected Response:
{
    "answer": "string"
}
"""
@app.post("/chat", response_model=AnswerResponse, dependencies=[Depends(get_api_key)])
async def chat(
    request: Request, 
    user_query: QueryRequest,
    session_id: str = Header(..., alias="Session-ID"),
    ):
    if not session_id:
        raise HTTPException(400, "Session-ID header is required")
    if not user_query.query.strip():
        raise HTTPException(400, "Query cannot be empty")

    
    # Get llm response
    try:
        answer = answer_query(user_query.query, session_id)
    except Exception as e:
        raise HTTPException(500, f"Error processing query: {str(e)}")

    return AnswerResponse(answer=answer)

app.mount("/static", StaticFiles(directory="static"), name="static")

@app.get("/", response_class=HTMLResponse)
async def index():
    with open("static/chat.html", "r", encoding="utf-8") as f:
        return f.read()