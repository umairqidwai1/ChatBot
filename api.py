import os
import csv
from fastapi import FastAPI, HTTPException, Depends, Security, Request, Header
from fastapi.security.api_key import APIKeyHeader
from pydantic import BaseModel
from search import answer_query
from typing import List, Tuple

API_KEY = os.getenv("CHAT_API_KEY")
api_key_header = APIKeyHeader(name="X-API-Key", auto_error=False)

async def get_api_key(api_key: str = Security(api_key_header)):
    if api_key == API_KEY:
        return api_key
    raise HTTPException(status_code=403, detail="Could not validate credentials")

MAX_HISTORY = 10
SESSION_DIR = "sessions"

os.makedirs(SESSION_DIR, exist_ok=True)

app = FastAPI()

class QueryRequest(BaseModel):
    query: str

class AnswerResponse(BaseModel):
    answer: str

@app.post("/chat", response_model=AnswerResponse, dependencies=[Depends(get_api_key)])
async def chat(
    request: Request, 
    user_query: QueryRequest,
    session_id: str = Header(..., alias="Session-ID"),
    ):
    session_id = request.headers.get("Session-ID")
    if not session_id:
        raise HTTPException(400, "Session-ID header is required")
    if not user_query.query.strip():
        raise HTTPException(400, "Query cannot be empty")

    # Load history and build context
    history = load_session_history(session_id)
    history_context_str = "\n".join([f"Question: {question}\nAnswer: {answer}" for question, answer in history])
    
    # combine history context with the new query
    if history_context_str:
        combined_query = f"{history_context_str}\n\nNow answer this new question:\n{user_query.query}"
    else:
        combined_query = user_query.query
    print(combined_query)
    
    # Get llm response
    try:
        answer = answer_query(combined_query)
    except Exception as e:
        raise HTTPException(500, f"Error processing query: {str(e)}")

    # Append and trim to last 10
    history.append((user_query.query, answer))
    history = history[-MAX_HISTORY:]
    save_session_history(session_id, history)

    return AnswerResponse(answer=answer)


def load_session_history(session_id: str) -> List[Tuple[str, str]]:
    path = os.path.join(SESSION_DIR, f"{session_id}.csv")
    if not os.path.exists(path):
        return []
    with open(path, newline='', encoding='utf-8') as f:
        reader = csv.reader(f)
        return [(row[0], row[1]) for row in reader if len(row) == 2]

def save_session_history(session_id: str, history: List[Tuple[str, str]]):
    path = os.path.join(SESSION_DIR, f"{session_id}.csv")
    with open(path, mode="w", newline='', encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerows(history)