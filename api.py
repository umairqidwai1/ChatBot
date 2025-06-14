import os
import pandas as pd
from pydantic import BaseModel
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles
from fastapi.security.api_key import APIKeyHeader
from search import answer_query, answer_csv_query
from fastapi import FastAPI, HTTPException, Depends, Security, Request, Header, File, UploadFile, Form

app = FastAPI()

# Setup saving csv file in folder
UPLOAD_FOLDER = "uploads"
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

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
# Chat endpoint
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

# CSV query endpoint
@app.post("/csv-query", response_model=AnswerResponse, dependencies=[Depends(get_api_key)])
async def csv_query(
    request: Request,
    query: str = Form(...),
    file: UploadFile = File(None),
    session_id: str = Header(..., alias="Session-ID"),
):
    try:
        session_csv_path = os.path.join(UPLOAD_FOLDER, f"{session_id}.csv")
        # 1. If a file is uploaded, save it for this session
        if file is not None and file.filename:
            with open(session_csv_path, "wb") as f_out:
                content = await file.read()
                f_out.write(content)
        # 2. If file not provided now, look for session's saved csv
        if not os.path.exists(session_csv_path):
            raise HTTPException(400, "No CSV file found for this session. Please upload one.")
        df = pd.read_csv(session_csv_path)
        answer = answer_csv_query(query, df)
    except Exception as e:
        raise HTTPException(500, f"Error processing CSV query: {str(e)}")

    return AnswerResponse(answer=answer)

app.mount("/static", StaticFiles(directory="static"), name="static")

@app.get("/", response_class=HTMLResponse)
async def index():
    with open("static/chat.html", "r", encoding="utf-8") as f:
        return f.read()