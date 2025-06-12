import os
import re
import csv
import duckdb
import logging
import pandas as pd
import openai
from openai import OpenAI
from pinecone import Pinecone
from typing import List, Tuple
from dotenv import load_dotenv
from presidio_analyzer import AnalyzerEngine
from presidio_anonymizer import AnonymizerEngine
from pinecone.openapi_support.exceptions import PineconeApiException

load_dotenv()

# Presidio setup
analyzer = AnalyzerEngine()
anonymizer = AnonymizerEngine()

# Setup logging
os.makedirs("logs", exist_ok=True)
logging.basicConfig(
    filename="logs/search.log",
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s"
)

MAX_HISTORY = 5
SESSION_FILE = "sessions.csv"
# Initialize session file if it doesn't exist and add header
if not os.path.exists(SESSION_FILE) or os.stat(SESSION_FILE).st_size == 0:
    with open(SESSION_FILE, mode="w", newline='', encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["session_id", "question", "context", "answer"])


# Initialize clients
client = OpenAI(api_key=os.environ["OPENAI_API_KEY"])
pc = Pinecone(
    api_key=os.getenv("PINECONE_API_KEY"),
    environment="us-east-1"
)
index = pc.Index("yq1-transcripts")

# Function to detect PII in text
def detect_pii_entities(text: str):
    return analyzer.analyze(text, language='en')

# Function to extract start seconds from text
def extract_start_sec(text: str) -> int:
    match = re.search(r"\[\s*([0-9.]+)\s*–", text)
    if match:
        try:
            return int(float(match.group(1)))
        except:
            return 0
    return 0

# Function to anonymize text
def anonymize_text(text: str) -> str:
    entities = detect_pii_entities(text)
    result = anonymizer.anonymize(text=text, analyzer_results=entities)
    return result.text

# Function to embed the query
def embed_query(query: str) -> List[float]:
    try:
        response = openai.embeddings.create(
            model="text-embedding-3-large",
            input=[query]
        )
        return response.data[0].embedding
    except PineconeApiException as e:
        logging.error(f"Pinecone embed error: {e}")
        return None
    
# Function to query Pinecone index
def search_pinecone(embedding: List[float], top_k: int = 3) -> List[dict]:
    try:
        results = index.query(
            namespace="ns1",
            vector=embedding,
            top_k=top_k,
            include_values=False,
            include_metadata=True
        )
        return [match["metadata"] for match in results.get("matches", [])]

    except PineconeApiException as e:
        logging.error(f"Pinecone query error: {e}")
        return []

# Fuction called by api.py to answer user text queries
def answer_query(query: str, session_id: str) -> str:
    # Run PII detection before continuing (anonymize text)
    query = anonymize_text(query)

    # Load history and build context
    history = load_session_history(session_id)
    history_str = "\n\n".join([
        f"Question: {question}\n\nContext: {context}\n\nAnswer: {answer}" 
        for question, context, answer in history
    ]) or "No prior chat history."

    # Embed the query
    embedding = embed_query(query)
    if not embedding:
        return "Sorry, I couldn't process your request... (Backend Error - Embedding)"

    # Query Pinecone
    pinecone_results = search_pinecone(embedding)
    if not pinecone_results:
        return "Sorry, I couldn't process your request... (Backend Error - Search Pinecone)"

    # Build context from pinecone results
    context = "\n\n---\n\n".join([r.get("text", "") for r in pinecone_results])
    sources = []
    for r in pinecone_results:
        link = r.get("Link")
        text = r.get("text", "")
        if link:
            start_sec = extract_start_sec(text)
            sources.append(f"[Watch here]({link}&t={start_sec})")


    final_user_prompt = f"""You are an assistant for Islamic content that uses Shaykh Dr. Yasir Qadhi's videos. Based on the following conversation history and provided context, answer the user's latest question.

Conversation history:
{history_str}

---

Context:
{context}

---

User's question:
{query}
"""
    
    # Send to LLM
    try:
        chat = client.chat.completions.create(

            model="gpt-4o-mini",
            messages=[
                {
                    "role":"system",
                    "content":"""
                        You are an Islamic Assistant that uses Shaykh Dr. Yasir Qadhi's videos to answer questions.

                        • Base every answer **only** on the transcript context provided.  
                        • If context does not answer the question, say “I'm not certain from these transcripts.”  
                        • If the question is off-topic, reply: “Sorry, I can only help with questions about Shaykh Yasir Qadhi's videos.”  
                        • Be concise, respectful, and format in Markdown.  
                        """
                },
                {
                    "role": "user", 
                    "content": final_user_prompt
                },

            ],
            temperature=0.3
        )
        answer = chat.choices[0].message.content
    except Exception as e:
        logging.error(f"OpenAI text query error: {e}")
        return "Sorry, I couldn't process your request... (Backend Error - LLM)"
    
    # Append to history and trim to last 10
    history.append((query, context, answer))
    history = history[-MAX_HISTORY:]
    save_session_history(session_id, history)

    if sources:
        answer += "\n\n**Sources:**\n" + "\n".join(f"- {link}" for link in sources)

    return answer

def load_session_history(session_id: str) -> List[Tuple[str, str, str]]:
    if not os.path.exists(SESSION_FILE):
        return []

    session_history = []
    with open(SESSION_FILE, newline='', encoding='utf-8') as f:
        reader = csv.reader(f)
        next(reader, None)  # Skip header row
        for row in reader:
            if len(row) == 4:
                row_session_id, question, context, answer = row
                if row_session_id == session_id:
                    session_history.append((question, context, answer))
    
    return session_history


def save_session_history(session_id: str, history: List[Tuple[str, str, str]]):
    existing_rows = []
    header = ["session_id", "question", "context", "answer"]
    
    # Load all existing data except for the current session
    if os.path.exists(SESSION_FILE):
        with open(SESSION_FILE, newline='', encoding='utf-8') as f:
            reader = csv.reader(f)
            next(reader, None)  # Skip header
            for row in reader:
                if len(row) == 4 and row[0] != session_id:
                    existing_rows.append(row)

    # Add new history for the current session
    session_rows = [
        [session_id, question, context, answer]
        for question, context, answer in history
    ]
    
    # Write combined data back to file
    with open(SESSION_FILE, mode="w", newline='', encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(header)
        writer.writerows(existing_rows + session_rows)


# For debugging purposes, can remove later
if __name__ == "__main__":
    query = input("\nYou: ").strip()
    session_id = "manual_run"
    answer = answer_query(query, session_id)
    print(f"\nAI: {answer}")
