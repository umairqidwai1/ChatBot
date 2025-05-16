import os
import csv
import logging
from openai import OpenAI
from pinecone import Pinecone
from typing import List, Tuple
from dotenv import load_dotenv
from pinecone.openapi_support.exceptions import PineconeApiException

load_dotenv()

# Setup logging
os.makedirs("logs", exist_ok=True)
logging.basicConfig(
    filename="logs/search.log",
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s"
)

MAX_HISTORY = 5
SESSION_FILE = "sessions.csv"
if not os.path.exists(SESSION_FILE):
    open(SESSION_FILE, "w", encoding="utf-8").close()

# Initialize clients
client = OpenAI(api_key=os.environ["OPENAI_API_KEY"])
pc = Pinecone(
    api_key=os.getenv("PINECONE_API_KEY"),
    environment="us-west1-gcp"
)
index = pc.Index("week-2")

# Uncomment the following line to see index stats
#print(index.describe_index_stats())

# Function to embed the query
def embed_query(query: str) -> List[float]:
    try:
        embedding = pc.inference.embed(
            model="multilingual-e5-large",
            inputs=[query],
            parameters={"input_type": "query"}
        )
        return embedding[0].values
    except PineconeApiException as e:
        logging.error(f"Pinecone embed error: {e}")
        return None
    
# Function to query Pinecone index
def search_pinecone(embedding: List[float], top_k: int = 3) -> List[str]:
    try:
        results = index.query(
            namespace="ns1",
            vector=embedding,
            top_k=top_k,
            include_values=False,
            include_metadata=True
        )
        return [match["metadata"]["text"] for match in results.get("matches", [])]
    except PineconeApiException as e:
        logging.error(f"Pinecone query error: {e}")
        return []

# Fuction called by api.py to answer user queries
def answer_query(query: str, session_id: str) -> str:
    # Load history and build context
    history = load_session_history(session_id)
    history_str = "\n\n".join([
        f"Question: {question}\n\nContext: {context}\n\nAnswer: {answer}" 
        for question, context, answer in history
    ]) or "No prior chat history."
    logging.info(f"History context: {history_str}")

    # Embed the query
    embedding = embed_query(query)
    if not embedding:
        return "Sorry, I couldn't process your request... (Backend Error - Embedding)"

    # Query Pinecone
    pinecone_results = search_pinecone(embedding)
    if not pinecone_results:
        return "Sorry, I couldn't process your request... (Backend Error - Search Pinecone)"

    # Build context from pinecone results
    context = "\n\n---\n\n".join(pinecone_results)
    
    final_user_prompt = f"""You are an assistant for AssetSonar. Based on the following conversation history and provided context, answer the user's latest question.

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

            model="gpt-4o",
            messages=[
                {
                    "role": "system", 
                    "content": """You are AssetSonar's helpful AI assistant.

                    - Only answer based on the provided context and question.
                    - Do not make up information.
                    - If context clearly contains the answer, respond accurately.
                    - If the answer isn't found in the context, but it's likely based on history or common AssetSonar knowledge, respond cautiously and state your answer.
                    - For off-topic questions (not about AssetSonar), respond with: “Sorry, I can only help with AssetSonar-related qeustions.”
                    - Be helpful, brief, and clear at all times, and format you answer in markdown format.
                    - You can answer general questions (eg: greetings) in a friendly manner."""
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
        logging.error(f"OpenAI API error: {e}")
        return "Sorry, I couldn't process your request... (Backend Error - LLM)"
    
    # Append to history and trim to last 10
    history.append((query, context, answer))
    history = history[-MAX_HISTORY:]
    save_session_history(session_id, history)

    return answer

def load_session_history(session_id: str) -> List[Tuple[str, str, str]]:
    if not os.path.exists(SESSION_FILE):
        return []

    session_history = []
    with open(SESSION_FILE, newline='', encoding='utf-8') as f:
        reader = csv.reader(f)
        for row in reader:
            if len(row) == 4:
                row_session_id, question, context, answer = row
                if row_session_id == session_id:
                    session_history.append((question, context, answer))
    
    return session_history


def save_session_history(session_id: str, history: List[Tuple[str, str, str]]):
    existing_rows = []
    
    # Load all existing data except for the current session
    if os.path.exists(SESSION_FILE):
        with open(SESSION_FILE, newline='', encoding='utf-8') as f:
            reader = csv.reader(f)
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
        writer.writerows(existing_rows + session_rows)


# For debugging purposes, can remove later
if __name__ == "__main__":
    query = input("\nYou: ").strip()
    session_id = "manual_run"
    answer = answer_query(query, session_id)
    print(f"\nAI: {answer}")