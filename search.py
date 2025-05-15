import os
import csv
import logging
from openai import OpenAI
from pinecone import Pinecone
from typing import List, Tuple
from dotenv import load_dotenv
from pinecone.openapi_support.exceptions import PineconeApiException

# Run this code to search for the most relevant news articles based on user inputted query

load_dotenv()

# Setup logging
os.makedirs("logs", exist_ok=True)
logging.basicConfig(
    filename="logs/search.log",
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s"
)

MAX_HISTORY = 10
SESSION_DIR = "sessions"
os.makedirs(SESSION_DIR, exist_ok=True)

# Initialize clients
client = OpenAI(api_key=os.environ["OPENAI_API_KEY"])
pc = Pinecone(
    api_key=os.getenv("PINECONE_API_KEY"),
    environment="us-west1-gcp"
)
index = pc.Index("week-2")

# Uncomment the following line to see index stats
#print(index.describe_index_stats())

def answer_query(query: str, session_id: str) -> str:
    # Load history and build context
    history = load_session_history(session_id)
    history_str = "\n\n".join([
        f"Question: {question}\n\nContext: {context}\n\nAnswer: {answer}" 
        for question, context, answer in history
    ]) or "No prior chat history."
    logging.info(f"History context: {history_str}")

    # Embed the query
    try:
        embedding = pc.inference.embed(
            model="multilingual-e5-large",
            inputs=[query],
            parameters={"input_type": "query"}
        )
    except PineconeApiException as e:
        logging.error(f"Pinecone embed error: {e}")
        return "Sorry, I couldn't process your request... (Backend Error - Embedding)"
    
    # Query Pinecone
    try: 
        pinecone_results = index.query(
            namespace="ns1",
            vector=embedding[0].values,
            top_k=3,
            include_values=False,
            include_metadata=True
        )
    except PineconeApiException as e:
        logging.error(f"Pinecone query error: {e}")
        return "Sorry, I couldn't process your request... (Backend Error - Query)"

    # Build context from pinecone results
    raw_context=[m["metadata"]["text"] for m in pinecone_results.get("matches", [])]
    context = "\n\n---\n\n".join(raw_context)
    
    final_user_prompt = f"""
    Chat History:
    {history_str}

    -----

    Retrieved Context:
    {context}

    -----
    
    Current Question:
    {query}"""

    # Send to LLM
    try:
        chat = client.chat.completions.create(

            model="gpt-4o-mini",
            messages=[
                {
                    "role": "system",
                    "content": """You are AssetSonar's AI assistant.

                    Your job is to help users by answering their questions based on the provided retrieved context.

                    - If the context contains the answer, use it to respond clearly and accurately.
                    - If the context does not contain the answer, say: "Sorry, I can't answer that question."
                    - You may still respond to general greetings like "hello", "hi", or "thanks" in a friendly way.
                    - Keep responses professional, helpful, concise and simple"."""
                },
                {
                    "role": "user", 
                    "content": final_user_prompt
                },

            ],
            temperature=0
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
    path = os.path.join(SESSION_DIR, f"{session_id}.csv")
    if not os.path.exists(path):
        return []
    with open(path, newline='', encoding='utf-8') as f:
        reader = csv.reader(f)
        return [(row[0], row[1], row[2]) for row in reader if len(row) == 3]


def save_session_history(session_id: str, history: List[Tuple[str, str, str]]):
    path = os.path.join(SESSION_DIR, f"{session_id}.csv")
    with open(path, mode="w", newline='', encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerows(history)
        
if __name__ == "__main__":
    query = input("\nYou: ").strip()
    session_id = "manual_run"
    answer = answer_query(query, session_id)
    print(f"\nAI: {answer}")