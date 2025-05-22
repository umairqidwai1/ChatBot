import os
import csv
import duckdb
import logging
import pandas as pd
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
    environment="us-west1-gcp"
)
index = pc.Index("week-2")

# Function to detect PII in text
def detect_pii_entities(text: str):
    return analyzer.analyze(text, language='en')

# Function to anonymize text
def anonymize_text(text: str) -> str:
    entities = detect_pii_entities(text)
    result = anonymizer.anonymize(text=text, analyzer_results=entities)
    return result.text

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
    # For debugging purposes, can remove later
    # logging.info(f"History context: {history_str}")

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

            model="gpt-4o-mini",
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
        logging.error(f"OpenAI text query error: {e}")
        return "Sorry, I couldn't process your request... (Backend Error - LLM)"
    
    # Append to history and trim to last 10
    history.append((query, context, answer))
    history = history[-MAX_HISTORY:]
    save_session_history(session_id, history)

    return answer

# Fuction called by api.py to answer user CSV queries
def answer_csv_query(query: str, df: pd.DataFrame) -> str:

    # 1. Ask LLM for the SQL query
    sql_prompt = f"""
You are an expert data analyst. Based on the following CSV data sample, generate a SQL query to answer the user's question.

CSV Data (first 5 rows):
{df.head(5).to_csv(index=False)}

---

User question:
{query}

Respond ONLY with the SQL query.
"""
    
    try:
        chat_sql = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[
                {"role": "system", "content": """You are a SQL expert. 
                 - Respond ONLY with the SQL query, do NOT use code blocks or markdown formatting.
                 - Do not include any other text or comments.
                 - The table name is always df. Do not use any other table name.
                 - Always return at least a 1-sentence summary, even if the data is repetitive or blank.
                 - If the user's question is not about the data, respond with: "I'm sorry, I can only answer questions about the provided data."""},
                {"role": "user", "content": sql_prompt}
            ],
            temperature=0.3
        )
        sql_query = chat_sql.choices[0].message.content.strip()
        logging.info(f"SQL query: {sql_query}")
    except Exception as e:
        logging.error(f"OpenAI CSV query error: {e}")
        return "Sorry, I couldn't process your request... (CSV LLM Error - SQL Query)"

    # 2. Execute the SQL query
    try:
        duckdb.register('df', df)
        result = duckdb.sql(sql_query).to_df()
        duckdb.unregister('df')
        logging.info(f"Result: {result}")
    except Exception as e:
        logging.error(f"SQL execution error: {e}")
        return f"Sorry, there was an error executing the generated SQL query: {str(e)}\n\nThe SQL was:\n{sql_query}"

    # 3. Summarize the result (send back to LLM for explanation)
    result_sample = result.head(5).to_csv(index=False) if not result.empty else "No rows returned."
    summarize_prompt = f"""
User question:
"{query}"

Here is a summary of the sql query results (first 5 rows):
{result_sample}

Please provide a short, clear, and human-readable answer based on the result. If the result is a table, summarize what it shows.
"""
    try:
        chat_summary = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[
                {"role": "system", "content": "You are a data assistant. Summarize results from a CSV/SQL query for a user, briefly and clearly."},
                {"role": "user", "content": summarize_prompt}
            ],
            temperature=0.3
        )
        summary = chat_summary.choices[0].message.content.strip()
    except Exception as e:
        logging.error(f"OpenAI CSV result summary error: {e}")
        return f"Sorry, I couldn't process your request... (CSV LLM Error - Result Summary)"
    
    return summary


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
