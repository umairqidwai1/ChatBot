import os
from openai import OpenAI
from pinecone import Pinecone
from dotenv import load_dotenv
from pinecone.openapi_support.exceptions import PineconeApiException

# Run this code to search for the most relevant news articles based on user inputted query

load_dotenv()
# Initialize clients
client = OpenAI(api_key=os.environ["OPENAI_API_KEY"])
pc = Pinecone(
    api_key=os.getenv("PINECONE_API_KEY"),
    environment="us-west1-gcp"
)
index = pc.Index("week-2")

# Uncomment the following line to see index stats
#print(index.describe_index_stats())

def answer_query(query: str) -> str:
    try:
        embedding = pc.inference.embed(
            model="multilingual-e5-large",
            inputs=[query],
            parameters={"input_type": "query"}
        )
    except PineconeApiException as e:
        print(f"Pinecone embed error: {e}")

    try: 
        pinecone_results = index.query(
            namespace="ns1",
            vector=embedding[0].values,
            top_k=3,
            include_values=False,
            include_metadata=True
        )
    except PineconeApiException as e:
        print(f"Pinecone query error: {e}")

    context=[m["metadata"]["text"] for m in pinecone_results.get("matches", [])]
    context = "\n\n---\n\n".join(context)
        
    try:
        chat = client.chat.completions.create(

            model="gpt-4o-mini",
            messages=[
                {
                    "role": "system",
                    "content": "You are an AI assistant that answers questions using the provided context. "
                            "If the answer cannot be found in the context, say Sorry I can't answer that question."
                },
                {
                    "role": "user", 
                    "content": f"Context:\n{context}\n\nQuestion: {query}"
                },

            ],
            temperature=0
        )
        return chat.choices[0].message.content
    except Exception as e:
        print(f"OpenAI API error: {e}")
        return "Sorry, I couldn't process your request..."

if __name__ == "__main__":
    query = input("\nYou: ").strip()
    answer = answer_query(query)
    print(f"\nAI: {answer}")