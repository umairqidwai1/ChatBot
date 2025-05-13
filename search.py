import os
from dotenv import load_dotenv
from pinecone import Pinecone
from pinecone.openapi_support.exceptions import PineconeApiException

# Run this code to search for the most relevant news articles based on user inputted query

load_dotenv()  
pc = Pinecone(
    api_key=os.getenv("PINECONE_API_KEY"),
    environment="us-west1-gcp"
)

index = pc.Index("week-2")

# Uncomment the following line to see index stats
#print(index.describe_index_stats())

query = input("Enter your search query: ")

try:
    embedding = pc.inference.embed(
        model="multilingual-e5-large",
        inputs=[query],
        parameters={"input_type": "query"}
    )
except PineconeApiException as e:
    print(f"Pinecone embed error: {e}")
    exit(1)

try: 
    results = index.query(
        namespace="ns1",
        vector=embedding[0].values,
        top_k=3,
        include_values=False,
        include_metadata=True
    )
except PineconeApiException as e:
    print(f"Pinecone query error: {e}")
    exit(1)

print(results)