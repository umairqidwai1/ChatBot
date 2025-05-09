from pinecone import Pinecone

# Run this code to search for the most relevant news articles based on user inputted query

pc = Pinecone(api_key="pcsk_5GNKLW_UXkcnZpG21PouoC5n7zhpefqKH8xaefi6k184m3kezNrcsP6XQuWxq1nX1qxRXe", environment="us-west1-gcp")

index = pc.Index("news")

# Uncomment the following line to see index stats
#print(index.describe_index_stats())

query = input("Enter your search query: ")

embedding = pc.inference.embed(
    model="multilingual-e5-large",
    inputs=[query],
    parameters={
        "input_type": "query"
    }
)

results = index.query(
    namespace="ns1",
    vector=embedding[0].values,
    top_k=3,
    include_values=False,
    include_metadata=True
)

print(results)
