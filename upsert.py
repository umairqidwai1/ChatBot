import time
from pinecone import Pinecone, ServerlessSpec
from langchain_community.document_loaders import DirectoryLoader, TextLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter

# Only need to run this once (already ran this no need to run again)

# Initialize client
pc = Pinecone(
    api_key="pcsk_5GNKLW_UXkcnZpG21PouoC5n7zhpefqKH8xaefi6k184m3kezNrcsP6XQuWxq1nX1qxRXe", 
    environment="us-west1-gcp"
)

index_name = "week-2"

pc.create_index(
    name=index_name,
    dimension=1024,
    metric="cosine",
    spec=ServerlessSpec(cloud="aws", region="us-east-1")
)

while not pc.describe_index(index_name).status["ready"]:
    time.sleep(1)

index = pc.Index(index_name)

# Load documents from the specified directory
loader = DirectoryLoader(
    "C:/Users/uqidw/Downloads/as_blog_content/content",
    glob="**/*.txt",
    loader_cls=TextLoader,
    loader_kwargs={
        "encoding": "utf-8"
    }
)
docs = loader.load()  

# Split documents into smaller chunks using RecursiveCharacterTextSplitter
splitter = RecursiveCharacterTextSplitter(
    chunk_size=1200,
    chunk_overlap=120,
    separators=["\n\n", "\n", ". ", " ", ""]
)
chunks = splitter.split_documents(docs)

print(f"Total chunks: {len(chunks)}")

data = [
    {"id": str(i), "text": chunk.page_content}
    for i, chunk in enumerate(chunks)
]
# Create embeddings for the chunks
EMBED_BATCH_SIZE = 95
texts = [d["text"] for d in data]
embeddings = []

for start in range(0, len(texts), EMBED_BATCH_SIZE):
    batch = texts[start : start + EMBED_BATCH_SIZE]
    embs = pc.inference.embed(
        model="multilingual-e5-large",
        inputs=batch,
        parameters={"input_type": "passage", "truncate": "END"}
    )
    embeddings.extend(embs)

print(embeddings[0])

vectors = []
for d, e in zip(data, embeddings):
    vectors.append({
        "id": d['id'],
        "values": e['values'],
        "metadata": {'text': d['text']}
    })

UPSERT_BATCH_SIZE = 200
for start in range(0, len(vectors), UPSERT_BATCH_SIZE):
    batch = vectors[start : start + UPSERT_BATCH_SIZE]
    index.upsert(vectors=batch, namespace="ns1")
    print(f"Upserted vectors {start + 1}–{start + len(batch)}")

print(index.describe_index_stats())