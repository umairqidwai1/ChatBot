import os
import time
import logging
from dotenv import load_dotenv
from pinecone import Pinecone, ServerlessSpec
from langchain.text_splitter import RecursiveCharacterTextSplitter
from pinecone.openapi_support.exceptions import PineconeApiException
from langchain_community.document_loaders import DirectoryLoader, TextLoader

# Only need to run this once (already ran this no need to run again)

# Setup logging
logging.basicConfig(
    filename="logs/upsert.log",
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s"
)

load_dotenv()  

# Initialize client
pc = Pinecone(
    api_key=os.getenv("PINECONE_API_KEY"),
    environment="us-west1-gcp"
)
index_name = "week-2"

try:
    pc.create_index(
        name=index_name,
        dimension=1024,
        metric="cosine",
        spec=ServerlessSpec(cloud="aws", region="us-east-1")
    )

    while not pc.describe_index(index_name).status["ready"]:
        time.sleep(1)

    index = pc.Index(index_name)
except (PineconeApiException, Exception) as e:
    logging.error(f"Index {index_name} already exists.")
    os._exit(1)


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

logging.info(f"Total chunks: {len(chunks)}")

data = [
    {"id": str(i), "text": chunk.page_content}
    for i, chunk in enumerate(chunks)
]
# Create embeddings for the chunks
EMBED_BATCH_SIZE = 95
texts = [d["text"] for d in data]
embeddings = []

# Use the Pinecone inference API to create embeddings
try: 
    for start in range(0, len(texts), EMBED_BATCH_SIZE):
        batch = texts[start : start + EMBED_BATCH_SIZE]
        embedding = pc.inference.embed(
            model="multilingual-e5-large",
            inputs=batch,
            parameters={"input_type": "passage", "truncate": "END"}
        )
        embeddings.extend(embedding)

    logging.info(embeddings[0])
except PineconeApiException as e:
    logging.error(f"Pinecone embed error: {e}")
    os._exit(1)

# Build the vectors to upsert
vectors = []
for chunk_payloads, embedding_result in zip(data, embeddings):
    vectors.append({
        "id": chunk_payloads['id'],
        "values": embedding_result['values'],
        "metadata": {'text': chunk_payloads['text']}
    })

# Upsert the vectors into the index
try:
    UPSERT_BATCH_SIZE = 200
    for start in range(0, len(vectors), UPSERT_BATCH_SIZE):
        batch = vectors[start : start + UPSERT_BATCH_SIZE]
        index.upsert(vectors=batch, namespace="ns1")
        logging.info(f"Upserted vectors {start + 1}–{start + len(batch)}")

    logging.info(index.describe_index_stats())
except PineconeApiException as e:
    logging.error(f"Pinecone upsert error: {e}")
    os._exit(1)
