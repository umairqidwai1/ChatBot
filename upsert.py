import os
import time
import logging
import unicodedata, re, pickle
import openai, tiktoken
import pandas as pd
from dotenv import load_dotenv
from pinecone import Pinecone, ServerlessSpec
from langchain.text_splitter import RecursiveCharacterTextSplitter
from pinecone.openapi_support.exceptions import PineconeApiException
from langchain_community.document_loaders import DirectoryLoader, TextLoader

# Only need to run this once (already ran this no need to run again)

# Function to create a slug from a string
def slug(s: str) -> str:
    s = unicodedata.normalize("NFKD", s).encode("ascii", "ignore").decode()
    return re.sub(r"[^a-z0-9]+", "_", s.lower()).strip("_")

EMBED_FILE   = "yq1_embeddings.pkl" 

# Load the CSV file containing metadata
CSV_PATH = "C:/Users/uqidw/Downloads/YQ1_transcripts/yasir_qadhi_videos.csv"
meta_df = pd.read_csv(CSV_PATH)

meta_lookup = {
    slug(row["Title"]): {**row.to_dict()}
    for _, row in meta_df.iterrows()
}

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
index_name = "yq1-transcripts"

try:
    pc.create_index(
        name=index_name,
        dimension=3072,
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
    "C:/Users/uqidw/Downloads/YQ1_transcripts/YQ1_transcripts",
    glob="**/*.txt",
    loader_cls=TextLoader,
    loader_kwargs={
        "encoding": "utf-8"
    }
)
docs = loader.load()  

# Create a lookup dictionary for metadata based on the CSV file
for doc in docs:
    base = slug(os.path.splitext(os.path.basename(doc.metadata["source"]))[0])
    if base in meta_lookup:
        doc.metadata.update(meta_lookup[base])  # ← adds the CSV columns

unmatched = [os.path.basename(d.metadata["source"]) 
             for d in docs if "Link" not in d.metadata]

if unmatched:
    logging.warning(f"{len(unmatched)} transcripts had no CSV match: {unmatched[:5]}")

# Split documents into smaller chunks using RecursiveCharacterTextSplitter
splitter = RecursiveCharacterTextSplitter(
    chunk_size=1200,
    chunk_overlap=120,
    separators=["\n\n", "\n", ". ", " ", ""]
)
chunks = splitter.split_documents(docs)

logging.info(f"Total chunks: {len(chunks)}")

data = [
    {"id": str(i), "text": chunk.page_content, "meta": chunk.metadata}
    for i, chunk in enumerate(chunks)
]

# Create embeddings for the chunks
# ── OpenAI embedding settings ───────────────────────────
openai.api_key = os.getenv("OPENAI_API_KEY")
EMBED_MODEL    = "text-embedding-3-large"
ENC            = tiktoken.encoding_for_model(EMBED_MODEL)
TOKENS_PER_MIN = 3_000_000
EMBED_BATCH    = 95

if os.path.exists(EMBED_FILE):
    logging.info("Loading cached embeddings …")
    with open(EMBED_FILE, "rb") as f:
        cache = pickle.load(f)
    embeddings = cache["embeddings"]
    data       = cache["data"]
    logging.info(f"Loaded {len(embeddings)} embeddings from disk")
else:
    logging.info("Creating embeddings …")
    texts, embeddings = [d["text"] for d in data], []
    for start in range(0, len(texts), EMBED_BATCH):
        batch = texts[start:start+EMBED_BATCH]
        resp  = openai.embeddings.create(model=EMBED_MODEL, input=batch)
        embeddings.extend([r.embedding for r in resp.data])
        # throttle
        batch_tokens = sum(len(ENC.encode(t)) for t in batch)
        time.sleep(batch_tokens / TOKENS_PER_MIN * 60)
    # 🔒 persist to disk
    with open(EMBED_FILE, "wb") as f:
        pickle.dump({"data": data, "embeddings": embeddings}, f)
    logging.info(f"Saved {len(embeddings)} embeddings → {EMBED_FILE}")

# Build the vectors to upsert
vectors = []
for payload, embedding_result in zip(data, embeddings):
    vectors.append({
        "id": payload['id'],
        "values": embedding_result,
        "metadata": {"text": payload["text"], **payload["meta"]}
    })

# Upsert the vectors into the index
try:
    UPSERT_BATCH_SIZE = 30
    for start in range(0, len(vectors), UPSERT_BATCH_SIZE):
        batch = vectors[start : start + UPSERT_BATCH_SIZE]
        index.upsert(vectors=batch, namespace="ns1")
        logging.info(f"Upserted vectors {start + 1}–{start + len(batch)}")

    logging.info(index.describe_index_stats())
except PineconeApiException as e:
    logging.error(f"Pinecone upsert error: {e}")
    os._exit(1)
