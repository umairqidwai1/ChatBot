import time
from pinecone import Pinecone, ServerlessSpec

# Only need to run this once (already ran this no need to run again)

# Initialize client
pc = Pinecone(api_key="pcsk_5GNKLW_UXkcnZpG21PouoC5n7zhpefqKH8xaefi6k184m3kezNrcsP6XQuWxq1nX1qxRXe", environment="us-west1-gcp")

index_name = "news"

pc.create_index(
    name=index_name,
    dimension=1024,
    metric="cosine",
    spec=ServerlessSpec(cloud="aws", region="us-east-1")
)
while not pc.describe_index(index_name).status["ready"]:
    time.sleep(1)

index = pc.Index(index_name)

data = [
    {"id": "1",  "text": "AI Breakthrough Sets New Milestone in Healthcare"},
    {"id": "2",  "text": "Stock Markets Rally Amid Economic Recovery Hopes"},
    {"id": "3",  "text": "Global Climate Summit Targets Net Zero by 2050"},
    {"id": "4",  "text": "Cryptocurrency Prices Plummet After Regulatory Crackdown"},
    {"id": "5",  "text": "New COVID-19 Variant Raises Global Concerns"},
    {"id": "6",  "text": "Tech Giants Invest Billions in Quantum Computing"},
    {"id": "7",  "text": "Oil Prices Surge as OPEC Cuts Production"},
    {"id": "8",  "text": "Electric Vehicle Sales Hit Record High in 2025"},
    {"id": "9",  "text": "US Fed Signals Possible Interest Rate Hike"},
    {"id": "10", "text": "Breakthrough in Cancer Research Promises New Treatment"},
    {"id": "11", "text": "Massive Earthquake Strikes Coastal City, Thousands Affected"},
    {"id": "12", "text": "Apple Unveils Next-Gen iPhone with AI Features"},
    {"id": "13", "text": "World Cup 2026: Host Cities Announced"},
    {"id": "14", "text": "China Launches New Space Station Module Successfully"},
    {"id": "15", "text": "Bitcoin Surpasses $100,000 Mark for First Time"},
    {"id": "16", "text": "UN Warns of Severe Drought Impact in Africa"},
    {"id": "17", "text": "Major Data Breach Exposes Millions of User Accounts"},
    {"id": "18", "text": "Global Art Market Booms with Record Auction Sales"},
    {"id": "19", "text": "Scientists Discover Water on Mars"},
    {"id": "20", "text": "Hollywood Strike Ends as New Deal Is Reached"},
    {"id": "21", "text": "New AI Tool Transforms the Education Sector"},
    {"id": "22", "text": "Global Inflation Slows, But Food Prices Remain High"},
    {"id": "23", "text": "Breakthrough Battery Technology Doubles EV Range"},
    {"id": "24", "text": "Floods Devastate South Asian Region, Relief Efforts Underway"},
    {"id": "25", "text": "New Trade Agreement Signed Between EU and Japan"},
    {"id": "26", "text": "NASA's Artemis Mission Successfully Lands on the Moon"},
    {"id": "27", "text": "Major Airline Files for Bankruptcy Amid Soaring Costs"},
    {"id": "28", "text": "Global Renewable Energy Capacity Hits All-Time High"},
    {"id": "29", "text": "Scientists Warn of Accelerating Ice Melt in Antarctica"},
    {"id": "30", "text": "Meta Launches New Virtual Reality Platform for Workspaces"}
]

embeddings = pc.inference.embed(
    model="multilingual-e5-large",
    inputs=[d['text'] for d in data],
    parameters={"input_type": "passage", "truncate": "END"}
)

print(embeddings[0])


# Wait for the index to be ready
while not pc.describe_index(index_name).status['ready']:
    time.sleep(1)

index = pc.Index(index_name)

vectors = []
for d, e in zip(data, embeddings):
    vectors.append({
        "id": d['id'],
        "values": e['values'],
        "metadata": {'text': d['text']}
    })

index.upsert(
    vectors=vectors,
    namespace="ns1"
)

print(index.describe_index_stats())
