# index_kb.py
import os
import json
from tqdm import tqdm
from pinecone import Pinecone, ServerlessSpec
from google.cloud import aiplatform
from langchain_google_vertexai import VertexAIEmbeddings

# --- Configuration ---

# Embedding Model
EMBEDDING_MODEL_NAME = "textembedding-gecko@003" # Using a stable and effective model

# Dataset
DATASET_PATH = "self_critique_loop_dataset.json"

# --- Initialization ---
print("Initializing clients and models...")
# Initialize Vertex AI
aiplatform.init(project=PROJECT_ID, location=LOCATION)

# Initialize Vertex AI Embeddings
embeddings = VertexAIEmbeddings(model_name=EMBEDDING_MODEL_NAME)

# Initialize Pinecone
pc = Pinecone(api_key=PINECONE_API_KEY)

# --- Main Indexing Logic ---
def create_pinecone_index():
    """Creates a Pinecone index if it doesn't already exist."""
    if PINECONE_INDEX_NAME not in pc.list_indexes().names():
        print(f"Creating new Pinecone index: {PINECONE_INDEX_NAME}")
        pc.create_index(
            name=PINECONE_INDEX_NAME,
            dimension=768,  # Gemini embeddings have a dimension of 768
            metric="cosine",
            spec=ServerlessSpec(
                cloud='gcp',
                region='us-central1'
            )
        )
        print("Index created successfully.")
    else:
        print(f"Index '{PINECONE_INDEX_NAME}' already exists.")

def load_and_prepare_data():
    """Loads the KB from the JSON file."""
    print(f"Loading knowledge base from {DATASET_PATH}...")
    with open(DATASET_PATH, 'r') as f:
        data = json.load(f)
    # The data is a dict, we need the list of entries
    documents = data.get("entries", [])
    print(f"Loaded {len(documents)} documents.")
    return documents

def embed_and_upsert_data(documents):
    """Generates embeddings and upserts data into Pinecone in batches."""
    print("Connecting to Pinecone index...")
    index = pc.Index(PINECONE_INDEX_NAME)

    batch_size = 100 # Process documents in batches to avoid overwhelming APIs
    print(f"Embedding and upserting {len(documents)} documents in batches of {batch_size}...")

    for i in tqdm(range(0, len(documents), batch_size)):
        batch_docs = documents[i:i + batch_size]
        
        # Prepare content for embedding
        contents_to_embed = [f"{doc['topic']}: {doc['content']}" for doc in batch_docs]
        
        # Generate embeddings
        embedded_vectors = embeddings.embed_documents(contents_to_embed)
        
        # Prepare vectors for upsert
        vectors_to_upsert = []
        for j, doc in enumerate(batch_docs):
            vectors_to_upsert.append({
                "id": doc["id"],
                "values": embedded_vectors[j],
                "metadata": {
                    "text": doc["content"],
                    "topic": doc["topic"]
                }
            })
            
        # Upsert batch to Pinecone
        index.upsert(vectors=vectors_to_upsert)
        
    print("\nData embedding and upserting complete.")
    print(f"Index stats: {index.describe_index_stats()}")

if __name__ == "__main__":
    create_pinecone_index()
    kb_documents = load_and_prepare_data()
    if kb_documents:
        embed_and_upsert_data(kb_documents)
    else:
        print("No documents found to process.")