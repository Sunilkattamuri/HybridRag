import os
import sys
import time
from pinecone import Pinecone, ServerlessSpec
from sentence_transformers import SentenceTransformer

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import config
import utils

def vectorize_data():
    print("Fetching corpus data...")
    corpus_data = utils.fetch_metadata()
    print(f"Fetched {len(corpus_data)} chunks.")

    # Initialize Sentence Transformer for embedding generation
    print(f"Loading embedding model: {config.EMBEDDING_MODEL}")
    model = SentenceTransformer(config.EMBEDDING_MODEL)

    # Initialize Pinecone
    print("Initializing Pinecone...")
    pc = Pinecone(api_key=config.PINECONE_API_KEY)

    index_name = config.PINECONE_INDEX_NAME

    # Check if index exists, if not create it (Serverless spec examples, adjust if using Pods)
    # Note: Free tier often allows 1 starter index. We assume it exists or we try to create it.
    existing_indexes = [index_info.name for index_info in pc.list_indexes()]
    
    if index_name not in existing_indexes:
        print(f"Index '{index_name}' not found. Creating it...")
        try:
            pc.create_index(
                name=index_name,
                dimension=768, # Dimension for all-mpnet-base-v2
                metric="cosine",
                spec=ServerlessSpec(
                    cloud="aws",
                    region="us-east-1"
                )
            )
            # Wait for index to be ready
            while not pc.describe_index(index_name).status['ready']:
                time.sleep(1)
        except Exception as e:
            print(f"Error creating index (might already exist or limit reached): {e}")
            print("Attempting to proceed with existing index if available...")
    
    index = pc.Index(index_name)
    print(f"Connected to index: {index_name}")

    # Prepare data for upsert
    batch_size = 100 # Pinecone recommends smaller batches than Chroma
    
    total_vectors = 0
    
    for i in range(0, len(corpus_data), batch_size):
        batch = corpus_data[i:i + batch_size]
        
        # Prepare batch data
        ids = [item['chunk_id'] for item in batch]
        texts = [f"{item['title']} {item['content']}" for item in batch]
        
        # Generate embeddings
        embeddings = model.encode(texts).tolist()
        
        # Prepare metadata
        # Pinecone metadata values must be strings, numbers, booleans, or lists of strings
        vectors_to_upsert = []
        for j, item in enumerate(batch):
            metadata = {
                "title": item['title'],
                "url": item['url'],
                "chunk_index": float(item['metadata']['chunk_index']), # Pinecone handles numbers
                "text": texts[j] # Store the actual text for retrieval
            }
            vectors_to_upsert.append({
                "id": ids[j],
                "values": embeddings[j],
                "metadata": metadata
            })
        
        # Upsert
        try:
            index.upsert(vectors=vectors_to_upsert)
            print(f"Upserted batch {i // batch_size + 1}: {len(batch)} vectors")
            total_vectors += len(batch)
        except Exception as e:
            print(f"Error upserting batch {i}: {e}")

    print(f"Pinecone index populated with {total_vectors} total documents.")


if __name__ == "__main__":
        vectorize_data()

