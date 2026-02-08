import os
import sys
from pinecone import Pinecone
from sentence_transformers import SentenceTransformer

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import config

# Initialize model once to avoid reloading on every call if possible effectively
# But for script usage, it will load each time. 
model = SentenceTransformer(config.EMBEDDING_MODEL)

def dense_response(query, top_n=5):
    # Initialize Pinecone client
    pc = Pinecone(api_key=config.PINECONE_API_KEY)
    index = pc.Index(config.PINECONE_INDEX_NAME)

    # Generate embedding for the query
    query_embedding = model.encode(query).tolist()

    # Perform the query
    results = index.query(
        vector=query_embedding,
        top_k=top_n,
        include_metadata=False, # We only need IDs and scores here usually, but keeping false for speed unless needed
        include_values=False
    )
    
    # Process and format results
    dense_hits = []
    
    # Pinecone results structure: {'matches': [{'id': '...', 'score': 0.9, ...}]}
    for match in results['matches']:
        chunk_id = match['id']
        score = match['score'] # Cosine similarity
        dense_hits.append((chunk_id, score))
    
    return dense_hits

if __name__ == "__main__":
    sample_query = "What is data privacy?"
    results = dense_response(sample_query, top_n=3)
    print(f"Top document indices for query '{sample_query}': {results}")