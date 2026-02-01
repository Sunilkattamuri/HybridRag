import chromadb
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import config

def dense_response(query, top_n=5):
    # Initialize ChromaDB client
    client = chromadb.PersistentClient(path=config.CHROMA_DB_PATH)

    # Get the collection
    collection = client.get_collection(name=config.COLLECTION_NAME)

    # Perform the query
    results = collection.query(
        query_texts=[query],
        n_results=top_n,
        include=["metadatas", "distances", "documents"]
    )

    # Process and format results
    dense_hits = []
    if results['ids'] and len(results['ids']) > 0:
        for i in range(len(results['ids'][0])):
            chunk_id = results['ids'][0][i]
            # In ChromaDB with cosine space, distances are 1 - similarity
            # So lower distance = higher similarity
            score = results['distances'][0][i]
            dense_hits.append((chunk_id, score))
    
    return dense_hits

if __name__ == "__main__":
    sample_query = "What is data privacy?"
    results = dense_response(sample_query, top_n=3)
    print(f"Top document indices for query '{sample_query}': {results}")