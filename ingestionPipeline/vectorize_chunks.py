import os
import sys
import chromadb
from chromadb.utils import embedding_functions
from sentence_transformers import SentenceTransformer

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import config
import utils



def vectorize_data():
    
    corpus_data = utils.fetch_metadata()

    embedding_function = embedding_functions.SentenceTransformerEmbeddingFunction(model_name=config.EMBEDDING_MODEL)

    client = chromadb.PersistentClient(path=config.CHROMA_DB_PATH)

    try:
        client.delete_collection(name=config.COLLECTION_NAME)
        print(f"Existing collection '{config.COLLECTION_NAME}' deleted.")
    except Exception as e:
        print(f"No existing collection '{config.COLLECTION_NAME}' found. Creating a new one.")

    collection = client.get_or_create_collection(name=config.COLLECTION_NAME, 
                                                        embedding_function = embedding_function,
                                                        metadata={"hnsw:space": "cosine"})
    ids = [item['chunk_id'] for item in corpus_data]
    documents = [f"{item['title']} {item['content']}" for item in corpus_data]
    metadatas = [{"title": item['title'], "url": item['url'],
                  "chunk_index": item['metadata']['chunk_index']} for item in corpus_data]
    
    batch_size = 5000
    for i in range(0, len(documents), batch_size):
        batch_ids = ids[i:i + batch_size]
        batch_documents = documents[i:i + batch_size]
        batch_metadatas = metadatas[i:i + batch_size]
        
        collection.add(
            documents=batch_documents,
            metadatas=batch_metadatas,
            ids=batch_ids
        )
        print(f"Added batch {i // batch_size + 1}: {len(batch_documents)} documents")

    print(f"ChromaDB collection created and populated with {len(ids)} total documents.")


if __name__ == "__main__":
        vectorize_data()

