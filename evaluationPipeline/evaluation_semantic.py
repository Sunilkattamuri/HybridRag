from sentence_transformers import SentenceTransformer, util
import sys
import os

# Add project root to sys.path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import config as CONFIG

_model = None

def get_model():
    global _model
    if _model is None:
        print(f"Loading Embedding Model: {CONFIG.EMBEDDING_MODEL}")
        _model = SentenceTransformer(CONFIG.EMBEDDING_MODEL)
    return _model

def calculate_semantic_similarity(reference, candidate):
    """
    Calculate Cosine Similarity between reference and candidate text.
    """
    if not reference or not candidate:
        return 0.0
        
    model = get_model()
    
    # Encode sentences to get their embeddings
    embeddings1 = model.encode(reference, convert_to_tensor=True)
    embeddings2 = model.encode(candidate, convert_to_tensor=True)
    
    # Compute cosine similarity
    score = util.pytorch_cos_sim(embeddings1, embeddings2)
    return float(score.item())
