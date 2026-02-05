import sys
import os

try:
    from dense_response import dense_response
    from BM25_reponse import reponse_BM25
except ImportError:
    from reponsePipeline.dense_response import dense_response
    from reponsePipeline.BM25_reponse import reponse_BM25

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import config
from collections import defaultdict
from sentence_transformers import CrossEncoder

# Global re-ranker model (lazy load)
_reranker = None

def get_reranker():
    global _reranker
    if _reranker is None:
        print(f"Loading Re-ranker: {config.RERANK_MODEL}")
        _reranker = CrossEncoder(config.RERANK_MODEL, max_length=512)
    return _reranker


def reciprocal_rank_fusion(dense_results, sparse_results, k=60, top_n=5):
    """
    Perform Reciprocal Rank Fusion (RRF) on multiple ranked lists.

    Args:
        ranked_lists (list of list): A list containing multiple ranked lists of document IDs.
        k (int): The RRF parameter that controls the influence of lower-ranked documents.

    Returns:
        list: A single ranked list of document IDs after applying RRF.
    """


    # Initialize dictionary to accumulate RRF scores
    rrf_scores = {}
    
    # Process dense retrieval results
    # Assign scores based on inverse rank (with RRF constant k)
    for rank, (chunk_id, _) in enumerate(dense_results, start=1):
        if chunk_id not in rrf_scores:
            rrf_scores[chunk_id] = 0
        # Add RRF contribution: 1 / (k + rank)
        # Lower rank (earlier in list) contributes more
        rrf_scores[chunk_id] += 1 / (k + rank)
    
    # Process sparse retrieval results
    # Documents appearing in both lists get additional score contributions
    for rank, (chunk_id, _) in enumerate(sparse_results, start=1):
        if chunk_id not in rrf_scores:
            rrf_scores[chunk_id] = 0
        rrf_scores[chunk_id] += 1 / (k + rank)
    
    # Sort all documents by their combined RRF score (descending)
    # Higher score = better combined ranking
    sorted_results = sorted(rrf_scores.items(), key=lambda x: x[1], reverse=True)
    
    # Return only the top_n results
    return sorted_results[:top_n]


def rerank_results(query, fused_results, top_n=5):
    """
    Re-ranks the top results from RRF using a Cross-Encoder.
    fused_results is a list of (chunk_id, rrf_score)
    """
    if not fused_results:
        return []

    # Get the text content for the candidates
    # We need to fetch the content from ChromaDB to pass to the CrossEncoder
    try:
        from reponsePipeline.llm_rag_response import get_context_from_ids, get_chunk_details
        
        # Helper to fetch details efficiently
        details = get_chunk_details(fused_results)
        
        # Prepare pairs for (Query, Document)
        pairs = []
        valid_details = []
        
        for item in details:
            pairs.append([query, item['text']])
            valid_details.append(item)
            
        if not pairs:
            return fused_results[:top_n]
            
        # Predict scores
        model = get_reranker()
        scores = model.predict(pairs)
        
        # Attach new scores
        reranked = []
        for i, score in enumerate(scores):
            reranked.append((valid_details[i]['id'], float(score)))
            
        # Sort by new Cross-Encoder score
        reranked.sort(key=lambda x: x[1], reverse=True)
        
        return reranked[:top_n]
        
    except ImportError:
        print("Could not import helpers for re-ranking. Skipping.")
        return fused_results[:top_n]
    except Exception as e:
        print(f"Error during re-ranking: {e}")
        return fused_results[:top_n]


def fuse_responses(query, top_n=5):
    # Fetch a larger pool for RRF
    retrieval_limit = 100
    dense_results = dense_response(query, top_n=retrieval_limit)
    sparse_results = reponse_BM25(query, top_n=retrieval_limit)
    
    # 1. RRF Fusion (Stage 1) -> Get Top 50 Candidates
    # We get more than top_n here to give the re-ranker some options
    candidates_count = 50 
    fused_results = reciprocal_rank_fusion(dense_results, sparse_results, k=60, top_n=candidates_count)
    
    # 2. Re-ranking (Stage 2) -> Get Top N
    final_results = rerank_results(query, fused_results, top_n=top_n)
    
    return final_results

if __name__ == "__main__":
    # Example usage
    sample_query = "What is data privacy?"
    dense_example = dense_response(sample_query, top_n=3)
    
    sparse_example = reponse_BM25(sample_query, top_n=3)
    
    fused_results = reciprocal_rank_fusion(dense_example, sparse_example, k=60, top_n=3)
    print("Fused Results:", fused_results)