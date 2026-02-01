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


def fuse_responses(query, top_n=5):
    dense_results = dense_response(query, top_n=top_n)
    sparse_results = reponse_BM25(query, top_n=top_n)
    fused_results = reciprocal_rank_fusion(dense_results, sparse_results, k=60, top_n=top_n)
    return fused_results

if __name__ == "__main__":
    # Example usage
    sample_query = "What is data privacy?"
    dense_example = dense_response(sample_query, top_n=3)
    
    sparse_example = reponse_BM25(sample_query, top_n=3)
    
    fused_results = reciprocal_rank_fusion(dense_example, sparse_example, k=60, top_n=3)
    print("Fused Results:", fused_results)