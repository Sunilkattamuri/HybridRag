import json
import os
import sys
from tqdm import tqdm

# Add project root to sys.path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import config as CONFIG
import utils

try:
    from reponsePipeline.rrf import fuse_responses, dense_response, reponse_BM25
except ImportError:
    from reponsePipeline.rrf import fuse_responses, dense_response, reponse_BM25

QA_FILE = "files/questionanswers.json"
ABLATION_FILE = "files/ablation_results.json"
TOP_N = 10

def run_ablation():
    print("Starting Ablation Study...")
    
    qa_data = utils.fetch_qa_pairs()
    metadata = utils.fetch_metadata()
    chunk_to_url = {item['chunk_id']: item['url'] for item in metadata}
    
    results = {
        "dense": {"mrr": 0, "hits": 0},
        "sparse": {"mrr": 0, "hits": 0},
        "hybrid": {"mrr": 0, "hits": 0}
    }
    
    count = 0
    
    for entry in tqdm(qa_data, desc="Ablation"):
        question = entry['question']
        ground_truth_url = entry['source_url']
        
        # 1. Hybrid (RRF)
        # fuse_responses returns list of (chunk_id, score)
        hybrid_res = fuse_responses(question, top_n=TOP_N)
        
        # 2. Dense (Embeddings)
        # dense_response returns list of (chunk_id, score)
        dense_res = dense_response(question, top_n=TOP_N)
        
        # 3. Sparse (BM25)
        # reponse_BM25 returns list of (chunk_id, score)
        sparse_res = reponse_BM25(question, top_n=TOP_N)
        
        # Helper to calc MRR
        def calc_rank_rr(retrieved_list):
            for i, (chunk_id, _) in enumerate(retrieved_list, start=1):
                url = chunk_to_url.get(chunk_id)
                if url == ground_truth_url:
                    return 1.0 / i
            return 0.0

        rr_hybrid = calc_rank_rr(hybrid_res)
        rr_dense = calc_rank_rr(dense_res)
        rr_sparse = calc_rank_rr(sparse_res)
        
        results["hybrid"]["mrr"] += rr_hybrid
        results["dense"]["mrr"] += rr_dense
        results["sparse"]["mrr"] += rr_sparse
        
        if rr_hybrid > 0: results["hybrid"]["hits"] += 1
        if rr_dense > 0: results["dense"]["hits"] += 1
        if rr_sparse > 0: results["sparse"]["hits"] += 1
        
        count += 1
        
    # Calculate Averages
    if count > 0:
        for key in results:
            results[key]["mrr"] = results[key]["mrr"] / count
            results[key]["hit_rate"] = results[key]["hits"] / count
            
    print("\nAblation Results (MRR):")
    print(f"Dense:  {results['dense']['mrr']:.4f}")
    print(f"Sparse: {results['sparse']['mrr']:.4f}")
    print(f"Hybrid: {results['hybrid']['mrr']:.4f}")
    
    # Save results
    with open(ABLATION_FILE, 'w', encoding='utf-8') as f:
        json.dump(results, f, indent=4)
    print(f"Saved ablation results to {ABLATION_FILE}")

if __name__ == "__main__":
    run_ablation()
