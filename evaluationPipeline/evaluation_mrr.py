import json
import os
import sys
from tqdm import tqdm

# Add project root to sys.path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import config as CONFIG
import utils

try:
    from reponsePipeline.rrf import fuse_responses
except ImportError:
    # If specific path needed (e.g. running from root)
    from reponsePipeline.rrf import fuse_responses

QA_FILE = "files/questionanswers.json"
METADATA_FILE = "files/metadata.json"
TOP_N = 10  # We check top 10 results for MRR

def load_data():
    qa_data = utils.fetch_qa_pairs()
     
    metadata = utils.fetch_metadata()
    # Create lookup chunk_id -> url
    chunk_to_url = {item['chunk_id']: item['url'] for item in metadata}
    
    return qa_data, chunk_to_url

def calculate_mrr():
    qa_data, chunk_to_url = load_data()
    
    total_rr = 0
    count = 0
    
    print(f"Calculating MRR for {len(qa_data)} questions (checking Top {TOP_N} results)...")
    
    for entry in tqdm(qa_data):
        question = entry['question']
        ground_truth_url = entry['source_url']
        
        # Get retrieval results
        try:
            # fuse_responses returns list of (chunk_id, score)
            results = fuse_responses(question, top_n=TOP_N)
        except Exception as e:
            print(f"Error retrieving for QID {entry['id']}: {e}")
            entry['retrieval_rank'] = -1
            entry['mrr_score'] = 0.0
            continue
            
        rank = -1
        rr = 0.0
        
        # Find first matching URL
        # ranks are 1-based
        for i, (chunk_id, score) in enumerate(results, start=1):
            retrieved_url = chunk_to_url.get(chunk_id)
            if retrieved_url == ground_truth_url:
                rank = i
                rr = 1.0 / rank
                break
        
        entry['retrieval_rank'] = rank
        entry['mrr_score'] = rr
        
        total_rr += rr
        count += 1
        
    avg_mrr = total_rr / count if count > 0 else 0
    print(f"\nFinal MRR: {avg_mrr:.4f}")
    
    # Save updated JSON
    with open(QA_FILE, 'w', encoding='utf-8') as f:
        json.dump(qa_data, f, indent=4)
    print(f"Updated {QA_FILE} with MRR scores.")

if __name__ == "__main__":
    calculate_mrr()
