import json
import os
import sys
from tqdm import tqdm

# Add project root to sys.path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import config as CONFIG
import utils

try:
    from reponsePipeline.llm_rag_response import llm_rag_response
    # Also need retrieval context logic if not baked into llm_rag_response fully?
    # llm_rag_response handles retrieval internally.
    from reponsePipeline.rrf import fuse_responses
    from reponsePipeline.llm_rag_response import get_chunk_details
except ImportError:
    from reponsePipeline.llm_rag_response import llm_rag_response

# Import Metric Modules
from evaluationPipeline import evaluation_semantic
from evaluationPipeline import evaluation_bleu
from evaluationPipeline import evaluation_llm_judge
from evaluationPipeline import evaluation_faithfulness

QA_FILE = "files/questionanswers.json"

def load_qa_pairs():
    return utils.fetch_qa_pairs()

def run_evaluation():
    qa_data = load_qa_pairs()
    
    print(f"Starting evaluation metrics for {len(qa_data)} questions Link...")
    
    semantic_scores = []
    bleu_scores = []
    faithfulness_scores = []
    
    for entry in tqdm(qa_data):
        question = entry['question']
        ground_truth = entry['ground_truth_answer']
        
        # 1. Retrieve Context
        fused_results = fuse_responses(question, top_n=5)

        metadata = utils.fetch_metadata()
        chunk_map = {m['chunk_id']: m['content'] for m in metadata}
        
        context_parts = []
        for chunk_id, _ in fused_results[:3]: # Top 3 for context
            if chunk_id in chunk_map:
                context_parts.append(chunk_map[chunk_id])
        
        context = "\n\n".join(context_parts)
        
        # 2. Generate Candidate Answer
        try:
            # Request metadata (Confidence, Latency)
            response_data = llm_rag_response(context, question, return_metadata=True)
            
            if isinstance(response_data, dict):
                candidate = response_data['answer']
                confidence = response_data['confidence']
                latency = response_data['latency']
            else:
                candidate = response_data
                confidence = 0.0
                latency = 0.0
                
        except Exception as e:
            print(f"Error generating answer for QID {entry['id']}: {e}")
            candidate = "Error"
            confidence = 0.0
            latency = 0.0
        
        # Store basic data
        entry['candidate_answer'] = candidate
        entry['confidence'] = confidence
        entry['latency'] = latency
        
        # 3. Compute Metrics
        sem_score = evaluation_semantic.calculate_semantic_similarity(ground_truth, candidate)
        bl_score = evaluation_bleu.calculate_bleu_score(ground_truth, candidate)
        judge_score = evaluation_llm_judge.evaluate_by_llm(question, ground_truth, candidate)
        faith_score = evaluation_faithfulness.calculate_faithfulness(context, candidate)
        
        entry['semantic_score'] = sem_score
        entry['bleu_score'] = bl_score
        entry['llm_judge_score'] = judge_score
        entry['faithfulness_score'] = faith_score
        
        semantic_scores.append(sem_score)
        bleu_scores.append(bl_score)
        faithfulness_scores.append(faith_score)
        
    # Averages
    avg_sem = sum(semantic_scores) / len(semantic_scores) if semantic_scores else 0
    avg_bleu = sum(bleu_scores) / len(bleu_scores) if bleu_scores else 0
    avg_faith = sum(faithfulness_scores) / len(faithfulness_scores) if faithfulness_scores else 0
    
    print("\nEvaluation Results:")
    print(f"Average Semantic Similarity: {avg_sem:.4f}")
    print(f"Average BLEU Score:        {avg_bleu:.4f}")
    print(f"Average Faithfulness Score:  {avg_faith:.4f}")
    
    # Save results
    with open(QA_FILE, 'w', encoding='utf-8') as f:
        json.dump(qa_data, f, indent=4)
    print(f"Updated {QA_FILE} with evaluation scores.")

if __name__ == "__main__":
    run_evaluation()
