import matplotlib.pyplot as plt
import json
import os
import sys

# Add project root to sys.path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import utils

QA_FILE = "files/questionanswers.json"
ABLATION_FILE = "files/ablation_results.json"
PLOTS_DIR = "files/plots"

def generate_visualizations():
    os.makedirs(PLOTS_DIR, exist_ok=True)
    
    qa_data = utils.fetch_qa_pairs()
    
    # 1. Metric Distributions (Semantic Score)
    semantic_scores = [entry.get('semantic_score', 0) for entry in qa_data]
    plt.figure(figsize=(10, 6))
    plt.hist(semantic_scores, bins=20, color='skyblue', edgecolor='black')
    plt.title('Distribution of Semantic Similarity Scores')
    plt.xlabel('Score')
    plt.ylabel('Count')
    plt.savefig(f"{PLOTS_DIR}/semantic_distribution.png")
    plt.close()
    
    # 2. Latency Distribution
    latencies = [entry.get('latency', 0) for entry in qa_data]
    plt.figure(figsize=(10, 6))
    plt.hist(latencies, bins=20, color='orange', edgecolor='black')
    plt.title('Response Time Distribution')
    plt.xlabel('Time (seconds)')
    plt.ylabel('Count')
    plt.savefig(f"{PLOTS_DIR}/latency_distribution.png")
    plt.close()
    
    # 3. Faithfulness Distribution
    faith_scores = [entry.get('faithfulness_score', 0) for entry in qa_data]
    plt.figure(figsize=(10, 6))
    plt.hist(faith_scores, bins=10, color='lightgreen', edgecolor='black', range=(0, 1))
    plt.title('Faithfulness Score Distribution (NLI)')
    plt.xlabel('Score (0-1)')
    plt.ylabel('Count')
    plt.savefig(f"{PLOTS_DIR}/faithfulness_distribution.png")
    plt.close()
    
    # 4. Ablation Results
    if os.path.exists(ABLATION_FILE):
        with open(ABLATION_FILE, 'r') as f:
            ablation = json.load(f)
            
        methods = list(ablation.keys())
        mrr_scores = [ablation[m]['mrr'] for m in methods]
        
        plt.figure(figsize=(10, 6))
        plt.bar(methods, mrr_scores, color=['#FF9999', '#66B2FF', '#99FF99'])
        plt.title('Retrieval Ablation Study (MRR)')
        plt.xlabel('Method')
        plt.ylabel('MRR Score')
        plt.savefig(f"{PLOTS_DIR}/ablation_comparison.png")
        plt.close()

    print(f"Plots saved to {PLOTS_DIR}")

if __name__ == "__main__":
    generate_visualizations()
