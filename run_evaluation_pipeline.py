import os
import sys

# Add project root to sys.path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from evaluationPipeline.evaluation_metrics import run_evaluation
from evaluationPipeline.evaluation_mrr import calculate_mrr
from evaluationPipeline.ablation_study import run_ablation
from evaluationPipeline.generate_plots import generate_visualizations
from evaluationPipeline.generate_report import generate_report
from evaluationPipeline.QA_generation import generate_qa_dataset

def main():
    print("Starting HybridRAG Automated Evaluation Pipeline...")
    
    # 0. Generate QA Dataset
    print(f"\n{'='*50}")
    print("STEP: Generating QA Dataset (Synthetic Data)")
    print(f"{'='*50}\n")
    generate_qa_dataset()
    
    # 1. Calculate Metrics (Semantic, BLEU, Judge, Latency, Confidence)
    print(f"\n{'='*50}")
    print("STEP: Calculating Base Metrics (Answer Generation, Semantic, BLEU, LLM Judge)")
    print(f"{'='*50}\n")
    run_evaluation()
    
    # 2. Calculate MRR
    print(f"\n{'='*50}")
    print("STEP: Calculating Retrieval Metrics (MRR)")
    print(f"{'='*50}\n")
    calculate_mrr()
    
    # 3. Ablation Studies
    print(f"\n{'='*50}")
    print("STEP: Running Ablation Studies (Dense vs Sparse vs Hybrid)")
    print(f"{'='*50}\n")
    run_ablation()
    
    # 4. Generate Visualizations
    print(f"\n{'='*50}")
    print("STEP: Generating Visualization Plots")
    print(f"{'='*50}\n")
    generate_visualizations()
    
    # 5. Generate Report
    print(f"\n{'='*50}")
    print("STEP: Generating Final PDF/HTML Report")
    print(f"{'='*50}\n")
    generate_report()
    
    print("\n\nPipeline Completed Successfully!")
    print("Check 'files/evaluation_report.pdf' (or .html) for the final output.")

if __name__ == "__main__":
    main()
