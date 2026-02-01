import json
import os
import sys

# Add project root to sys.path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import utils

try:
    from xhtml2pdf import pisa
except ImportError:
    pisa = None

QA_FILE = "files/questionanswers.json"
ABLATION_FILE = "files/ablation_results.json"
REPORT_HTML = "files/evaluation_report.html"
REPORT_PDF = "files/evaluation_report.pdf"
PLOTS_DIR = "files/plots"

def generate_report():
    qa_data = utils.fetch_qa_pairs()
    
    # Calculate Summaries
    avg_mrr = sum(e.get('mrr_score', 0) for e in qa_data) / len(qa_data)
    avg_semantic = sum(e.get('semantic_score', 0) for e in qa_data) / len(qa_data)
    avg_bleu = sum(e.get('bleu_score', 0) for e in qa_data) / len(qa_data)
    avg_latency = sum(e.get('latency', 0) for e in qa_data) / len(qa_data)
    avg_judge = sum(e.get('llm_judge_score', 0) for e in qa_data) / len(qa_data)
    
    avg_faith = sum(e.get('faithfulness_score', 0) for e in qa_data) / len(qa_data)
    
    # Ablation
    ablation_html = ""
    if os.path.exists(ABLATION_FILE):
        with open(ABLATION_FILE, 'r') as f:
            ablation = json.load(f)
        
        rows = ""
        for method, data in ablation.items():
            rows += f"<tr><td>{method.capitalize()}</td><td>{data['mrr']:.4f}</td><td>{data['hit_rate']:.4f}</td></tr>"
            
        ablation_html = f"""
        <h3>Ablation Study Results</h3>
        <table border="1" cellpadding="5" cellspacing="0">
            <tr><th>Method</th><th>MRR</th><th>Hit Rate</th></tr>
            {rows}
        </table>
        """

    # HTML Template
    html_content = f"""
    <html>
    <head>
        <style>
            body {{ font-family: Arial, sans-serif; }}
            h1 {{ color: #2E86C1; }}
            h2 {{ color: #2874A6; border-bottom: 2px solid #2874A6; }}
            table {{ width: 100%; border-collapse: collapse; margin: 20px 0; }}
            th, td {{ border: 1px solid #ddd; padding: 8px; text-align: left; }}
            th {{ background-color: #f2f2f2; }}
            .metric-box {{ background: #f9f9f9; padding: 15px; border-radius: 5px; margin-bottom: 20px; }}
            img {{ max-width: 100%; height: auto; margin: 20px 0; }}
        </style>
    </head>
    <body>
        <h1>HybridRAG Evaluation Report</h1>
        <p>Generated automatically by the Evaluation Pipeline.</p>
        
        <div class="metric-box">
            <h2>Executive Summary</h2>
            <table style="width:50%">
                <tr><th>Metric</th><th>Score</th></tr>
                <tr><td>Average MRR (Hybrid)</td><td>{avg_mrr:.4f}</td></tr>
                <tr><td>Average Semantic Similarity</td><td>{avg_semantic:.4f}</td></tr>
                <tr><td>Average BLEU Score</td><td>{avg_bleu:.4f}</td></tr>
                <tr><td>Average Faithfulness (NLI)</td><td>{avg_faith:.4f}</td></tr>
                <tr><td>Average LLM Judge Score (1-5)</td><td>{avg_judge:.2f}</td></tr>
                <tr><td>Average Response Time</td><td>{avg_latency:.2f} s</td></tr>
            </table>
        </div>
        
        <h2>Visualizations</h2>
        
        <h3>Metric Distributions</h3>
        <img src="{os.path.abspath(PLOTS_DIR + '/semantic_distribution.png')}">
        
        <h3>Faithfulness Distribution</h3>
        <img src="{os.path.abspath(PLOTS_DIR + '/faithfulness_distribution.png')}">
        
        <h3>Response Time</h3>
        <img src="{os.path.abspath(PLOTS_DIR + '/latency_distribution.png')}">
        
        <h3>Retrieval Ablation</h3>
        <img src="{os.path.abspath(PLOTS_DIR + '/ablation_comparison.png')}">
        
        {ablation_html}
        
        <h2>Error Analysis (Bottom 5 Semantic Scores)</h2>
        <table border="1">
            <tr><th>ID</th><th>Question</th><th>Semantic Score</th><th>Issue</th></tr>
            {"".join([f"<tr><td>{e['id']}</td><td>{e['question']}</td><td>{e['semantic_score']:.4f}</td><td>Low Similarity</td></tr>" 
                      for e in sorted(qa_data, key=lambda x: x.get('semantic_score', 0))[:5]])}
        </table>
        
    </body>
    </html>
    """
    
    # Save HTML
    with open(REPORT_HTML, "w", encoding='utf-8') as f:
        f.write(html_content)
    print(f"Report saved to {REPORT_HTML}")
    
    # Convert to PDF
    if pisa:
        with open(REPORT_PDF, "wb") as pdf_file:
            pisa_status = pisa.CreatePDF(html_content, dest=pdf_file)
        if not pisa_status.err:
            print(f"PDF Report saved to {REPORT_PDF}")
        else:
            print("Error creating PDF")
    else:
        print("xhtml2pdf not installed. PDF generation skipped.")

if __name__ == "__main__":
    generate_report()
