from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
import torch
import sys
import os
import re

# Add project root to sys.path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import config as CONFIG

_tokenizer = None
_model = None

def get_judge_model():
    global _tokenizer, _model
    if _model is None:
        print(f"Loading Judge Model: {CONFIG.LLM_RAG_MODEL_NAME}")
        _tokenizer = AutoTokenizer.from_pretrained(CONFIG.LLM_RAG_MODEL_NAME)
        _model = AutoModelForSeq2SeqLM.from_pretrained(CONFIG.LLM_RAG_MODEL_NAME)
        device = "cuda" if torch.cuda.is_available() else "cpu"
        _model.to(device)
    return _tokenizer, _model

def evaluate_by_llm(question, ground_truth, candidate):
    """
    Uses the LLM to evaluate the quality of the candidate answer compared to the ground truth.
    Returns a score from 1 to 5.
    """
    if not candidate:
        return 0
        
    tokenizer, model = get_judge_model()
    device = model.device
    
    # Prompt for evaluation
    prompt = f"""
You are an impartial judge. Evaluate the quality of the provided Answer based on the Ground Truth for the given Question.
Score the Answer from 1 to 5, where 1 is incorrect and 5 is perfect.

Question: {question}
Ground Truth: {ground_truth}
Answer: {candidate}

Task: Provide a score (1-5) and a brief reason.
Output Format:
Score: [Score]
Reason: [Reason]
"""
    
    inputs = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=1024).to(device)
    
    outputs = model.generate(
        **inputs, 
        max_new_tokens=64,
        do_sample=True,
        temperature=0.1 # Low temp for consistency
    )
    
    output_text = tokenizer.decode(outputs[0], skip_special_tokens=True).strip()
    
    # Parse score
    try:
        match = re.search(r"Score:\s*(\d)", output_text)
        if match:
            score = int(match.group(1))
            return min(max(score, 1), 5)
        else:
            # Fallback parsing if format is loose
            # Look for just a number
            numbers = re.findall(r"\b[1-5]\b", output_text)
            if numbers:
                return int(numbers[0])
            return 3 # Neutral default if parsing fails
    except Exception:
        return 3

