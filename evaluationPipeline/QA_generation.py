import json
import random
import os
import sys
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
import torch
from tqdm import tqdm

# Add project root to sys.path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import config as CONFIG
import utils

# Configuration
OUTPUT_FILE = "files/questionanswers.json"
METADATA_FILE = "files/metadata.json"
TOTAL_QA_PAIRS = 100

def load_text_chunks():
    """Load text chunks from metadata.json."""
    chunks = utils.fetch_metadata()
    print(f"Loaded {len(chunks)} chunks from {METADATA_FILE}")
    return chunks

def init_llm():
    """Initialize the LLM and tokenizer."""
    print(f"Loading model: {CONFIG.LLM_RAG_MODEL_NAME}")
    tokenizer = AutoTokenizer.from_pretrained(CONFIG.LLM_RAG_MODEL_NAME)
    model = AutoModelForSeq2SeqLM.from_pretrained(CONFIG.LLM_RAG_MODEL_NAME)
    
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")
    model.to(device)
    
    return tokenizer, model, device

def generate_question(tokenizer, model, device, context, q_type):
    """Generate a question based on the context and type."""
    prompts = {
        "factual": f"Context: {context}\n\nTask: Generate a factual question based on the text above.",
        "comparative": f"Context: {context}\n\nTask: Generate a comparative question (comparing two things mentioned) based on the text above.",
        "inferential": f"Context: {context}\n\nTask: Generate an inferential question (requires reasoning) based on the text above.",
        "multi-hop": f"Context: {context}\n\nTask: Generate a complex multi-hop question based on the text above."
    }
    
    prompt = prompts.get(q_type, prompts["factual"])
    
    inputs = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=1024).to(device)
    
    outputs = model.generate(
        **inputs, 
        max_new_tokens=64,
        do_sample=True,
        temperature=0.7,
        top_p=0.9
    )
    
    return tokenizer.decode(outputs[0], skip_special_tokens=True).strip()

def generate_answer(tokenizer, model, device, context, question):
    """Generate an answer based on the context and question."""
    prompt = f"Context: {context}\n\nQuestion: {question}\n\nTask: Answer the question based on the context above."
    
    inputs = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=1024).to(device)
    
    outputs = model.generate(
        **inputs, 
        max_new_tokens=128,
        do_sample=True,
        temperature=0.7,
        top_p=0.9
    )
    
    return tokenizer.decode(outputs[0], skip_special_tokens=True).strip()

def generate_qa_dataset():
    tokenizer, model, device = init_llm()
    all_chunks = load_text_chunks()
    
    # Filter for reasonable content length
    valid_chunks = [c for c in all_chunks if len(c.get('content', '')) > 200]
    
    qa_dataset = []
    generated_count = 0
    
    q_types = ["factual", "comparative", "inferential", "multi-hop"]
    counts = {t: 0 for t in q_types}
    target_per_type = TOTAL_QA_PAIRS // len(q_types)
    
    with tqdm(total=TOTAL_QA_PAIRS, desc="Generating QA Pairs") as pbar:
        while generated_count < TOTAL_QA_PAIRS:
            chunk = random.choice(valid_chunks)
            
            # Select a type that hasn't reached the target yet
            available_types = [t for t in q_types if counts[t] < target_per_type]
            if not available_types:
                break
            
            q_type = random.choice(available_types)
            
            # Step 1: Generate Question
            question = generate_question(tokenizer, model, device, chunk['content'], q_type)
            
            if not question or len(question) < 10: 
                continue

            # Step 2: Generate Answer
            answer = generate_answer(tokenizer, model, device, chunk['content'], question)
            
            if answer and len(answer) > 5:
                qa_entry = {
                    "id": generated_count + 1,
                    "category": q_type,
                    "question": question,
                    "ground_truth_answer": answer,
                    "source_chunk_id": chunk['chunk_id'],
                    "source_title": chunk['title'],
                    "source_url": chunk['url'],
                    "context": chunk['content']
                }
                qa_dataset.append(qa_entry)
                generated_count += 1
                counts[q_type] += 1
                pbar.update(1)
                pbar.set_postfix(counts)
            else:
                # Retry if generation failed or formatting was wrong
                continue
                
    # Save to file
    os.makedirs(os.path.dirname(OUTPUT_FILE), exist_ok=True)
    with open(OUTPUT_FILE, 'w', encoding='utf-8') as f:
        json.dump(qa_dataset, f, indent=4)
        
    print(f"Successfully generated {len(qa_dataset)} Q&A pairs and saved to {OUTPUT_FILE}")

if __name__ == "__main__":
    generate_qa_dataset()
