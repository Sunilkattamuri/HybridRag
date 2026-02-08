import re
import re
from pinecone import Pinecone

from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
import torch
import sys
import os
try:
    from rrf import fuse_responses
except ImportError:
    from reponsePipeline.rrf import fuse_responses
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import config

model_name= config.LLM_RAG_MODEL_NAME
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForSeq2SeqLM.from_pretrained(model_name)

def get_context_from_ids(fused_results):
    """
    Retrieves the actual text content for the given chunk IDs from Pinecone.
    """
    # Extract just the chunk IDs from the RRF results (which are tuples of (id, score))
    chunk_ids = [result[0] for result in fused_results]
    
    if not chunk_ids:
        return ""
    
    # Initialize Pinecone client
    pc = Pinecone(api_key=config.PINECONE_API_KEY)
    index = pc.Index(config.PINECONE_INDEX_NAME)
    
    # Fetch from Pinecone
    response = index.fetch(ids=chunk_ids)
    vectors = response.vectors
    
    # Construct the context string preserving the order from fused_results
    context_parts = []
    for chunk_id in chunk_ids:
        if chunk_id in vectors:
            # We stored text in metadata
            text = vectors[chunk_id].metadata.get('text', '')
            context_parts.append(text)
    
    return "\n\n".join(context_parts)

def get_chunk_details(fused_results):
    """
    Retrieves detailed information (text, metadata) for the given chunk IDs.
    Returns a list of dictionaries.
    """
    chunk_ids = [result[0] for result in fused_results]
    rrf_scores = {result[0]: result[1] for result in fused_results}
    
    if not chunk_ids:
        return []
    
    pc = Pinecone(api_key=config.PINECONE_API_KEY)
    index = pc.Index(config.PINECONE_INDEX_NAME)
    
    response = index.fetch(ids=chunk_ids)
    vectors = response.vectors
    
    details = []
    
    for chunk_id in chunk_ids:
        if chunk_id in vectors:
            vector_data = vectors[chunk_id]
            metadata = vector_data.metadata
            text = metadata.get('text', '')
            
            detail = {
                "id": chunk_id,
                "text": text,
                "metadata": metadata,
                "rrf_score": rrf_scores.get(chunk_id, 0.0)
            }
            details.append(detail)
            
    return details

import time
import numpy as np

def llm_rag_response(context, query, max_length=config.MAX_NEW_TOKENS_LONG, return_metadata=False):

     # Optimized prompt for Flan-T5 (Instruction -> Context -> Question)
     prompt = f"""Answer the following question using the context below.
If the answer is not in the context, YOU MUST RESPOND with "NOT_FOUND_IN_CONTEXT".
Do not make up an answer.

Question: {query}

Context:
{context}

Answer:"""
      
     # encdoing the given prompt which has context and query
     inputs = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=2048)

     # Move inputs to the same device as the model
     inputs = {key: value.to(model.device) for key, value in inputs.items()}
     
     start_time = time.time()
     
     # generating the output
     outputs = model.generate(
         **inputs, 
         max_new_tokens=max_length, 
         num_beams=5,
         do_sample=False,
         early_stopping=True,
         length_penalty=2.0,
         no_repeat_ngram_size=2,
         return_dict_in_generate=True,
         output_scores=True
     )
     
     end_time = time.time()
     latency = end_time - start_time

     # decoding the output
     answer = tokenizer.decode(outputs.sequences[0], skip_special_tokens=True).strip()

     if return_metadata:
         # Calculate confidence
         # outputs.sequences_scores contains the log probability of the chosen sequence
         if hasattr(outputs, 'sequences_scores') and outputs.sequences_scores is not None:
             # This is the log prob of the entire sequence. We can normalize by length or just use it.
             # Ideally we want average probability per token.
             # sequences_scores is sum of log probs (normalized by length if length_penalty used).
             confidence = torch.exp(outputs.sequences_scores[0]).item()
         else:
             confidence = 0.0 # Fallback
             
         return {
             "answer": answer,
             "confidence": confidence,
             "latency": latency
         }
     
     return answer

if __name__ == "__main__":
    sample_query = "What is data privacy?"
    
    fused_results = fuse_responses(sample_query, top_n=3)
    context = get_context_from_ids(fused_results)

    llm_rag_response_text = llm_rag_response(context=context, query=sample_query)
    print(f"LLM RAG Response: {llm_rag_response_text}")