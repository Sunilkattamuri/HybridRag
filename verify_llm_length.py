
import sys
import os

# Add project root to sys.path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from reponsePipeline.llm_rag_response import llm_rag_response
import config

def verify_response_length():
    print(f"Current MAX_NEW_TOKENS_LONG: {config.MAX_NEW_TOKENS_LONG}")
    
    # Mock context and query
    context = "Artificial intelligence (AI) is intelligence associated with machines and software. " * 20  # Make it long enough
    query = "Explain what Artificial Intelligence is in detail."
    
    response = llm_rag_response(context, query)
    print("\n--- LLM Response ---")
    print(response)
    print("--------------------")
    print(f"Response Length (chars): {len(response)}")

if __name__ == "__main__":
    verify_response_length()
