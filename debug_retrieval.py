import sys
import os

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import config
try:
    from reponsePipeline.rrf import fuse_responses
    from reponsePipeline.llm_rag_response import get_context_from_ids, llm_rag_response, get_chunk_details
except ImportError as e:
    import traceback
    traceback.print_exc()
    print(f"Import Error: {e}")
    sys.exit(1)

def debug_query(query):
    print(f"Query: {query}")
    print("-" * 50)
    
    # Search deeper to find the chunk
    SEARCH_DEPTH = 20
    print(f"Running Retrieval (Top {SEARCH_DEPTH})...")
    try:
        fused_results = fuse_responses(query, top_n=SEARCH_DEPTH)
        chunk_details = get_chunk_details(fused_results)
        
        found_rank = -1
        for i, detail in enumerate(chunk_details):
            # Check for unique infobox string
            if "First published: 2000" in detail['text']:
                found_rank = i + 1
                print(f"\n>>> FOUND INFOBOX CHUNK AT RANK: {found_rank} <<<")
                print(f"Score: {detail['rrf_score']}")
                print(f"Content: {detail['text'][:200]}...")
                break
        

        if found_rank == -1:
            print("\n!!! INFOBOX CHUNK NOT FOUND IN TOP 20 !!!")
        else:
            # Check if it fits in config.TOP_K_RESULTS
            if found_rank <= config.TOP_K_RESULTS:
                 print(f"It is within TOP_K ({config.TOP_K_RESULTS}).")
            else:
                 print(f"It is OUTSIDE TOP_K ({config.TOP_K_RESULTS}). Needs TOP_K >= {found_rank}.")

        # --- NEW: Test Generation ---
        print("\n" + "="*50)
        print(f"TESTING LLM GENERATION (Using Top {config.TOP_K_RESULTS} chunks to match App)...")
        
        # Analyze potential distractors in Top 3
        print("--- Top 3 Chunks Analysis ---")
        for i in range(min(3, len(fused_results))):
             cid = fused_results[i][0]
             # fast fetch
             c_context = get_context_from_ids([fused_results[i]])
             print(f"RANK {i+1} [ID: {cid}]: {c_context[:150]}...")
             print("-" * 20)
        
        context_results = fused_results[:config.TOP_K_RESULTS]
        context = get_context_from_ids(context_results)
        print(f"Context Length (chars): {len(context)}")
        
        # Check if our target text is actually IN the context string
        if "First published: 2000" in context:
            print("SUCCESS: Target text 'First published: 2000' IS present in the final context string.")
        else:
            print("FAILURE: Target text 'First published: 2000' is MISSING from the final context string (Truncated?).")

        response = llm_rag_response(context, query)
        print("\nLLM GENERATED ANSWER:")
        print(response)
        print("="*50)

    except Exception as e:
        print(f"Error: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    queries = [
        "Camellia (cipher) first published year?",
        "How does cyber security engineering differ from information privacy?",
        "Who won the Super Bowl in 2030?"
    ]
    
    for q in queries:
        print("\n\n" + "#"*50)
        print(f"DEBUGGING QUERY: {q}")
        print("#"*50)
        debug_query(q)
