import streamlit as st
import time
import sys
import os

# Ensure project root is in sys.path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import config
from ingestionPipeline.ingest_pipeline import ingest_pipeline
from reponsePipeline.llm_rag_response import llm_rag_response, get_context_from_ids, get_chunk_details
from reponsePipeline.rrf import fuse_responses

# Set page configuration
st.set_page_config(page_title="HybridRAG Search", layout="wide")

st.title("HybridRAG: Intelligent Search System")

# Function to run ingestion
def run_ingestion():
    if "ingestion_done" not in st.session_state:
        with st.spinner("Running Ingestion Pipeline... This may take a while as we process URLs and chunks."):
            try:
                # Redirect stdout to capture progress (optional, but good for debugging/user feedback)
                # For now, we rely on the spinner.
                ingest_pipeline()
                st.session_state["ingestion_done"] = True
                st.success("Ingestion Pipeline Completed Successfully!")
            except Exception as e:
                st.error(f"Error during ingestion: {e}")
                st.stop()

# Trigger ingestion on start
# Trigger ingestion on start
# run_ingestion()

# Sidebar options
with st.sidebar:
    st.header("Configuration")
    if st.button("Run Ingestion Pipeline"):
        run_ingestion()

# Main Search Interface
# For testing, we allow UI to show even if ingestion isn't strictly "done" in this session,
# assuming data persists in ChromaDB.
if True: # st.session_state.get("ingestion_done"):
    st.divider()
    
    query = st.text_area("Enter your query:", height=100)
    
    if st.button("Generate Answer"):
        if query.strip():
            with st.spinner("Generating Response..."):
                start_time = time.time()
                
                # 1. Retrieval
                try:
                    fused_results = fuse_responses(query, top_n=config.TOP_K_RESULTS)
                    
                    # 2. Get Context
                    context = get_context_from_ids(fused_results)
                    chunk_details = get_chunk_details(fused_results)
                    
                    # 3. Generate LLM Response
                    answer = llm_rag_response(context, query)
                    
                    end_time = time.time()
                    elapsed_time = end_time - start_time
                    
                    # Display Results
                    st.subheader("Generated Answer")
                    st.markdown(answer)
                    
                    st.info(f"Response Time: {elapsed_time:.2f} seconds")
                    
                    # Display Retrieved Chunks
                    st.subheader("Retrieved Context & Sources")
                    
                    for i, detail in enumerate(chunk_details):
                        with st.expander(f"Chunk {i+1} (RRF Score: {detail['rrf_score']:.4f})"):
                            st.markdown(f"**Source:** {detail['metadata'].get('url', 'N/A')}")
                            st.markdown(f"**Title:** {detail['metadata'].get('title', 'N/A')}")
                            st.markdown("**Content:**")
                            st.text(detail['text'])
                            
                except Exception as e:
                    st.error(f"An error occurred during processing: {e}")
        else:
            st.warning("Please enter a query.")