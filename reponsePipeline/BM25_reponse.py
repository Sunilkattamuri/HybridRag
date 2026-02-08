import os
from rank_bm25 import BM25Okapi
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import utils
import pickle
import config

from nltk.tokenize import word_tokenize

def reponse_BM25(query, top_n=5):
    
    # Load the BM25 index from disk
    with open(config.BM25_INDEX_PATH, 'rb') as f:
        bm25 = pickle.load(f)

    # Fetch documents from metadata json file
    corpus_data = utils.fetch_metadata()

    # Tokenize the query
    tokenized_query = utils.preprocess_text(query)

    # Get BM25 scores
    scores = bm25.get_scores(tokenized_query)

    # Pair each document chunk with its BM25 score
    scored_corpus = []
    for idx, score in enumerate(scores):
        scored_corpus.append((corpus_data[idx]['chunk_id'], score))

    # Sort by score (descending) and return top-k results
    # Higher BM25 score = better match
    sparse_hits = sorted(scored_corpus, key=lambda x: x[1], reverse=True)[:top_n]

    return sparse_hits

if __name__ == "__main__":
    sample_query = "What is data privacy?"
    top_indices, scores = reponse_BM25(sample_query, top_n=3)
    print(f"Top document indices for query '{sample_query}': {top_indices}")
    print(f"Scores: {[scores[i] for i in top_indices]}")
   