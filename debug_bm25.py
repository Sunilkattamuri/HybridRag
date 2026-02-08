
import pickle
import config
from nltk.tokenize import word_tokenize
import utils

def test_bm25():
    try:
        print(f"Loading index from {config.BM25_INDEX_PATH}...")
        with open(config.BM25_INDEX_PATH, 'rb') as f:
            bm25 = pickle.load(f)
        
        corpus_data = utils.fetch_metadata()
        print(f"Corpus size (metadata): {len(corpus_data)}")
        print(f"Index size (documents): {bm25.corpus_size}")
        
        query = "What is data privacy?"
        tokenized_query = utils.preprocess_text(query)
        print(f"Query tokens: {tokenized_query}")
        
        # Check vocabulary
        for token in tokenized_query:
            if token in bm25.idf:
                print(f"Token '{token}' found in vocab (IDF: {bm25.idf[token]:.4f})")
            else:
                print(f"Token '{token}' NOT found in vocab")

        scores = bm25.get_scores(tokenized_query)
        top_n = 5
        scored_corpus = []
        for idx, score in enumerate(scores):
            scored_corpus.append((idx, score))
            
        sparse_hits = sorted(scored_corpus, key=lambda x: x[1], reverse=True)[:top_n]
        
        print("\nTop Results:")
        for idx, score in sparse_hits:
            content_preview = corpus_data[idx]['content'][:100].replace('\n', ' ')
            print(f"Score: {score:.4f} | ID: {corpus_data[idx]['chunk_id']} | Content: {content_preview}...")
            
    except Exception as e:
        print(f"Error: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    test_bm25()
