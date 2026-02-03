import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import utils
import config
from rank_bm25 import BM25Okapi
import pickle
from nltk.tokenize import word_tokenize


def build_bm25_index():

    # fetch documents from metadata json file
    corpus_data = utils.fetch_metadata()
    # tokenize documents
    tokenized_corpus = [word_tokenize(f"{item['title']} {item['content']}".lower()) for item in corpus_data]
    # build BM25 index
    bm25 = BM25Okapi(tokenized_corpus)

     # Create directory if it doesn't exist
    os.makedirs(os.path.dirname(config.BM25_INDEX_PATH), exist_ok=True)

    # Save the BM25 index to disk
    with open(config.BM25_INDEX_PATH, 'wb') as f:
        pickle.dump(bm25, f)

    print(f"BM25 index built and saved to {config.BM25_INDEX_PATH}")

if __name__ == "__main__":
    build_bm25_index()


