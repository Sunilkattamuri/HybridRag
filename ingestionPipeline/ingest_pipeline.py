import sys
import os
import json

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import utils
import config as CONFIG

try:
    from vectorize_chunks import vectorize_data
    from fetch_text_chunking import chunk_text, fetch_text_title, prepareMetaData
    from fetch_dynamicUrls import get_dynamic_urls
    from build_bm25model import build_bm25_index
except ImportError:
    # When imported from root (app.py), these need to be package relative or absolute
    from ingestionPipeline.vectorize_chunks import vectorize_data
    from ingestionPipeline.fetch_text_chunking import chunk_text, fetch_text_title, prepareMetaData
    from ingestionPipeline.fetch_dynamicUrls import get_dynamic_urls
    from ingestionPipeline.build_bm25model import build_bm25_index


def ingest_pipeline():

    # fetching dynamic and fixed urls combing all 
    
    get_dynamic_urls(CONFIG.CATEGORIES, target=CONFIG.TARGET_DYNAMIC_URLS, min_words=CONFIG.MIN_WORDS)

    dynamic_urls = utils.fetch_dynamic_urls_from_file()
    fixed_urls = utils.fetch_fixed_urls()
    all_urls = {**fixed_urls, **dynamic_urls}

    print(f"Total URLs to process: {len(all_urls)}")

    metadata_list = []

    # loop through all urls fetch text chunk and prepare metadata
    for idx, (title, url) in enumerate(all_urls.items()):
        print(f"{idx+1}. {title} - {url}")
        text = fetch_text_title(title)
        chunks = chunk_text(text)
        metadata = prepareMetaData(title, url, chunks)
        metadata_list.extend(metadata)
        
    # saving the metadata to a json file
    # Ensure correct path for metadata file
    metadata_path = os.path.join("files", "metadata.json")
    # If running from inside ingestionPipeline, adjust path or assume run from root
    if not os.path.exists("files") and os.path.exists("../files"):
         metadata_path = os.path.join("../files", "metadata.json")

    with open(metadata_path, 'w') as f:
        json.dump(metadata_list, f, indent=4)
    print(f"Metadata for {len(metadata_list)} chunks saved to '{metadata_path}'")

    # finally vectorizing the chunks and storing into chromadb
    vectorize_data()

    # building BM25 index
    
    build_bm25_index()


if __name__ == "__main__":
    ingest_pipeline()