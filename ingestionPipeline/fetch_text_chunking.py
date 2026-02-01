import sys
from bs4 import BeautifulSoup
import requests
import os
import re
import uuid

from transformers import AutoTokenizer
from langchain_text_splitters import RecursiveCharacterTextSplitter

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import utils
import config

HEADERS = {
    'User-Agent': config.USER_AGENT
}

tokenizer = AutoTokenizer.from_pretrained(config.TOKENIZER_MODEL)



def get_token_length(text):
    return len(tokenizer.encode(text)) # encoding the text to get token length based on the length we will do chunking


text_splitter = RecursiveCharacterTextSplitter(
    chunk_size= config.CHUNK_SIZE,
    chunk_overlap= config.CHUNK_OVERLAP,
    length_function=get_token_length,
    add_start_index=True,
    separators=["\n\n", "\n", " ", ""]
)


def fetch_text_title(title):
    try:
        url = config.WIKIPEDIA_API_URL
        
        #required params to fetch text as json
        params = {
         "action": "parse",
         "page": title,
         "prop": "text",
         "format": "json"
        }
        
        # api call to fetch text
        response = requests.get(url, headers=HEADERS, params= params)

        #getting raw html from json response
        raw_html = response.json()['parse']['text']['*']

        # using beautiful soup to parse html and get text, beautiful soup handles html tags and entities
        soup = BeautifulSoup(raw_html, 'html.parser')
        text = soup.get_text()

        if not text:
            utils.print_error(f"No text found for title: {title}")
            return ""
        
        # cleaning text by removing extra new lines and spaces
        # Clean up references and excessive whitespace
        text = re.sub(r'\[\d+\]', '', text)
        # Remove multiple spaces and newlines
        text = re.sub(r'\s+', ' ', text).strip()

        return text


        
    except Exception as e:
        print(f"Error fetching URL {url}: {e}")
        return None





    words = text.split()
    chunks = []
    start = 0
    while start < len(words):
        end = min(start + chunk_size, len(words))
        chunk = ' '.join(words[start:end])
        chunks.append(chunk)
        start += chunk_size - overlap
    return chunks

def chunk_text(text):
    if text is None or text.strip() == "":
        return []
    return text_splitter.split_text(text) 
# Using Langchain's RecursiveCharacterTextSplitter for chunking it will split the text based on token length configured in config.py

def prepareMetaData(title, url, chunks: list):
    metadata_list = []
    # Prepare metadata for each chunk this will be saved into a json file as metadata
    for i, chunk in enumerate(chunks):
        metadata = {
            "chunk_id":f"chunk_{uuid.uuid4().hex[:8]}",
            "title": title,
            "url": url,
            "chunk_index": i,
            "content": chunk,
            "metadata":{
                "token_length": get_token_length(chunk),
                "chunk_index": i
            }
        }
        metadata_list.append(metadata)
    return metadata_list


if __name__ == "__main__":
    
    dynamic_urls = utils.fetch_dynamic_urls_from_file()
    
    print(type(dynamic_urls))
   # dynamic_urls = list(dynamic_urls.values())
    print(len(dynamic_urls))

    for  title in (dynamic_urls):
        idx = 0
        print(f"{idx+1}. {title}")
        text = fetch_text_title(title)
        print(text)
        if text is None or text.strip() == "":
            continue
        chunks = chunk_text(text)
        print(chunks)
        metadata = prepareMetaData(title, dynamic_urls[title], chunks)
        print(metadata)
        if idx >= 9:  # Just fetch first 10 for demo
            break



   