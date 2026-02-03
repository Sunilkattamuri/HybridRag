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
        
        # 1. Fetch Infobox (using action=parse for HTML)
        infobox_text = ""
        try:
            parse_params = {
                "action": "parse",
                "page": title,
                "prop": "text",
                "format": "json"
            }
            response = requests.get(url, headers=HEADERS, params=parse_params)
            data = response.json()
            if 'parse' in data and 'text' in data['parse']:
                raw_html = data['parse']['text']['*']
                soup = BeautifulSoup(raw_html, 'html.parser')
                infobox = soup.find('table', {'class': 'infobox'})
                if infobox:
                    extracted_data = []
                    for tr in infobox.find_all('tr'):
                        th = tr.find('th')
                        td = tr.find('td')
                        if th and td:
                            key = th.get_text(" ", strip=True)
                            value = td.get_text(" ", strip=True)
                            value = re.sub(r'\[\d+\]', '', value) # clean refs
                            extracted_data.append(f"{key}: {value}")
                    if extracted_data:
                         infobox_text = "Infobox:\n" + "\n".join(extracted_data) + "\n\n"
        except Exception as e:
            utils.print_error(f"Error fetching infobox for {title}: {e}")

        # 2. Fetch Content (using action=query for clean text)
        params = {
            "action": "query",
            "format": "json",
            "titles": title,
            "prop": "extracts",
            "explaintext": True,
        }
        
        # api call to fetch text
        response = requests.get(url, headers=HEADERS, params= params)

        #getting raw html from json response
        data = response.json()
        if 'error' in data:
            utils.print_error(f"API Error: {data['error']}")
            return ""
            
        page = next(iter(data['query']['pages'].values()))
        text = page.get('extract', '')

        if not text:
            utils.print_error(f"No text found for title: {title}")
            return ""
        
        # Combine infobox and main text
        return infobox_text + text


        
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


def fetch_text_camelia():
    text = fetch_text_title("Camellia_(cipher)")
    print(text)

if __name__ == "__main__":
    
    fetch_text_camelia()

#     dynamic_urls = utils.fetch_dynamic_urls_from_file()
    
#     print(type(dynamic_urls))
#    # dynamic_urls = list(dynamic_urls.values())
#     print(len(dynamic_urls))

#     for  title in (dynamic_urls):
#         idx = 0
#         print(f"{idx+1}. {title}")
#         text = fetch_text_title(title)
#         print(text)
#         if text is None or text.strip() == "":
#             continue
#         chunks = chunk_text(text)
#         print(chunks)
#         metadata = prepareMetaData(title, dynamic_urls[title], chunks)
#         print(metadata)
#         if idx >= 9:  # Just fetch first 10 for demo
#             break



   