import json
import os

class Colors:
    RED = '\033[91m'
    GREEN = '\033[92m'
    YELLOW = '\033[93m'
    BLUE = '\033[94m'
    RESET = '\033[0m'


def fetch_fixed_urls():
  
    with open("files/fixed_urls.json", 'r') as f:
        fixed_urls = json.load(f)
    return fixed_urls

def fetch_dynamic_urls_from_file():

    if not os.path.exists("files/dynamic_urls.json"):
        return []

    with open("files/dynamic_urls.json", 'r') as f:
        dynamic_urls = json.load(f)
    return (dynamic_urls)

def delete_if_exists_dynamic_urls_file():

    file_path = "files/dynamic_urls.json"
    if os.path.exists(file_path):
        os.remove(file_path)
        print(f"File '{file_path}' has been deleted.")


def fetch_metadata():
    with open("files/metadata.json", 'r') as f:
        metadata = json.load(f)
    return metadata


def fetch_qa_pairs():
    with open("files/questionanswers.json", 'r') as f:
        qa_pairs = json.load(f)
    return qa_pairs

def print_error(message):
    print(f"{Colors.RED}ERROR: {message}{Colors.RESET}")

def print_success(message):
    print(f"{Colors.GREEN}SUCCESS: {message}{Colors.RESET}")
