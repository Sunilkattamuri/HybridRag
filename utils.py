import json
import os

class Colors:
    RED = '\033[91m'
    GREEN = '\033[92m'
    YELLOW = '\033[93m'
    BLUE = '\033[94m'
    RESET = '\033[0m'


def fetch_fixed_urls():
    base_dir = os.path.dirname(os.path.abspath(__file__))
    file_path = os.path.join(base_dir, "files", "fixed_urls.json")
    with open(file_path, 'r') as f:
        fixed_urls = json.load(f)
    return fixed_urls

def fetch_dynamic_urls_from_file():
    base_dir = os.path.dirname(os.path.abspath(__file__))
    file_path = os.path.join(base_dir, "files", "dynamic_urls.json")
    
    if not os.path.exists(file_path):
        return []

    with open(file_path, 'r') as f:
        dynamic_urls = json.load(f)
    return (dynamic_urls)

def delete_if_exists_dynamic_urls_file():
    base_dir = os.path.dirname(os.path.abspath(__file__))
    file_path = os.path.join(base_dir, "files", "dynamic_urls.json")
    if os.path.exists(file_path):
        os.remove(file_path)
        print(f"File '{file_path}' has been deleted.")


def fetch_metadata():
    base_dir = os.path.dirname(os.path.abspath(__file__))
    file_path = os.path.join(base_dir, "files", "metadata.json")
    with open(file_path, 'r') as f:
        metadata = json.load(f)
    return metadata


def fetch_qa_pairs():
    base_dir = os.path.dirname(os.path.abspath(__file__))
    file_path = os.path.join(base_dir, "files", "questionanswers.json")
    with open(file_path, 'r') as f:
        qa_pairs = json.load(f)
    return qa_pairs

def print_error(message):
    print(f"{Colors.RED}ERROR: {message}{Colors.RESET}")

def print_success(message):
    print(f"{Colors.GREEN}SUCCESS: {message}{Colors.RESET}")
