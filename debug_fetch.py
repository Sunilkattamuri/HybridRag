from ingestionPipeline.fetch_text_chunking import fetch_text_title
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

def test_fetch(title):
    print(f"Testing fetch for: '{title}'")
    text = fetch_text_title(title)
    if text:
        print(f"Success! Length: {len(text)}")
        print(f"Preview: {text[:100]}...")
    else:
        print("Failed to fetch text.")

if __name__ == "__main__":
    test_fetch("W-CDMA")
    test_fetch("UMTS (telecommunications)") # Potential redirect target
    test_fetch("WCDMA")
