from bs4 import BeautifulSoup
import wikipediaapi
import json
import os
import sys
import random

# Add parent directory to path for imports
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import utils

HEADERS = {
    'User-Agent': 'BIT-User-Agent/1.0 (acedamic research;)'
}

def fetch_from_category(cat, wiki, fixedURls, skipped_titles, min_words, target, current_urls, depth=0, max_depth=5):
    if depth > max_depth:
        return

    try:
        members_list = list(cat.categorymembers.values())
    except Exception as e:
        print(f"Error fetching members for {cat.title}: {e}")
        return

    random.shuffle(members_list)

    # Separate pages and subcats
    pages = [m for m in members_list if m.ns == wikipediaapi.Namespace.MAIN]
    subcats = [m for m in members_list if m.ns == wikipediaapi.Namespace.CATEGORY]

    # Collect from pages first
    for member in pages:
        if len(current_urls) >= target:
            return
        if member.title in fixedURls or member.title in skipped_titles:
            continue  # Skip if in fixed URLs or already collected globally
        try:
            page = wiki.page(member.title)
            words = len(page.text.split())
            if words >= min_words:
                current_urls[member.title] = page.fullurl
                print(f"[{len(current_urls)}] Collected: {member.title} (Depth: {depth})")
        except Exception as e:
            print(f"Error processing page {member.title}: {e}")

    # Then recurse into subcats
    for sub_member in subcats:
        if len(current_urls) >= target:
            return
        if sub_member.title in fixedURls or sub_member.title in skipped_titles:
            continue  # Skip if in fixed or already collected (though for cats, maybe not necessary)
        try:
            sub_cat = wiki.page(sub_member.title)
            fetch_from_category(sub_cat, wiki, fixedURls, skipped_titles, min_words, target, current_urls, depth + 1, max_depth)
        except Exception as e:
            print(f"Error processing subcategory {sub_member.title}: {e}")

def get_dynamic_urls(category_list, target=200, min_words=200):
    urls = {}
    wiki = wikipediaapi.Wikipedia(user_agent='BITS-CAI-Assignment2/1.0 (2024AA05522@wilp.bits-pilani.ac.in)', 
                                  language='en', extract_format=wikipediaapi.ExtractFormat.WIKI)

    fixedURls = set(utils.fetch_fixed_urls())

    no_of_urls_by_category = target // len(category_list)
    global_count = 0

    for cat_name in category_list:
        print(f"Expanding: {cat_name}")
        per_cat_urls = {}
        skipped_titles = set(urls.keys())
        try:
            cat = wiki.page(cat_name)
            if not cat.exists():
                print(f"Category {cat_name} does not exist. Skipping.")
                continue
            fetch_from_category(cat, wiki, fixedURls, skipped_titles, min_words, no_of_urls_by_category, per_cat_urls)
        except Exception as e:
            print(f"Error with category {cat_name}: {e}")
            continue

      

        # If still less, retry with higher max_depth
        if len(per_cat_urls) < no_of_urls_by_category:
            print(f"Retrying {cat_name} with higher max_depth (10)")
            remaining = no_of_urls_by_category - len(per_cat_urls)
            fetch_from_category(cat, wiki, fixedURls, skipped_titles, min_words, remaining, per_cat_urls, max_depth=10)

        urls.update(per_cat_urls)
        global_count += len(per_cat_urls)
        print(f"Completed {cat_name}: {len(per_cat_urls)} URLs collected for this category")

        if global_count >= target:
            break

    # If total is still less than target, fetch additional from Cryptography
    if len(urls) < target:
        print("Fetching additional from Category:Cryptography")
        cat_name = "Category:Cryptography"
        try:
            cat = wiki.page(cat_name)
            if cat.exists():
                remaining = target - len(urls)
                skipped_titles = set(urls.keys())
                fetch_from_category(cat, wiki, fixedURls, skipped_titles, min_words, remaining, urls, max_depth=10)
        except Exception as e:
            print(f"Error fetching additional from {cat_name}: {e}")

    # If still less, retry additional with lower min_words
    if len(urls) < target:
        print("Retrying additional from Category:Cryptography with lower min_words (100)")
        remaining = target - len(urls)
        skipped_titles = set(urls.keys())
        fetch_from_category(cat, wiki, fixedURls, skipped_titles, 100, remaining, urls, max_depth=10)

    return urls

if __name__ == "__main__":
    categories = [
        "Category:Computer security",
        "Category:Cryptography",
        "Category:Information privacy",
        "Category:Information sensitivity",
        "Category:Cybersecurity engineering"
    ]
    collected_urls = get_dynamic_urls(categories, target=300, min_words=200)
    output_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "files", "dynamic_urls.json")
    with open(output_path, "w") as f:
        json.dump(collected_urls, f, indent=2)

    print(f"Total URLs collected: {len(collected_urls)}")