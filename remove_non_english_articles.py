import json
from langdetect import detect
import os
from tqdm import tqdm

DATASET_WITH_ARTICLES = "data/dataset_with_articles.jsonl"
ENG_DATASET = "data/english_articles.jsonl"

def is_english(text):
    try:
        detected_lang = detect(text)
        return detected_lang == 'en'
    except Exception as e:
        print(f"Error detecting language: {e}")
        return False

if __name__ == "__main__":
    # This can be run standalone to remove non-English articles.
    # However, it's best to run this as part of extract_main_article_text.py
    if os.path.exists(ENG_DATASET):
        os.remove(ENG_DATASET)
    with open(DATASET_WITH_ARTICLES, 'r') as f_in:
        for line in tqdm(f_in.readlines()):
            data = json.loads(line)
            article = data.get('article', '')
            if is_english(article):
                with open(ENG_DATASET, 'a') as f_out:
                    f_out.write(line)
            else:
                print(f"Skipping non-English article: {data.get('id', 'N/A')}")