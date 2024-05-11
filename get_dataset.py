import json
from tqdm import tqdm
from concurrent.futures import ThreadPoolExecutor
from remove_non_english_articles import is_english
import os
import time
import math
from newspaper import Article
import re
import requests

# Base, unformatted dataset
DATASET_FILENAME = "data/dataset.txt"
# Output dataset with article text
DATASET_WITH_ARTICLES = "data/dataset_with_articles.jsonl"


def get_and_reformat_dataset():
    # Download public dataset

    if os.path.exists(DATASET_FILENAME):
        os.remove(DATASET_FILENAME)
    res = requests.get("https://datacommons.org/data/factcheck/fact_checks_20190605.txt.gz")
    dataset = res.text

    for line in dataset.split("\n"):
        reformatted_line = re.sub(r"\<(.*?)\>", "", line)
        with open(DATASET_FILENAME, "a") as reformatted_f:
            reformatted_f.write(reformatted_line + "\n")


def scrape_article(line, tries=0):
    """
    Try 5 times to scrape main article text from URL
    """
    try:
        dict_line = json.loads(line)
        article_url = dict_line["url"]
        # The Weekly Standard, a common source in this dataset, has since been shut down
        # and absorbed by the Washington Examiner.
        if "weeklystandard" in article_url:
            return None

        article = Article(article_url)
        article.download()
        article.parse()
        if not is_english(article.text):
            print(f"Skipping non-English article: {article_url}")
            return None
        dict_line["article"] = article.text
        return dict_line

    except Exception as e:
        print(f"Error processing {article_url}: {e}")
        if tries > 5:
            print(f"Too many retries for {article_url}, skipping")
            return None
        # Cool down in case of too many requests
        time.sleep(30)
        return scrape_article(line, tries+1)
    

def remove_html_tags(text):
    return re.sub(r"\<(.*?)\>", "", text)


if __name__ == "__main__":
    get_and_reformat_dataset()

    with open(DATASET_FILENAME, "r") as f:
        lines = f.readlines()

    BATCH_SIZE = len(lines) // 2
    num_batches = math.ceil(len(lines) / BATCH_SIZE)

    # Multithreading is necessary here: sequential HTTP calls + I/O operations 
    # can take up to 10 hours on this dataset.
    with ThreadPoolExecutor(max_workers=10) as executor:
        print("Adding scraping jobs to pool")
        if os.path.exists(DATASET_WITH_ARTICLES):
            print("Existing dataset detected. Deleting...")
            os.remove(DATASET_WITH_ARTICLES)

        with open(DATASET_WITH_ARTICLES, "w") as f:
            for batch_num in range(num_batches):
                start = batch_num * BATCH_SIZE
                end = min(start + BATCH_SIZE, len(lines))
                batch_lines = lines[start:end]

                print(f"Processing batch {batch_num + 1} of {num_batches}")
                futures = [executor.submit(scrape_article, line) for line in tqdm(batch_lines)]

                for future in tqdm(futures):
                    dict_line = future.result()
                    if dict_line:
                        f.write(json.dumps(dict_line) + "\n")
