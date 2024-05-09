import json
import os
from nltk.tokenize import word_tokenize
from nltk.tokenize.treebank import TreebankWordDetokenizer
from tqdm import tqdm
from collections import Counter
import re

ENG_DATASET = "data/dataset_with_articles.jsonl"
MATCHED_DATASET = "data/matched_articles.jsonl"

transl_table = dict( [ (ord(x), ord(y)) for x,y in zip( u"`‘’´“”–-",  u"''''\"\"--\``") ] ) 
unstable_chars = r"[\u2018\u2019\u201C\u201D\u2013\u2014\u2026'`]"

# Helper functions for fuzzy matching algorithm ------------------

def string_overlap(s, t):
    t_counts = Counter(t)
    s_counts = Counter(s)
    return sum([t_elem in s and s_counts[t_elem] >= t_counts[t_elem] for t_elem in t])/len(t)


def minimum_window_substring(s, t):
    window_size = len(t)
    l, r = 0, window_size
    if r > len(s):
        return ""
    
    while window_size <= len(s):
        matches = []
        while r <= len(s):
            if string_overlap(s[l:r], t) >= 2/3:
                matches.append(s[l:r])
                # return s[l:r]
            l, r = l+1, r+1
        if matches:
            return max(matches, key=lambda match: string_overlap(match, t))
        window_size += 1
        l, r = 0, window_size
    
    return ""

# ----------------------------------------------------------------
    

def fuzzy_match(factor, article):
    paragraphs = article.split('[PARAGRAPH_SEP]')

    # Special characters show up a lot in fact checks, but if there's a perfect match
    # without them, it should be accepted
    standardized_factor = re.sub(r'[^a-zA-Z0-9\s]', '', factor)
    standardized_paragraphs = [re.sub(r'[^a-zA-Z0-9\s]', '', pg) for pg in paragraphs]

    if standardized_factor in " ".join(standardized_paragraphs):
        return factor

    paragraphs = [word_tokenize(pg) for pg in paragraphs]
    factor, article = word_tokenize(factor), word_tokenize(article)
    overlap = [len(list(set(factor) & set(pg)))/len(set(factor)) for pg in paragraphs]
    best_score = max(overlap)
    best_match = paragraphs[overlap.index(best_score)]
    if best_score < 2/3:
        return None
    return TreebankWordDetokenizer().detokenize(minimum_window_substring(best_match, factor))


if __name__ == "__main__":
    if os.path.exists(MATCHED_DATASET):
        os.remove(MATCHED_DATASET)

    with open(ENG_DATASET, 'r') as f_in:
        file_len = len(f_in.readlines())
    with open(ENG_DATASET, 'r') as f_in:
        for line in tqdm(f_in, total=file_len):
            data = json.loads(line)
            article = data['article']

            # If the entry doesn't have all of these fields, it's not useful
            try:
                claim = data["claimReviewed"]
                claimant = data["itemReviewed"]["author"]["name"]
                verdict = data["reviewRating"]["alternateName"]
            except KeyError:
                continue
            
            claim = fuzzy_match(claim, article)
            claimant = fuzzy_match(claimant, article)
            verdict = fuzzy_match(verdict, article)

            if claim is not None and claim != "":
                data["claimReviewed"] = claim
            if claimant is not None and claimant != "":
                data["itemReviewed"]["author"]["name"] = claimant
            if verdict is not None and verdict != "":
                data["reviewRating"]["alternateName"] = verdict

            with open(MATCHED_DATASET, 'a') as f_out:
                f_out.write(json.dumps(data) + '\n')