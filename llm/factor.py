import anthropic
from openai import OpenAI
import openai

from dotenv import load_dotenv
import json
import os
from tqdm import tqdm
import time

DATASET = "../data/matched_articles.jsonl"
PREDICTED_DATASET = "../data/predicted_factors.jsonl"

load_dotenv()

system_message = """
This is a fact-checking article. Please extract the claim, claimant, and verdict in the following format:

CLAIM: <claim>
CLAIMANT: <claimant>
VERDICT: <verdict>

For each factor (claim/claimant/verdict), only include spans of text taken directly from the article. Only include one of each claim, claimant, and verdict.
"""

def factor_line(line, model, client):
    data = json.loads(line)
    article = data["article"]
    claim = data["claimReviewed"]
    claimant = data["itemReviewed"]["author"]["name"]
    verdict = data["reviewRating"]["alternateName"]

    if "gpt" in model:
        completion = client.chat.completions.create(
            model=model,
            max_tokens=100,
            temperature=0.0,
            messages=[
                {"role": "system", "content": system_message},
                {"role": "user", "content": article}
            ]
        )

        with open(PREDICTED_DATASET, "a") as f:
            f.write(json.dumps({
                "claim": claim,
                "claimant": claimant,
                "verdict": verdict,
                "prediction": completion.choices[0].message.content
            }) + "\n")


    else:
        message = client.messeges.create(
            model=model,
            max_tokens=100,
            temperature=0.0,
            messages=[
                {"role": "system", "content": system_message},
                {"role": "user", "content": article}
            ]
        )
        with open(PREDICTED_DATASET, "a") as f:
            f.write(json.dumps({
                "claim": claim,
                "claimant": claimant,
                "verdict": verdict,
                "prediction": message.content
            }) + "\n")

    return True



def factor_fact_checks(model, client):
    """
    Use the GPT-3.5-turbo model to extract the claim, claimant, and verdict
    ()
    """
    if os.path.exists(PREDICTED_DATASET):
        os.remove(PREDICTED_DATASET)

    with open(DATASET, "r") as f:
        # Read 450 instances for consistency with fine-tuned
        # models (and to avoid extra cost/rate limits)
        lines = f.readlines()[:450]
        for line in tqdm(lines):
            processed_successfully = False
            while processed_successfully == False:
                try:

                    processed_successfully = factor_line(line, model, client)

                except openai.RateLimitError as e:
                    print(f"Error: {e}")
                    time.sleep(60)

if __name__ == "__main__":
    client = OpenAI()
    factor_fact_checks("gpt-3.5-turbo", client)
    print("Predictions saved to", PREDICTED_DATASET)