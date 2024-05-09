from dsp import LM
import dspy
import anthropic
import os
import pandas as pd
import json
from dspy.teleprompt import BootstrapFewShot
from dspy.evaluate import Evaluate
from rouge_score import rouge_scorer
from sklearn.model_selection import train_test_split
from dotenv import load_dotenv
import time
from tqdm import tqdm

load_dotenv()

client = anthropic.Anthropic()

system_message = """
This is a fact-checking article. Please extract the claim, claimant, and verdict in the following format:

CLAIM: <claim>
CLAIMANT: <claimant>
VERDICT: <verdict>

For each factor (claim/claimant/verdict), only include spans of text taken directly from the article. Only include one of each claim, claimant, and verdict.
EVERY WORD SHOULD COME EXACTLY FROM THE TEXT! Do not paraphrase. Do not edit the text in any way.
"""
DATASET_PATH = "../data/matched_articles.jsonl"    

class CoT(dspy.Module):
    def __init__(self):
        super().__init__()
        self.prog = dspy.ChainOfThought("article -> gold")
    
    def forward(self, article):
        return self.prog(article=article)

    
# Custom evaluation metric of ROUGE-1 F1 between predictions and gold
def eval_factors(gold, prediction, trace=None):
    pred = "\n".join([comp.gold for comp in prediction.completions])
    scorer = rouge_scorer.RougeScorer(['rouge1'])
    f1_score = scorer.score(gold.gold, pred)
    f1_score = f1_score["rouge1"].fmeasure
    return f1_score


def get_predictions(cot):
    print("Getting predictions...")
    with open("../data/test.jsonl", "r") as f:
        lines = f.readlines()
        for line in tqdm(lines[283:]):
            data = json.loads(line)
            prediction = cot(article = data["article"])
            pred = "\n".join([comp.gold for comp in prediction.completions])
            
            with open(f"../data/dspy_tuned.jsonl", "a") as f:
                f.write(json.dumps({
                    "claim": data["claimReviewed"],
                    "claimant": data["itemReviewed"]["author"]["name"],
                    "verdict": data["reviewRating"]["alternateName"],
                    "prediction": pred
                }) + "\n")


if __name__ == "__main__":
    # Set up the LM
    turbo = dspy.OpenAI(model='gpt-3.5-turbo', max_tokens=250)
    dspy.settings.configure(lm=turbo)

    with open(DATASET_PATH, "r") as f:
        lines = f.readlines()

    line_dicts = [json.loads(line) for line in lines]
    df = pd.DataFrame(line_dicts)

    # DSPy needs a singular gold string to tune on
    df["gold"] = df.apply(lambda x: (f"""CLAIM: {x["claimReviewed"]}\nCLAIMANT: {x["itemReviewed"]["author"]["name"]}\nVERDICT: {x["reviewRating"]["alternateName"]}"""), axis=1)
        
    # Prepare data for DSPy
    formatted_data = [
        dspy.Example(article=f"TASK: {system_message} ARTICLE: {row['article']}",
                     gold=row["gold"]
                     ).with_inputs("article") for _, row in df.iterrows()
    ]

    train, dev = train_test_split(formatted_data, test_size=0.5)
    # Only tune on first 1,000 instances (any more is unnecessary and costly)
    train, dev = train[:1000], dev[:1000]

    # Set up optimizer
    config = dict(max_bootstrapped_demos=3, max_labeled_demos=5)

    teleprompter = BootstrapFewShot(metric=eval_factors, **config)
    optimized_cot = teleprompter.compile(CoT(), trainset=train)

    evaluate = Evaluate(devset=dev, metric=eval_factors, num_threads=4, display_progress=True, display_table=0)

    # Evaluate our `optimized_cot` program.
    get_predictions(optimized_cot)