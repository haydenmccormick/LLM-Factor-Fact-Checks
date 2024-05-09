from sklearn.model_selection import train_test_split
import json
import openai
import os
import dotenv
from tqdm import tqdm

dotenv.load_dotenv()


DATASET = "../data/matched_articles.jsonl"

system_message = """
This is a fact-checking article. Please extract the claim, claimant, and verdict in the following format:

CLAIM: <claim>
CLAIMANT: <claimant>
VERDICT: <verdict>

For each factor (claim/claimant/verdict), only include spans of text taken directly from the article. Only include one of each claim, claimant, and verdict.
"""


def format_data():
    """
    Generate train/dev/test files
    """
    with open(DATASET, "r") as f:
        lines = f.readlines()
        lines = [json.loads(line) for line in lines]
        for line in lines:
            line["gold"] = f"""CLAIM: {line["claimReviewed"]}\nCLAIMANT: {line["itemReviewed"]["author"]["name"]}\nVERDICT: {line["reviewRating"]["alternateName"]}"""
            line["messages"] = [
                {"role": "system", "content": system_message},
                {"role": "user", "content": line["article"]},
                {"role": "assistant", "content": line["gold"]}
            ]

        train, val_test = train_test_split(lines, test_size=0.2)
        test, val = train_test_split(val_test, test_size=0.5)

    with open("../data/train.jsonl", "w") as f:
        for line in train:
            f.write(json.dumps(line) + "\n")

    with open("../data/val.jsonl", "w") as f:
        for line in val:
            f.write(json.dumps(line) + "\n")

    with open("../data/test.jsonl", "w") as f:
        for line in test:
            f.write(json.dumps(line) + "\n")


def fine_tune(client, model):
    """
    Fine-tune model on train.jsonl using Anyscale API
    (note: if this isn't working, make sure your API
    key is saved to your environment)
    """

    training_file_id = client.files.create(
        file=open('../data/train.jsonl','rb'),
        purpose="fine-tune",
    ).id

    valid_file_id = client.files.create(
        file=open('../data/val.jsonl','rb'),
        purpose="fine-tune",
    ).id

    finetuning_job_id = client.fine_tuning.jobs.create(
        training_file=training_file_id,
        validation_file=valid_file_id,
        model=model,
    ).id

def get_predictions(model, model_name):
    """
    Populate a jsonl file with predictions from your fine-tuned model.
    """
    with open("../data/test.jsonl", "r") as f:
        lines = f.readlines()
        for line in tqdm(lines[253:]):
            data = json.loads(line)
            messages = data["messages"]
            completion = client.chat.completions.create(
                model = model,
                messages = messages[:-1],# drop the assistant message
                temperature = 0
            )

            with open(f"../data/predicted_factors_{model_name}.jsonl", "a") as f:
                f.write(json.dumps({
                    "claim": data["claimReviewed"],
                    "claimant": data["itemReviewed"]["author"]["name"],
                    "verdict": data["reviewRating"]["alternateName"],
                    "prediction": completion.choices[0].message.content
                }) + "\n")


if __name__ == "__main__":
    client = openai.OpenAI(
        base_url = "https://api.endpoints.anyscale.com/v1",
        api_key = os.environ.get("ANYSCALE_API_KEY")
    )

    model = "mistralai/Mixtral-8x7b"

    format_data()
    fine_tune(client, model)
    get_predictions(model, "Mistral_tuned")