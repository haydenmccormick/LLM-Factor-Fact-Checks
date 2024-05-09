# LLM-based Fact Check Factoring

This repository implements functions for extracting *factors* from *fact-checks*, based on the appropriately-named paper [Factoring Fact-Checks:
Structured Information Extraction from Fact-Checking Articles](https://dl.acm.org/doi/pdf/10.1145/3366423.3380231). If you're interested in evaluation results, see the `docs` directory for a mini-writeup.

## Getting started

First, if you haven't, run `pip install -r requirements.txt`.

This repository has two main functions:

1. Download and reformat the [ClaimReview](https://www.claimreviewproject.com/) dataset for NLP processing
2. Fine tune and apply LLM models on ClaimReview data

## Getting the data

To download the ClaimReview data, first run:

```
python get_dataset.py
```

This will download the [Datacommons ClaimReview dataset](https://datacommons.org/factcheck/download#research-data) ClaimReview dataset, filter out English articles, and extract the article text for each line. The output of this will be saved to `data/dataset_with_articles.jsonl`.

Then, to perform the fuzzy matching algorithm, run:

```
python fuzzy_match_factors.py
```

This will produce `data/matched_articles.jsonl`, which is ready for processing!

## NLP

First, change to the `llm` directory. Here, there are a few methods that you can use for processing the data:

- `python factor.py`: Run a cloud LLM (GPT3.5) on a subset of the dataset, and produce a new file with predictions. This produces `data/predicted_factors.jsonl`. **NOTE**: you must have an OpenAI API key saved as the environment variable `OPENAI_API_KEY` for this to work! (see .env)

- `python fine_tune.py`: Fine-tune Mistral-8B on the dataset. **NOTE**: you must have an AnyScale API key saved as the environment variable `ANYSCALE_API_KEY` for this to work!

- `python tune_dspy.py`: Use [DSPy](https://dspy-docs.vercel.app/) to "fine-tune" a CoT few-shot prompt on the datase, and run evalutaion. Like `factor.py`, you must have an OpenAI API key saved.

- `python eval.py <your_file.jsonl>`: This will run the evalutation script on your prediction file and return ROUGE-1 scores.