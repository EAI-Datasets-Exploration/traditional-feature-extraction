"""
The functions in this script assume you have already run feature extraction tools
on the robotics dataset.

These functions leverage existing NLG metrics, e.g., ROUGE-L and BLEU-4, to measure
how similar natural language instructions in EAI VLN datasets are to one another.

This approach was first proposed by Zhang et al. [https://arxiv.org/pdf/2005.03086]
"""
import evaluate
from Levenshtein import distance as levenshtein_distance
import pandas as pd
import numpy as np
import random
from concurrent.futures import ProcessPoolExecutor


def get_bleu(query_sentence: str, reference_sentences: list, bleu_fn) -> tuple:
    """Compute BLEU scores for a query and references."""
    while True:
        try:
            bleu_score = bleu_fn.compute(
                predictions=[query_sentence], references=reference_sentences
            )["bleu"]
            break
        except ZeroDivisionError as e:
            print(e)
    
    return bleu_score

def sample_and_compute_bleu(references: list, bleu_fn, n_samples: int) -> list:
    """Helper function to sample and compute ROUGE scores."""
    scores = []
    for _ in range(n_samples):
        query = random.choice(references)
        
        reference_sample = [random.choice(references)]
        bleu_score = get_bleu(query, reference_sample, bleu_fn)
        scores.append(bleu_score)
    return scores

def summary_bleu(fp: str, nl_column="nl_instructions", n_samples=1000, num_workers=100):
    # Load the rouge evaluation function
    bleu_fn = evaluate.load("bleu")

    # Load the CSV file and extract the references
    df = pd.read_csv(fp)
    references = list(df[nl_column])
    random.shuffle(references)

    # Split the workload into batches for parallel processing
    batch_size = n_samples // num_workers

    # Parallel processing of ROUGE score computation
    with ProcessPoolExecutor(max_workers=num_workers) as executor:
        futures = [
            executor.submit(sample_and_compute_bleu, references, bleu_fn, batch_size)
            for _ in range(num_workers)
        ]

    # Gather the results from the futures
    bleu_scores = []
    for future in futures:
        bleu_scores.extend(future.result())
    
    # Calculate average ROUGE scores
    avg_bleu = np.mean(bleu_scores)

    return avg_bleu


def get_rouge(query_sentence: str, reference_sentences: list, rouge_fn) -> tuple:
    """Compute ROUGE scores for a query and references."""
    while True:
        try:
            rouge_L_score = rouge_fn.compute(
                predictions=[query_sentence], references=reference_sentences
            )["rougeL"]
            rouge_1_score = rouge_fn.compute(
                predictions=[query_sentence], references=reference_sentences
            )["rouge1"]
            break
        except ZeroDivisionError as e:
            print(e)
    
    return rouge_L_score, rouge_1_score

def sample_and_compute_rouge(references: list, rouge_fn, n_samples: int) -> list:
    """Helper function to sample and compute ROUGE scores."""
    scores = []
    for _ in range(n_samples):
        query = random.choice(references)
        
        reference_sample = [random.choice(references)]
        rouge_l, rouge_1 = get_rouge(query, reference_sample, rouge_fn)
        scores.append((rouge_l, rouge_1))
    return scores

def summary_rouge(fp: str, nl_column="nl_instructions", n_samples=1000, num_workers=100):
    # Load the rouge evaluation function
    rouge_fn = evaluate.load("rouge")

    # Load the CSV file and extract the references
    df = pd.read_csv(fp)
    references = list(df[nl_column])
    random.shuffle(references)

    # Split the workload into batches for parallel processing
    batch_size = n_samples // num_workers

    # Parallel processing of ROUGE score computation
    with ProcessPoolExecutor(max_workers=num_workers) as executor:
        futures = [
            executor.submit(sample_and_compute_rouge, references, rouge_fn, batch_size)
            for _ in range(num_workers)
        ]

    # Gather the results from the futures
    all_scores = []
    for future in futures:
        all_scores.extend(future.result())
    # Split the results into rouge_L_scores and rouge_1_scores
    rouge_L_scores, rouge_1_scores = zip(*all_scores)

    # Calculate average ROUGE scores
    avg_rouge_l = np.mean(rouge_L_scores)
    avg_rouge_1 = np.mean(rouge_1_scores)

    return avg_rouge_l, avg_rouge_1


# def summary_bertscore(fp: str, nl_column="nl_instructions", n_samples=1000):
#     bertscore_fn = evaluate.load("bertscore")

#     df = pd.read_csv(fp)
#     references = list(df[nl_column])

#     def get_bertscore(query_sentence: str, reference_sentences: list):
#         try:
#             bertscore = bertscore_fn.compute(
#                 predictions=[query_sentence],
#                 references=reference_sentences,
#                 model_type="distilbert-base-uncased",
#             )["f1"][0]
#         except ZeroDivisionError as e:
#             print(e)

#         return bertscore

#     bertscores = []
#     for _ in range(n_samples):
#         query = random.choice(references)
#         references = [random.choice(references)]
#         bertscores.append(get_bertscore(query, references))
#     return sum(bertscores) / len(bertscores)


# def summary_levenshtein(fp: str, nl_column="nl_instructions", n_samples=1000):
#     df = pd.read_csv(fp)
#     references = list(df[nl_column])

#     levenshtein_dists = []
#     for _ in range(n_samples):
#         query = random.choice(references)
#         references = random.choice(references)
#         levenshtein_dists.append(levenshtein_distance(query, references))
#     return sum(levenshtein_dists) / len(levenshtein_dists)
