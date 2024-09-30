"""
The functions in this script assume you have already run feature extraction tools
on the robotics dataset.

These functions leverage existing NLG metrics, e.g., ROUGE-L and BLEU-4, to measure
how similar natural language instructions in EAI VLN datasets are to one another.

This approach was first proposed by Zhang et al. [https://arxiv.org/pdf/2005.03086]
"""
import evaluate
import torch
import multiprocessing as mp
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

    # Calculate average BLEU scores
    avg_bleu = np.mean(bleu_scores)

    return avg_bleu


def get_rouge(query_sentence: str, reference_sentences: list, rouge_fn) -> tuple:
    """Compute ROUGE scores for a query and references."""
    while True:
        try:
            rouge_l_score = rouge_fn.compute(
                predictions=[query_sentence], references=reference_sentences
            )["rougeL"]
            rouge_1_score = rouge_fn.compute(
                predictions=[query_sentence], references=reference_sentences
            )["rouge1"]
            break
        except ZeroDivisionError as e:
            print(e)

    return rouge_l_score, rouge_1_score


def sample_and_compute_rouge(references: list, rouge_fn, n_samples: int) -> list:
    """Helper function to sample and compute ROUGE scores."""
    scores = []
    for _ in range(n_samples):
        query = random.choice(references)

        reference_sample = [random.choice(references)]
        rouge_l, rouge_1 = get_rouge(query, reference_sample, rouge_fn)
        scores.append((rouge_l, rouge_1))
    return scores


def summary_rouge(
    fp: str, nl_column="nl_instructions", n_samples=1000, num_workers=100
):
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
    rouge_l_scores, rouge_1_scores = zip(*all_scores)

    # Calculate average ROUGE scores
    avg_rouge_l = np.mean(rouge_l_scores)
    avg_rouge_1 = np.mean(rouge_1_scores)

    return avg_rouge_l, avg_rouge_1


def get_levenshtein(
    query_sentence: str,
    reference_sentences: list,
) -> tuple:
    """Compute Lev Distances for a query and references."""
    while True:
        try:
            lev_dist = levenshtein_distance(query_sentence, reference_sentences)
            break
        except ZeroDivisionError as e:
            print(e)

    return lev_dist


def sample_and_compute_levscore(references: list, n_samples: int) -> list:
    """Helper function to sample and compute Levscores."""
    scores = []
    for _ in range(n_samples):
        query = random.choice(references)
        reference_sample = random.choice(references)
        lev_dist = get_levenshtein(query, reference_sample)
        scores.append(lev_dist)
    return scores


def summary_levenshtein(
    fp: str, nl_column="nl_instructions", n_samples=1000, num_workers=100
):
    # Load the CSV file and extract the references
    df = pd.read_csv(fp)
    references = list(df[nl_column])
    random.shuffle(references)

    # Split the workload into batches for parallel processing
    batch_size = n_samples // num_workers

    # Parallel processing of ROUGE score computation
    with ProcessPoolExecutor(max_workers=num_workers) as executor:
        futures = [
            executor.submit(sample_and_compute_levscore, references, batch_size)
            for _ in range(num_workers)
        ]

    # Gather the results from the futures
    lev_scores = []
    for future in futures:
        lev_scores.extend(future.result())

    # Calculate average BERTscores
    avg_levscore = np.mean(lev_scores)

    return avg_levscore


# Function to compute BERTscore
def get_bertscore(
    query_sentence: str,
    reference_sentences: list,
    bertscore_fn,
    device: str,
    model_type: str,
) -> dict:
    """Compute BERTscores for a query and references."""
    while True:
        try:
            # Compute BERTscore using the specified function
            bert_score = bertscore_fn.compute(
                predictions=[query_sentence],
                references=reference_sentences,
                model_type=model_type,
                device=device,
            )
            break
        except ZeroDivisionError as e:
            print(f"Error computing BERTscore: {e}")

    return bert_score


# Helper function to sample and compute BERTscore
def sample_and_compute_bertscore(
    references: list, n_samples: int, device: str, model_type: str
) -> list:
    """Sample random queries and compute their BERTscores using GPU or CPU."""
    # Load the BERTscore evaluation function (no device here)
    bertscore_fn = evaluate.load("bertscore")

    scores = []
    for _ in range(n_samples):
        # Sample a query and a reference sentence
        query = random.choice(references)
        reference_sample = [random.choice(references)]

        # Compute BERTscore for the sampled query and reference
        while True:
            try:
                # Compute BERTscore using the specified function
                bert_score = bertscore_fn.compute(
                    predictions=[query],
                    references=reference_sample,
                    model_type=model_type,  # Specify the model type
                    device=device,  # Specify the device (CPU or GPU)
                )
                break
            except ZeroDivisionError as e:
                print(f"Error computing BERTscore: {e}")
        scores.append(bert_score)

    return scores


# Multiprocessing function wrapper
def worker_process(
    references: list,
    n_samples: int,
    device: str,
    model_type: str,
    result_queue: mp.Queue,
):
    """Worker process function to compute BERTscores and store results in a queue."""
    result = sample_and_compute_bertscore(references, n_samples, device, model_type)
    result_queue.put(result)


def summary_bertscore(
    fp: str,
    nl_column="nl_instructions",
    n_samples=1000,
    num_workers=4,
    model_type="microsoft/deberta-xlarge-mnli",
) -> float:
    # Load the CSV file and extract the references
    df = pd.read_csv(fp)
    references = list(df[nl_column])
    random.shuffle(references)

    # Split the workload into batches for parallel processing
    batch_size = n_samples // num_workers

    # Create a queue to collect results from workers
    result_queue = mp.Queue()

    # Determine device: Use GPU if available, otherwise CPU
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")

    # Create a list to hold worker processes
    processes = []

    # Start worker processes
    for _ in range(num_workers):
        process = mp.Process(
            target=worker_process,
            args=(references, batch_size, device, model_type, result_queue),
        )
        processes.append(process)
        process.start()

    # Collect results from the queue
    bert_scores = []
    for _ in range(num_workers):
        bert_scores.extend(result_queue.get())

    # Ensure all worker processes have finished
    for process in processes:
        process.join()

    # Calculate the average BERTscore (F1 score)
    avg_bertscore = np.mean([score["f1"][0] for score in bert_scores])

    return avg_bertscore
