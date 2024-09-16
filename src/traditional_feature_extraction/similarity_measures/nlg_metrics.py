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
import random


def summary_bleu(use_corpus, fp: str, nl_column="nl_instructions", n_samples=1000):
    bleu_fn = evaluate.load("bleu")
    bleu_scores = []

    df = pd.read_csv(fp)
    references = list(df[nl_column])

    def get_bleu(query_sentence: str, reference_sentences: list):
        try:
            bleu_score = bleu_fn.compute(
                predictions=[query_sentence], references=reference_sentences
            )["bleu"]
        except ZeroDivisionError as e:
            print(e)

        return bleu_score

    if use_corpus:
        for _ in range(n_samples):
            query = random.choice(references)
            bleu_scores.append(get_bleu(query, [references]))
    else:
        for _ in range(n_samples):
            query = random.choice(references)
            references = [random.choice(references)]
            bleu_scores.append(get_bleu(query, references))
    return sum(bleu_scores) / len(bleu_scores)


def summary_rouge(fp: str, nl_column="nl_instructions", n_samples=1000):
    rouge_fn = evaluate.load("rouge")

    df = pd.read_csv(fp)
    references = list(df[nl_column])

    def get_rouge(query_sentence: str, reference_sentences: list):
        try:
            rouge_score = rouge_fn.compute(
                predictions=[query_sentence], references=reference_sentences
            )["rougeL"]
        except ZeroDivisionError as e:
            print(e)

        return rouge_score

    rouge_scores = []
    for _ in range(n_samples):
        query = random.choice(references)
        references = [random.choice(references)]
        rouge_scores.append(get_rouge(query, references))
    return sum(rouge_scores) / len(rouge_scores)


def summary_bertscore(fp: str, nl_column="nl_instructions", n_samples=1000):
    bertscore_fn = evaluate.load("bertscore")

    df = pd.read_csv(fp)
    references = list(df[nl_column])

    def get_bertscore(query_sentence: str, reference_sentences: list):
        try:
            bertscore = bertscore_fn.compute(
                predictions=[query_sentence],
                references=reference_sentences,
                model_type="distilbert-base-uncased",
            )["f1"][0]
        except ZeroDivisionError as e:
            print(e)

        return bertscore

    bertscores = []
    for _ in range(n_samples):
        query = random.choice(references)
        references = [random.choice(references)]
        bertscores.append(get_bertscore(query, references))
    return sum(bertscores) / len(bertscores)


def summary_levenshtein(fp: str, nl_column="nl_instructions", n_samples=1000):
    df = pd.read_csv(fp)
    references = list(df[nl_column])

    levenshtein_dists = []
    for _ in range(n_samples):
        query = random.choice(references)
        references = random.choice(references)
        levenshtein_dists.append(levenshtein_distance(query, references))
    return sum(levenshtein_dists) / len(levenshtein_dists)
