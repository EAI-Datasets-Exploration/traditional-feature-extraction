"""
Generates text files that summarize the text 
similarity metrics.
"""
from diversity import compression_ratio, homogenization_score, ngram_diversity_score
import pandas as pd
import numpy as np
from traditional_feature_extraction.similarity_measures.nlg_metrics import (
    summary_bleu,
    summary_rouge,
    summary_levenshtein,
    summary_bertscore,
    summary_treekernel,
    summary_jaccard,
)


def calc_lev(fp, n_trials=3, n_samples=1000):
    leven_pairwise_list = []
    for _ in range(n_trials):
        leven_pairwise_list.append(
            summary_levenshtein(
                fp=fp,
                n_samples=n_samples,
            )
        )
    avg_lev = np.average(leven_pairwise_list)
    stddev_lev = np.std(leven_pairwise_list)

    # Adjust the formatting to show the values with 4 decimal places
    output = (
        f"Average levenshtein pairwise distances: {avg_lev:.4f}. "
        f"Stddev: {stddev_lev:.4f}."
    )
    return output


def calc_rouge(fp, n_trials=3, n_samples=1000):
    rouge_l_pairwise_list = []
    rouge_1_pairwise_list = []
    for _ in range(n_trials):
        rougel, rouge1 = summary_rouge(
            fp=fp,
            n_samples=n_samples,
        )
        rouge_l_pairwise_list.append(rougel)
        rouge_1_pairwise_list.append(rouge1)

    avg_rougel = np.average(rouge_l_pairwise_list)
    stddev_rougel = np.std(rouge_l_pairwise_list)

    avg_rouge1 = np.average(rouge_1_pairwise_list)
    stddev_rouge1 = np.std(rouge_1_pairwise_list)
    # Adjust the formatting to show the values with 4 decimal places
    output = (
        f"Average rouge-l pairwise scores: {avg_rougel:.4f}. "
        f"Stddev: {stddev_rougel:.4f}.\n"
        f"Average rouge-1 pairwise scores: {avg_rouge1:.4f}. "
        f"Stddev: {stddev_rouge1:.4f}."
    )

    return output


def calc_bleu(fp, n_trials=3, n_samples=1000):
    bleu_pairwise_list = []
    for _ in range(n_trials):
        bleu_pairwise_list.append(
            summary_bleu(
                fp=fp,
                n_samples=n_samples,
            )
        )

    avg_bleu = np.average(bleu_pairwise_list)
    stddev_bleu = np.std(bleu_pairwise_list)

    output = (
        f"Average bleu pairwise scores: {avg_bleu:.4f}. " f"Stddev: {stddev_bleu:.4f}."
    )

    return output


def calc_bertscore(fp, n_trials=3, n_samples=1000):
    bertscore_pairwise_list = []
    for _ in range(n_trials):
        bertscore_pairwise_list.append(
            summary_bertscore(
                fp=fp,
                n_samples=n_samples,
            )
        )

    avg_bertscore = np.average(bertscore_pairwise_list)
    stddev_bertscore = np.std(bertscore_pairwise_list)

    output = (
        f"Average bertscore pairwise scores: {avg_bertscore:.4f}. "
        f"Stddev: {stddev_bertscore:.4f}."
    )

    return output


def calc_treekernel(fp, n_trials=3, n_samples=1000):
    treekernel_pairwise_list = []
    for _ in range(n_trials):
        treekernel_pairwise_list.append(
            summary_treekernel(
                fp=fp,
                n_samples=n_samples,
            )
        )
    avg_tk = np.average(treekernel_pairwise_list)
    stddev_tk = np.std(treekernel_pairwise_list)

    # Adjust the formatting to show the values with 4 decimal places
    output = (
        f"Average tree kernel pairwise distances: {avg_tk:.4f}. "
        f"Stddev: {stddev_tk:.4f}."
    )
    return output


def calc_jaccard(fp, n_trials=3, n_samples=1000):
    jaccard_pairwise_list = []
    for _ in range(n_trials):
        jaccard_pairwise_list.append(
            summary_jaccard(
                fp=fp,
                n_samples=n_samples,
            )
        )
    avg_jaccard = np.average(jaccard_pairwise_list)
    stddev_jaccard = np.std(jaccard_pairwise_list)

    # Adjust the formatting to show the values with 4 decimal places
    output = (
        f"Average Jaccard pairwise distances: {avg_jaccard:.4f}. "
        f"Stddev: {stddev_jaccard:.4f}."
    )
    return output


def calc_compression_rato(fp, nl_column="nl_instructions"):
    df = pd.read_csv(fp)
    references = list(df[nl_column])

    cr = compression_ratio(references, 'gzip')

    output = (
        f"Compression Ratio: {cr:.4f}. "
    )
    return output


def calc_homogenization(fp, nl_column="nl_instructions", homo_type="rougel"):
    df = pd.read_csv(fp)
    references = list(df[nl_column])

    cr = homogenization_score(references, homo_type)

    output = (
        f"Homogenization Score for {homo_type}: {cr:.4f}. "
    )
    return output


def calc_ngram_diversity(fp, nl_column="nl_instructions", n=4):
    df = pd.read_csv(fp)
    references = list(df[nl_column])

    cr = ngram_diversity_score(references, n)

    output = (
        f"Ngram Diversity Score for {n}-gram: {cr:.4f}. "
    )
    return output