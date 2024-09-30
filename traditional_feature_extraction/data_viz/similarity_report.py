"""
Generates text files that summarize the text 
similarity metrics.
"""
import os
from traditional_feature_extraction.data_viz.utils import get_ds_name, get_total_instructions, get_unique_ds
import numpy as np
from traditional_feature_extraction.similarity_measures.nlg_metrics import (
    summary_bleu,
    summary_rouge,
    summary_levenshtein,
    summary_bertscore,
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
        f"Average levenshtein pairwise distances: {avg_lev:.4f}. " f"Stddev: {stddev_lev:.4f}."
    )
    return output


def calc_rouge(fp, n_trials=3, n_samples=1000):
    rouge_l_pairwise_list = []
    rouge_1_pairwise_list = []
    for _ in range(n_trials):
        rougel, rouge1 =  summary_rouge(
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
        f"Average bertscore pairwise scores: {avg_bertscore:.4f}. " f"Stddev: {stddev_bertscore:.4f}."
    )

    return output