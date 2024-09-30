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
    # summary_levenshtein,
    # summary_bertscore,
)


def nlg_metrics_neural(fp, n_samples=50, n_trials=3) -> str:
    outputs = []
    outputs.append(f"Number of neural NLG metrics trials: {n_trials}.")

    bertscore_pairwise_list = []
    for _ in range(n_trials):
        bertscore_pairwise_list.append(
            summary_bertscore(
                fp=fp,
                n_samples=n_samples,
            )
        )
    outputs.append(
        f"Average bertscore pairwise scores: {np.average(bertscore_pairwise_list)}. "
        f"Stddev: {np.std(bertscore_pairwise_list)}."
    )
    return "\n".join(outputs)


def nlg_metrics_trad(fp, n_samples=50, n_trials=3) -> str:
    """
    n_samples references how many examples from the dataset you would
    want to do comparisons against.
    """
    outputs = []
    outputs.append(f"Number of trad NLG metrics trials: {n_trials}.")

    bleu_corpus_list = []
    for _ in range(n_trials):
        bleu_corpus_list.append(
            summary_bleu(
                use_corpus=True,
                fp=fp,
                n_samples=n_samples,
            )
        )
    outputs.append(
        f"Average bleu corpus scores: {np.average(bleu_corpus_list)}. "
        f"Stddev: {np.std(bleu_corpus_list)}."
    )

    bleu_pairwise_list = []
    for _ in range(n_trials):
        bleu_pairwise_list.append(
            summary_bleu(
                use_corpus=False,
                fp=fp,
                n_samples=n_samples,
            )
        )
    outputs.append(
        f"Average bleu pairwise scores: {np.average(bleu_pairwise_list)}. "
        f"Stddev: {np.std(bleu_pairwise_list)}."
    )

    rouge_pairwise_list = []
    for _ in range(n_trials):
        rouge_pairwise_list.append(
            summary_rouge(
                fp=fp,
                n_samples=n_samples,
            )
        )
    outputs.append(
        f"Average rouge pairwise scores: {np.average(rouge_pairwise_list)}. "
        f"Stddev: {np.std(rouge_pairwise_list)}."
    )

    leven_pairwise_list = []
    for _ in range(n_trials):
        leven_pairwise_list.append(
            summary_levenshtein(
                fp=fp,
                n_samples=n_samples,
            )
        )
    outputs.append(
        f"Average levenshtein pairwise scores: {np.average(leven_pairwise_list)}. "
        f"Stddev: {np.std(leven_pairwise_list)}."
    )

    return "\n".join(outputs)


def generate_report(fp, use_trad_sim=True, use_neural_sim=True):
    report = []

    report.append(get_total_instructions(fp=fp))
    report.append(get_unique_ds(fp=fp))

    if use_trad_sim:
        report.append(nlg_metrics_trad(fp=fp))

    if use_neural_sim:
        # For Agnes to play with
        # report = report + output
        report.append(nlg_metrics_neural(fp=fp))

    result_dir = os.path.split(fp)[0]
    ds_name = get_ds_name(fp)
    with open(
        result_dir + "/" + ds_name + "_text_similarity_report.txt",
        "w+",
        encoding="utf-8",
    ) as f:
        f.write("\n".join(report))


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
