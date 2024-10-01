"""
Module Entrypoint

This file is used to allow the project to be invoked via:
    `python -m traditional_feature_extraction`
"""
import configparser
import json
import os
import pandas as pd
import multiprocessing as mp
import time

from traditional_feature_extraction.feature_extraction.preprocessing import (
    clean_and_standardize_text,
)
from traditional_feature_extraction.feature_extraction.utils import (
    spacy_processing_parallel,
    get_constituency_parse_tree,
)
from traditional_feature_extraction.data_viz.utils import (
    get_word_cloud,
)
from traditional_feature_extraction.data_viz.similarity_report import (
    calc_bertscore,
    calc_bleu,
    calc_rouge,
    calc_lev,
    calc_treekernel,
)


if __name__ == "__main__":
    # Setup multiprocessing for multi-gpu work
    mp.set_start_method("spawn", force=True)

    config = configparser.ConfigParser()
    config.read("../traditional-feature-extraction/config.ini")

    # Load Configuration Variables
    dataset_dir_path = config["paths"]["dataset_dir_path"]
    results_dir = config["paths"]["results_dir"]

    dataset_name = config["experiment"]["dataset_name"]

    with open(f"{dataset_dir_path}metadata.json", "r", encoding="utf-8") as json_file:
        # The metadata file referenced here is contained in the repo:
        # dataset-download-scripts package hosted in the larger GitHub group.
        metadata_dict = json.load(json_file)

    ds_path = metadata_dict[dataset_name]

    # Begin text feature processing
    df = pd.read_csv(dataset_dir_path + ds_path)

    df = clean_and_standardize_text(df, nl_column="nl_instructions")

    if config.getboolean("experiment", "run_consitutency_parsing"):
        # This took 6 hours with rt1 dataset
        start_time = time.time()
        df = get_constituency_parse_tree(
            df,
            nl_column="nl_instructions",
            parse_tree_column="constituency_parse_tree",
            num_models_per_gpu=100,
        )

    spacy_kwargs = {
        "extract_nouns": config.getboolean("experiment", "extract_nouns"),
        "extract_verbs": config.getboolean("experiment", "extract_verbs"),
        "extract_seq_len": config.getboolean("experiment", "extract_seq_len"),
    }

    if any(spacy_kwargs.values()):
        df = spacy_processing_parallel(
            df, nl_column="nl_instructions", num_workers=32, **spacy_kwargs
        )

    # Save results
    out_path = results_dir + "/" + dataset_name
    if not os.path.exists(out_path):
        os.makedirs(out_path, exist_ok=True)

    resulting_fp = out_path + "/" + dataset_name
    df.to_csv(resulting_fp + "_results.csv", index=False)

    ###
    ### Building Visualizations
    ###

    reporting_kwargs = {
        "run_verb_cloud": config.getboolean("reporting", "run_verb_cloud"),
        "run_noun_cloud": config.getboolean("reporting", "run_noun_cloud"),
        "run_rouge_score": config.getboolean("reporting", "run_rouge_score"),
        "run_bert_score": config.getboolean("reporting", "run_bert_score"),
        "run_bleu_score": config.getboolean("reporting", "run_bleu_score"),
        "run_levenshtein_distance": config.getboolean(
            "reporting", "run_levenshtein_distance"
        ),
        "run_treekernels": config.getboolean("reporting", "run_treekernels"),
    }

    if any(reporting_kwargs.values()):
        if reporting_kwargs.get("run_verb_cloud"):
            get_word_cloud(resulting_fp + "_results.csv", column="verbs", bigrams=False)
        if reporting_kwargs.get("run_noun_cloud"):
            get_word_cloud(resulting_fp + "_results.csv", column="nouns", bigrams=False)

        outputs = []
        if reporting_kwargs.get("run_rouge_score"):
            outputs.append(calc_rouge(resulting_fp + "_results.csv"))
        if reporting_kwargs.get("run_bleu_score"):
            outputs.append(calc_bleu(resulting_fp + "_results.csv"))
        if reporting_kwargs.get("run_bert_score"):
            outputs.append(calc_bertscore(resulting_fp + "_results.csv"))
        if reporting_kwargs.get("run_levenshtein_distance"):
            outputs.append(calc_lev(resulting_fp + "_results.csv"))
        if reporting_kwargs.get("run_treekernels"):
            outputs.append(calc_treekernel(resulting_fp + "_results.csv"))
        with open(
            resulting_fp + "_text_similarity_report.txt",
            "w+",
            encoding="utf-8",
        ) as f:
            f.write("\n".join(outputs))
