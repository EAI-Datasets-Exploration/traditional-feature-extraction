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

    df.to_csv(out_path + "/" + dataset_name + "_results.csv", index=False)
