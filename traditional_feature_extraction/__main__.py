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
import numpy as np

from traditional_feature_extraction.feature_extraction.preprocessing import clean_and_standardize_text
from traditional_feature_extraction.feature_extraction.utils import (
    spacy_processing,
    get_dep_parse_tree,
    get_constituency_parse_tree,
    get_seq_len,
    get_verbs,
    get_nouns,
)

if __name__ == "__main__":
    mp.set_start_method('spawn', force=True)
    config = configparser.ConfigParser()
    config.read("../traditional-feature-extraction/config.ini")

    # Load Path Variables
    dataset_dir_path = config["paths"]["dataset_dir_path"]
    results_dir = config["paths"]["results_dir"]

    dataset_name = config["experiment"]["dataset_name"]

    with open(f"{dataset_dir_path}metadata.json", "r") as json_file:
        """
        The metadata file referenced here is contained in the repo:
        dataset-download-scripts package hosted in the larger GitHub group.
        """
        metadata_dict = json.load(json_file)

    ds_path = metadata_dict[dataset_name]

    df = pd.read_csv(dataset_dir_path+ds_path)

    # TODO: Figure out if I want to keep the preprocessing step in this pkg
    # or if I should include it in the dataset download pkg.
    df = clean_and_standardize_text(df, nl_column="nl_instructions")

    if config.getboolean("experiment", "run_consitutency_parsing"):
        df = get_constituency_parse_tree(
            df, 
            nl_column="nl_instructions", 
            parse_tree_column="constituency_parse_tree",
            num_models_per_gpu=100,
        )
        print("Completed Constituency Parsing!")

    # df = spacy_processing(df, nl_column="nl_instructions")

    # TODO: Figure out why dep parse is broken
    # if config.getboolean("experiment", "run_dependency_parsing"):
    #     df = get_dep_parse_tree(df)
    #     print("Completed Dependency Parsing!")
    # if config.getboolean("experiment", "extract_nouns"):
    #     df = get_nouns(df)
    #     print("Completed Noun Extraction!")
    # if config.getboolean("experiment", "extract_verbs"):
    #     df = get_verbs(df)
    #     print("Completed Verb Extraction!")
    # if config.getboolean("experiment", "extract_seq_len"):
    #     df = get_seq_len(df)
    #     print("Completed Sequence Length Extraction!")
    
    # Save results
    out_path = results_dir + "/" + dataset_name
    if not os.path.exists(out_path):
        os.makedirs(out_path, exist_ok=True)

    df.to_csv(out_path + "/" + dataset_name + "_results.csv", index=False)