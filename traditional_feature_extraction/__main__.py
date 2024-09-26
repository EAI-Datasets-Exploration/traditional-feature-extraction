"""
Module Entrypoint

This file is used to allow the project to be invoked via:
    `python -m text_cleaning_tool_kit`
"""
import configparser
import os
import pandas as pd
import numpy as np

from feature_extraction.utils import (
    spacy_processing,
    get_dep_parse_tree,
    get_constituency_parse_tree,
    get_seq_len,
    get_verbs,
    get_nouns,
)

if __name__ == "__main__":
    config = configparser.ConfigParser()
    config.read("traditional_feature_extraction/config.ini")

    # Load Path Variables
    dataset_dir_path = config["paths"]["dataset_dir_path"]
    results_dir = config["paths"]["results_dir"]
    
    # TODO: Make this a file within the dataset-download-scripts pkg.
    ds_name_to_file_name = {
        "alfred": "alfred_both_results.csv",
        "bridge": "bridge_nl_only.csv",
        "rt1": "fractal20220817_data_nl_only.csv",
        "scout": "ARL-SCOUT_Commander_results.csv",
        "tacoplay": "taco_play_nl_only.csv",
    }

    dataset_name = config["experiment"]["dataset_name"]
    ds_path = ds_name_to_file_name[dataset_name]

    df = pd.read_csv(dataset_dir_path+ds_path)

    if config.getboolean("experiment", "run_constituency_parsing"):
        df = get_constituency_parse_tree(
            df, 
            nl_column="nl_instructions", 
            parse_tree_column="constituency_parse_tree"
        )
    if config.getboolean("experiment", "run_dependency_parsing"):
        df = get_dep_parse_tree(df)
    if config.getboolean("experiment", "extract_nouns"):
        df = get_nouns(df)
    if config.getboolean("experiment", "extract_verbs"):
        df = get_verbs(df)
    if config.getboolean("experiment", "extract_seq_len"):
        df = get_seq_len(df)

    out_path = results_dir + dataset_name
    if not os.path.exists(out_path):
        os.makedirs(out_path, exist_ok=True)

    df.to_csv(out_path + "/" + dataset_name + "_results.csv")

    # TODO: check if this code even runs... then figure out the reporting structure...lol.
