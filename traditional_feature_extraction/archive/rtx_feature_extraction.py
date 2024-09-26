"""
This file analyzes Google's RT-1 dataset. This dataset is called
fractal20220817 through Open-X-Embodiement.

To download this dataset, follow the instructions in the download_scripts
folder at the top-level of this directory.
"""

import os
from argparse import ArgumentParser
import pandas as pd
import tensorflow_datasets as tfds

from feature_extraction.utils import (
    drop_na,
    spacy_processing,
    get_dep_parse_tree,
    get_constituency_parse_tree,
    get_seq_len,
    get_verbs,
    get_nouns,
)


def main(params):
    ds_path = params.ds_path

    b = tfds.builder_from_directory(ds_path)
    ds = b.as_dataset(split="train")  # full dataset

    all_nl_instructions = []
    for episode in ds:
        for step in episode["steps"]:
            instruct = (
                step["observation"]["natural_language_instruction"]
                .numpy()
                .decode("UTF-8")
            )
            all_nl_instructions.append(instruct)

    df = pd.DataFrame(all_nl_instructions, columns=["nl_instructions"])

    df = drop_na(df, nl_column="nl_instructions")

    df = spacy_processing(df, nl_column="nl_instructions")
    df = get_dep_parse_tree(df)
    df = get_constituency_parse_tree(
        df, nl_column="nl_instructions", parse_tree_column="constit_parse_tree"
    )
    df = get_seq_len(df)
    df = get_verbs(df)
    df = get_nouns(df)

    results_dir_path = "/".join(os.getcwd().split("/")[:-2]) + "/results/"
    ds_name = ds_path.split("/")[ds_path.split("/").index("downloaded_datasets") + 1]

    out_path = results_dir_path + ds_name
    if not os.path.exists(out_path):
        os.makedirs(out_path, exist_ok=True)

    df.to_csv(out_path + "/" + ds_name + "_results.csv")


if __name__ == "__main__":
    parser = ArgumentParser()
    """ 
    ds_path should be the absolute path to the folder, e.g.,
    "/vast/home/slwanna/HRI_data_audit/downloaded_datasets/fractal20220817_data/0.1.0" 
    """
    parser.add_argument("--ds_path", default="None", type=str, required=True)

    args = parser.parse_args()

    main(args)
