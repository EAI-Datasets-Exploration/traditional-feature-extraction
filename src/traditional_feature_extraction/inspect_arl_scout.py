"""
This file analyzes ARL-SCOUT dataset.

To download this dataset, follow the instructions in the download_scripts
folder at the top-level of this directory.
"""

import os
import regex as re
from argparse import ArgumentParser
import pandas as pd

from feature_extraction.utils import (
    drop_na,
    spacy_processing,
    get_dep_parse_tree,
    get_constituency_parse_tree,
    get_seq_len,
    get_verbs,
    get_nouns,
)


def compare_experiment_4_data_streams(df: pd.DataFrame, output_file_path: str):
    columns_to_compare = [
        "Commander ASR",
        "Commander Normalized",
        "Commander Transcribed",
    ]
    differing_rows = df[df.apply(lambda x: len(set(x[columns_to_compare])) > 1, axis=1)]
    mode = "a" if os.path.exists(output_file_path) else "w"
    with open(output_file_path, mode, encoding="utf-8") as f:
        header = os.path.exists(output_file_path)
        differing_rows.to_csv(f, index=False, header=header, columns=columns_to_compare)


def rename_and_drop_columns_to_match(df: pd.DataFrame) -> pd.DataFrame:
    """
    Renames and drops columns in the DataFrame for handling data from Experiment 4,
    which contains ASR data not present in other experiments.
    """
    df = df.drop(columns=["Commander ASR", "Commander Normalized"], errors="ignore")
    return df.rename(columns={"Commander Transcribed": "Commander"})


def add_filename_info_to_df(df: pd.DataFrame, filename: str) -> pd.DataFrame:
    """
    Adds information extracted from the filename to the DataFrame.
    The filename contains the experiment number, commander ID, and trial information.
    """
    pattern = r"^(p\d+)\.(\d+)_([^_]+)"
    match = re.match(pattern, filename)
    experiment, commander_id, trial = match.groups()
    experiment_int = int(experiment[1:])
    commander_id_int = int(commander_id)
    df["Experiment number"] = experiment_int  # Experiment number 1, 2, 3 or 4
    df[
        "Commander ID"
    ] = commander_id_int  # Commander id, different test subjects, total 93
    df["Trial"] = trial  # train, main1 or main2
    return df


def read_xlsx_to_df(ds_path: str, log_file: str = None) -> pd.DataFrame:
    """
    Reads all xlsx files in the specified dialogue-structure folder and
    returns a DataFrame with the following columns:
    'ID#', 'Timestamp', 'Commander', 'DM->CMD', 'DM->RN', 'RN',
    'Transaction', 'Antecedent', 'Relation', 'Contextual Info',
    'Experiment number', 'Commander ID', 'Trial'.
    """
    dialogue_dir = f"{ds_path}/data/dialogue-structure"
    xlsx_files = [file for file in os.listdir(dialogue_dir) if file.endswith(".xlsx")]
    all_dfs = pd.DataFrame()

    if log_file and os.path.isfile(log_file):
        os.remove(os.path.join(log_file))

    for file in xlsx_files:
        xl = pd.ExcelFile(f"{dialogue_dir}/{file}")
        df = xl.parse()

        if file.startswith("p4"):
            if log_file:
                compare_experiment_4_data_streams(df, log_file)
            df = rename_and_drop_columns_to_match(df)

        df = add_filename_info_to_df(df, file)
        all_dfs = pd.concat([all_dfs, df], ignore_index=True)
    all_dfs = all_dfs.sample(frac=1, random_state=42).reset_index(
        drop=True
    )  # shuffle the dataframe
    return all_dfs


def remove_tags(
    df: pd.DataFrame, nl_column: str, log_out_path: str = None
) -> pd.DataFrame:
    """
    Removes specific tags (e.g., <noise>, <silence>) from a specified column.
    Logic could be improved!
    """

    def extract_text_from_tags(text):
        if pd.isna(text):
            return pd.NA
        matches = re.findall(r"<([XxabmkK]):\s?([^>]*?(?:<.*?>)?.*?)>", text)
        for tag, match in matches:
            text = text.replace(f"<{tag}: {match}>", match).replace(
                f"<{tag}:{match}>", match
            )
        return text

    df[f"{nl_column}_no_preproc"] = df[nl_column].copy()

    df[nl_column] = df[nl_column].apply(extract_text_from_tags)
    df[nl_column] = df[nl_column].astype(str).str.replace(r"<.*?>", "", regex=True)
    df[nl_column] = df[nl_column].str.replace(r"\s+", " ", regex=True)
    df[nl_column] = df[nl_column].str.replace(r"\s+([,.!?])", r"\1", regex=True)
    df[nl_column] = df[nl_column].apply(
        lambda x: pd.NA if pd.isna(x) or str(x).strip() == "" else x
    )

    if log_out_path:
        fully_removed_df = df[
            pd.isna(df[nl_column]) & pd.notna(df[f"{nl_column}_no_preproc"])
        ]
        fully_removed_df[[f"{nl_column}_no_preproc"]].to_csv(
            os.path.join(log_out_path, "fully_removed_utterances_only_tags.tsv"),
            sep="\t",
            index=False,
            header=["Original"],
        )

        changed_utterances_df = df[
            pd.notna(df[nl_column]) & (df[f"{nl_column}_no_preproc"] != df[nl_column])
        ]
        changed_utterances_df[[f"{nl_column}_no_preproc", nl_column]].to_csv(
            os.path.join(log_out_path, "changed_utterances.tsv"),
            sep="\t",
            index=False,
            header=["Original", "Preprocessed"],
        )

    df = df.drop(columns=[f"{nl_column}_no_preproc"], errors="ignore")
    return df


def main(params):
    ds_path = params.ds_path
    data_stram_name = params.datastream
    columns_to_keep = [
        "ID#",
        "Timestamp",
        "nl_instructions",
        "Experiment number",
        "Commander ID",
        "Trial",
    ]

    results_dir_path = "/".join(os.getcwd().split("/")[:-2]) + "/results/"
    ds_name = ds_path.split("/")[ds_path.split("/").index("downloaded_datasets") + 1]
    out_path = results_dir_path + ds_name

    if not os.path.exists(out_path):
        os.makedirs(out_path, exist_ok=True)
    out_path_logs = out_path + "/preprocessing_logs"

    if not os.path.exists(out_path_logs):
        os.makedirs(out_path_logs, exist_ok=True)

    exp4_log_file = os.path.join(
        out_path_logs, "experiment_4_commander_data_comparison_differing_rows.csv"
    )
    df = read_xlsx_to_df(ds_path, log_file=exp4_log_file)

    df = df.rename(columns={data_stram_name: "nl_instructions"})
    df = df[columns_to_keep]
    df = remove_tags(df, nl_column="nl_instructions", log_out_path=out_path_logs)
    df = drop_na(df, nl_column="nl_instructions")
    df = spacy_processing(df, nl_column="nl_instructions")
    df = get_dep_parse_tree(df, parse_tree_column="dep_parse_tree")
    df = get_constituency_parse_tree(
        df, nl_column="nl_instructions", parse_tree_column="constit_parse_tree"
    )
    df = get_seq_len(df)
    df = get_verbs(df)
    df = get_nouns(df)

    df.to_csv(out_path + "/" + ds_name + "_" + data_stram_name + "_results.csv")


if __name__ == "__main__":
    parser = ArgumentParser()
    """ 
    ds_path should be the absolute path to the folder, e.g.,
    "/home/slwanna/HRI_data_audit/downloaded_datasets/ARL-SCOUT" 
    """
    parser.add_argument("--ds_path", default="None", type=str, required=True)
    parser.add_argument(
        "--datastream",
        default="Commander",
        choices=["Commander", "DM->CMD", "DM->RN", "RN"],
        type=str,
        required=False,
    )

    args = parser.parse_args()

    main(args)
