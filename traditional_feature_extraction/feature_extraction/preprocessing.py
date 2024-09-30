"""
Any preprocessing steps necessary to clean the downloaded datasets from the
dataset-download-scripts package to ready them for feature extraction.
"""

import pandas as pd
import string


def clean_and_standardize_text(df: pd.DataFrame, nl_column: str) -> pd.DataFrame:
    df_copy = df.copy()  # Make a copy of the dataframe to avoid modifying the original

    # Define a helper function to clean and standardize text
    def clean_text(text):
        if pd.isna(text):
            return pd.NA

        text = str(text)
        text = text.strip()
        text = " ".join(text.split())

        text = text.translate(str.maketrans("", "", string.punctuation))

        return text

    # Apply the clean_text function to the specified column
    df_copy[nl_column] = df_copy[nl_column].apply(clean_text)

    # Replace empty strings with pd.NA
    df_copy[nl_column] = df_copy[nl_column].replace("", pd.NA)

    # Drop rows where the specified column is NaN
    df_copy = df_copy.dropna(subset=[nl_column])

    # Strip spaces again, just in case
    df_copy[nl_column] = df_copy[nl_column].str.strip()

    return df_copy
