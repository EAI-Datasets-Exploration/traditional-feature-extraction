"""
Any preprocessing steps necessary to clean the downloaded datasets from the
dataset-download-scripts package to ready them for feature extraction.
"""

import pandas as pd
import string
import re

import pandas as pd
import string
import re


def clean_and_standardize_text(df: pd.DataFrame, nl_column: str) -> pd.DataFrame:
    df_copy = df.copy()  # Make a copy of the dataframe to avoid modifying the original

    # Define a helper function to clean and standardize text
    def clean_text(text):
        if pd.isna(text):
            return pd.NA

        text = str(text).strip()  # Strip leading/trailing whitespace
        text = " ".join(text.split())  # Remove extra spaces

        # Remove punctuation
        text = text.translate(str.maketrans("", "", string.punctuation))

        # Return None or pd.NA if the text is an empty string after cleaning
        if not text:
            return pd.NA

        return text

    def split_sentences(text):
        if pd.isna(text):
            return [pd.NA]

        # Split by period, question mark, or exclamation mark followed by space
        sentences = re.split(r"(?<=[.!?])\s+", text)

        # Clean each sentence individually
        sentences = [clean_text(sentence) for sentence in sentences if sentence]

        # Filter out any sentences that are pd.NA or empty strings
        sentences = [
            sentence
            for sentence in sentences
            if not pd.isna(sentence) and sentence != ""
        ]

        return sentences if sentences else [pd.NA]

    # Apply the split_sentences function to the specified column
    df_copy[nl_column] = df_copy[nl_column].apply(split_sentences)

    # Explode the list of sentences into separate rows
    df_copy = df_copy.explode(nl_column).reset_index(drop=True)

    # Replace empty strings with pd.NA (though it's unlikely at this point)
    df_copy[nl_column] = df_copy[nl_column].replace("", pd.NA)

    # Drop rows where the specified column is NaN
    df_copy = df_copy.dropna(subset=[nl_column])

    return df_copy
