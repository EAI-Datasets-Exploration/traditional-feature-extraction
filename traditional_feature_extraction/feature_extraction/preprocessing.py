import pandas as pd
import string

def clean_and_standardize_text(df: pd.DataFrame, nl_column: str) -> pd.DataFrame:
    df_copy = df.copy()  # Make a copy of the dataframe to avoid modifying the original
    
    # Define a helper function to clean and standardize text
    def clean_text(text):
        # 1. Convert text to string in case there are NaNs or other non-string values
        text = str(text)
        
        # 2. Strip leading/trailing spaces
        text = text.strip()
        
        # 3. Replace multiple spaces with a single space
        text = " ".join(text.split())
        
        # 4. Remove punctuation using string.punctuation
        text = text.translate(str.maketrans("", "", string.punctuation))
        
        return text
    
    # Apply the clean_text function to the specified column
    df_copy[nl_column] = df_copy[nl_column].apply(clean_text)

    # drop empty rows
    df_copy[nl_column] = df_copy[nl_column].replace("", pd.NA)
    df_copy[nl_column] = df_copy[nl_column].str.strip()
    df_copy = df_copy.dropna(subset=[nl_column])
    return df_copy
