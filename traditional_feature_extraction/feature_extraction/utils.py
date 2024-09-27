"""
Consolidates all the natural language feature information we explore in 
analyzing and characterizing the complexity of each HRI dataset.

Each function expects a pandas dataframe with relevantly defined
column names. Then each function outputs a pandas dataframe.
"""
from concurrent.futures import ProcessPoolExecutor
import math

import pandas as pd
import spacy
import stanza


def drop_na(df: pd.DataFrame, nl_column: str) -> pd.DataFrame:
    return df[df[nl_column].notnull()]


def get_num_unique_values(df: pd.DataFrame, nl_column: str) -> int:
    return len(df[nl_column].unique())


def spacy_processing(df: pd.DataFrame, nl_column: str, spacy_col="spacy_parse", batch_size=1000) -> pd.DataFrame:
    df_copy = df.copy()
    
    # Load the spaCy model and disable unnecessary components
    nlp = spacy.load("en_core_web_sm", disable=["ner", "parser"])  # Disable components you don't need

    # Use the nlp.pipe() method for batch processing
    texts = df_copy[nl_column].tolist()  # Get all texts in a list

    # Use nlp.pipe() to process texts in batches
    spacy_docs = nlp.pipe(texts, batch_size=batch_size)

    # Convert each document to JSON and store in the new column
    df_copy[spacy_col] = [doc.to_json() for doc in spacy_docs]

    return df_copy



def get_dep_parse_tree(
    df: pd.DataFrame, spacy_col="spacy_parse", parse_tree_column="dep_parse_tree"
) -> pd.DataFrame:
    def get_dep_tree(dict_tree: dict) -> list:
        dep_tree = []
        for word in dict_tree["tokens"]:
            dep_tree.append("head: " + str(word["head"]) + " dep: " + word["dep"])
        return "; ".join(dep_tree)

    df[parse_tree_column] = df[spacy_col].apply(get_dep_tree)
    return df

"""
START: EVERYTHING TO DO WITH CONSTITUENCY PARSING GPU PARALLELIZATION!!!

# TODO: Fix batching for ALFRED -- I would suspect that ALFRED row entries contain multiple sentences.
# maybe include a preprocessing step in ALFRED dataset_download pkg that handles this --- parses out multiple
# sentences as separate rows.

# To improve parallelization, setup multiple pipeline processes on GPU and reassmble. I would have to write code to
#  manage the total number of CUDA devices on the system in order to split things up.

# pipeline object accepts: device parameter, e.g., cuda:0 or cuda:1 or ...

# see respecting document boundaries here: https://stanfordnlp.github.io/stanza/getting_started.html#building-a-pipeline
"""
# The function that runs the parsing process with batching (no multiprocessing)
def get_constituency_parse_tree(df: pd.DataFrame, nl_column: str, parse_tree_column="constit_parse_tree", batch_size=32768) -> pd.DataFrame:
    # Initialize the stanza pipeline with GPU enabled

    nlp = stanza.Pipeline(lang="en", processors="pos, tokenize, constituency", use_gpu=True) 

    # Define the batch parsing function to handle multiple texts at once using GPU
    def batch_parse(texts):    
        # Join texts with a separator to treat them as separate documents
        doc = nlp("\n\n".join(texts))  # Use a double newline to separate texts in the batch
        
        # Ensure that the number of parsed sentences matches the number of input texts
        if len(doc.sentences) != len(texts):
            parsed_texts = set(doc.sentences)
            original_texts = set(texts)

            # Find items in list1 but not in list2
            only_in_list1 = parsed_texts.difference(original_texts)
            
            print(f"The items added by parser:\n{only_in_list1}")
            raise ValueError(f"Number of parsed sentences ({len(doc.sentences)}) does not match number of input texts ({len(texts)})")
        
        return [sentence.constituency for sentence in doc.sentences]  # Return list of parse trees

    # A helper function to process batches of rows
    def process_batch(batch_rows, nl_column):
        # Extract the text for the batch
        batch_texts = batch_rows[nl_column].tolist()  # Convert the column to a list of texts
        
        # Sanitize input by stripping extra newlines and whitespaces
        batch_texts = [text.strip() for text in batch_texts]
        
        # Perform the batch parsing on GPU
        return batch_parse(batch_texts)

    df_copy = df.copy()
    
    # Split the rows into batches
    num_batches = math.ceil(len(df_copy) / batch_size)
    row_batches = [df_copy.iloc[i*batch_size:(i+1)*batch_size] for i in range(num_batches)]
    
    # Store results of parsing
    results = []
    for batch in row_batches:
        batch_result = process_batch(batch, nl_column)
        results.extend(batch_result)
    
    # Ensure that the number of results matches the number of rows in the DataFrame
    if len(results) != len(df_copy):
        raise ValueError(f"Number of results ({len(results)}) does not match number of rows in DataFrame ({len(df_copy)})")
    
    # Assign the results to the DataFrame
    df_copy[parse_tree_column] = results
    
    return df_copy


"""
END: EVERYTHING TO DO WITH CONSTITUENCY PARSING GPU PARALLELIZATION!!!
"""

def get_seq_len(
    df: pd.DataFrame, spacy_col="spacy_parse", word_len_column="word_len"
) -> pd.DataFrame:
    def get_len(dict_tree: dict) -> int:
        return len(dict_tree["tokens"])

    df[word_len_column] = df[spacy_col].apply(get_len)
    return df


def get_verbs(
    df: pd.DataFrame, spacy_col="spacy_parse", verb_column="verbs"
) -> pd.DataFrame:
    def get_v(dict_tree: dict) -> list:
        verb_list = [
            "VERB",
            "VB",
            "VBD",
            "VBG",
            "VBN",
            "VBP",
            "VBZ",
        ]
        verbs = []
        for word in dict_tree["tokens"]:
            if word["pos"] in verb_list:
                verbs.append(word["lemma"])
        return verbs

    df[verb_column] = df[spacy_col].apply(get_v)
    return df


def get_nouns(
    df: pd.DataFrame, spacy_col="spacy_parse", noun_column="nouns"
) -> pd.DataFrame:
    def get_n(dict_tree: dict) -> list:
        noun_list = [
            "NOUN",
            "NN",
            "NNP",
            "NNS",
        ]
        nouns = []
        for word in dict_tree["tokens"]:
            if word["pos"] in noun_list:
                nouns.append(word["lemma"])
        return nouns

    df[noun_column] = df[spacy_col].apply(get_n)
    return df
