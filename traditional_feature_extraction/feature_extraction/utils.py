"""
Consolidates all the natural language feature information we explore in 
analyzing and characterizing the complexity of each HRI dataset.

Each function expects a pandas dataframe with relevantly defined
column names. Then each function outputs a pandas dataframe.
"""
from concurrent.futures import ProcessPoolExecutor
import math
import multiprocessing as mp
import pandas as pd
import spacy
import stanza
import torch


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
"""
# Function to load multiple Stanza models on the same GPU
def load_stanza_on_gpu(gpu_id, num_models_per_gpu=1):
    device = f'cuda:{gpu_id}' if torch.cuda.is_available() else 'cpu'
    
    # Load multiple models on the same GPU
    models = []
    for _ in range(num_models_per_gpu):
        nlp = stanza.Pipeline('en', processors='pos,tokenize,constituency', use_gpu=True, device=device)
        models.append(nlp)
    
    return models

# Parse text using Stanza models on a specific GPU
def parse_text_on_gpu(gpu_id, texts, num_models_per_gpu=1):
    models = load_stanza_on_gpu(gpu_id, num_models_per_gpu=num_models_per_gpu)
    
    results = []
    
    # Distribute text across models
    model_count = len(models)
    chunk_size = len(texts) // model_count
    
    for i, nlp in enumerate(models):
        model_texts = texts[i * chunk_size : (i + 1) * chunk_size] if i < model_count - 1 else texts[i * chunk_size:]
        for text in model_texts:
            doc = nlp(text)
            results.append(doc)
    
    return results

# The function that runs the parsing process with batching
def get_constituency_parse_tree(df: pd.DataFrame, nl_column: str, parse_tree_column="constit_parse_tree", num_models_per_gpu=64) -> pd.DataFrame:
    # Initialize the stanza pipeline with GPU enabled
    num_gpus = torch.cuda.device_count()
    texts = df[nl_column].tolist()
    chunks = [texts[i::num_gpus] for i in range(num_gpus)]

    # Distribute the work across GPUs
    with mp.Pool(processes=num_gpus) as pool:
        results = pool.starmap(parse_text_on_gpu, [(gpu_id, chunks[gpu_id], num_models_per_gpu) for gpu_id in range(num_gpus)])
    
    results = [item for sublist in results for item in sublist]
    
    # Extract constituency trees
    constituency_trees = [sentence.constituency for doc in results for sentence in doc.sentences]
    
    # Update the DataFrame with the results
    df[parse_tree_column] = constituency_trees
    return df
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
