"""
Consolidates all the natural language feature information we explore in 
analyzing and characterizing the complexity of each HRI dataset.

Each function expects a pandas dataframe with relevantly defined
column names. Then each function outputs a pandas dataframe.
"""
from collections import defaultdict
from functools import partial
import multiprocessing as mp
from concurrent.futures import ProcessPoolExecutor
import pandas as pd
import spacy
import stanza
import torch


def drop_na(df: pd.DataFrame, nl_column: str) -> pd.DataFrame:
    return df[df[nl_column].notnull()]


def get_num_unique_commands(fp: str, nl_column: str) -> int:
    df = pd.read_csv(fp)
    num_unique = len(df[nl_column].unique())
    s = f"Number of unique commands in dataset: {num_unique}"
    return s


def count_unique_unigrams(fp: str, nl_column: str) -> int:
    df = pd.read_csv(fp)

    all_text = " ".join(df[nl_column].astype(str).tolist())
    unigrams = all_text.split()  # You can use more sophisticated tokenization if needed
    unique_unigrams = set(unigrams)
    num_unique = len(unique_unigrams)
    s = f"Number of unique unigrams in dataset: {num_unique}"
    return s


### SPACY PARALLEL PROCESSING

VERBS = [
    "VERB",
    "VB",
    "VBD",
    "VBG",
    "VBN",
    "VBP",
    "VBZ",
]

NOUNS = [
    "NOUN",
    "NN",
    "NNP",
    "NNS",
]


def spacy_process_batch(
    texts,
    batch_size=1000,
    extract_nouns=False,
    extract_verbs=False,
    extract_seq_len=False,
):
    nlp = spacy.load(
        "en_core_web_sm", disable=["ner"]
    )  # Disable ner, but keep parser for POS tagging
    spacy_docs = nlp.pipe(texts, batch_size=batch_size)

    results = []
    for doc in spacy_docs:
        result = {
            "text": doc.text,
            "nouns": [],
            "verbs": [],
            "seq_len": len(doc) if extract_seq_len else None,
        }

        if extract_nouns or extract_verbs:
            for token in doc:
                if extract_nouns and token.pos_ in NOUNS:
                    result["nouns"].append(token.text)
                if extract_verbs and token.pos_ in VERBS:
                    result["verbs"].append(token.text)

        results.append(result)

    return results


def spacy_processing_parallel(
    df: pd.DataFrame,
    nl_column: str,
    spacy_col="spacy_parse",
    batch_size=1000,
    num_workers=4,
    **kwargs,
) -> pd.DataFrame:
    extract_nouns = kwargs.get("extract_nouns")
    extract_verbs = kwargs.get("extract_verbs")
    extract_seq_len = kwargs.get("extract_seq_len")

    df_copy = df.copy()

    # Split data into chunks for parallel processing
    texts = df_copy[nl_column].tolist()
    chunks = [texts[i : i + batch_size] for i in range(0, len(texts), batch_size)]

    spacy_batch_fn = partial(
        spacy_process_batch,
        batch_size=batch_size,
        extract_nouns=extract_nouns,
        extract_verbs=extract_verbs,
        extract_seq_len=extract_seq_len,
    )

    # Process the chunks in parallel
    with ProcessPoolExecutor(max_workers=num_workers) as executor:
        results = list(executor.map(spacy_batch_fn, chunks))

    # Flatten the list of results and assign to the dataframe
    df_copy[spacy_col] = [doc["text"] for result in results for doc in result]
    if extract_nouns:
        df_copy["nouns"] = [doc["nouns"] for result in results for doc in result]
    if extract_verbs:
        df_copy["verbs"] = [doc["verbs"] for result in results for doc in result]
    if extract_seq_len:
        df_copy["seq_len"] = [doc["seq_len"] for result in results for doc in result]

    return df_copy


###


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


###
### START: EVERYTHING TO DO WITH CONSTITUENCY PARSING GPU PARALLELIZATION!!!
###


# Function to load multiple Stanza models on the same GPU
def load_stanza_on_gpu(gpu_id, num_models_per_gpu=1):
    device = f"cuda:{gpu_id}" if torch.cuda.is_available() else "cpu"

    # Load multiple models on the same GPU
    models = []
    for _ in range(num_models_per_gpu):
        nlp = stanza.Pipeline(
            "en", processors="pos,tokenize,constituency", use_gpu=True, device=device
        )
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
        model_texts = (
            texts[i * chunk_size : (i + 1) * chunk_size]
            if i < model_count - 1
            else texts[i * chunk_size :]
        )
        for text in model_texts:
            doc = nlp(text)
            processed_constituencies = [
                sentence.constituency for sentence in doc.sentences
            ]
            results.append(processed_constituencies)

    return results


# The function that runs the parsing process with batching
def get_constituency_parse_tree(
    df: pd.DataFrame,
    nl_column: str,
    parse_tree_column="constit_parse_tree",
    num_models_per_gpu=1,
) -> pd.DataFrame:
    # Initialize the stanza pipeline with GPU enabled
    num_gpus = torch.cuda.device_count()
    texts = df[nl_column].tolist()
    chunks = [texts[i::num_gpus] for i in range(num_gpus)]

    # Distribute the work across GPUs
    with mp.Pool(processes=num_gpus) as pool:
        results = pool.starmap(
            parse_text_on_gpu,
            [
                (gpu_id, chunks[gpu_id], num_models_per_gpu)
                for gpu_id in range(num_gpus)
            ],
        )

    results = [item for sublist in results for item in sublist]

    # Update the DataFrame with the extracted constituency trees
    df[parse_tree_column] = results
    df = df.explode(parse_tree_column).reset_index(drop=True)
    return df


###
### END: EVERYTHING TO DO WITH CONSTITUENCY PARSING GPU PARALLELIZATION!!!
###

def flatten_nested_list(nested_list):
    flat_list = []
    for item in nested_list:
        if isinstance(item, list):
            flat_list.extend(flatten_nested_list(item))  # Recursively flatten lists
        else:
            flat_list.append(item)  # Append non-list items directly
    return flat_list



def flatten_and_combine_dicts(dict_list):
    combined_dict = defaultdict(list)

    for d in dict_list:
        if isinstance(d, dict):  # Ensure the element is a dictionary
            for key, value in d.items():
                if isinstance(value, list):
                    combined_dict[key].extend(value)
                elif isinstance(value, set):  # Handle cases where the value is a set
                    combined_dict[key].extend(list(value))  # Convert set to list before extending
                else:
                    combined_dict[key].append(value)
        else:
            print(f"gdi")

    # Remove duplicates from the combined lists and handle cases where elements are not hashable
    combined_dict = {
        key: list(set(value)) if isinstance(value, list) else value
        for key, value in combined_dict.items()
    }

    return dict(combined_dict)