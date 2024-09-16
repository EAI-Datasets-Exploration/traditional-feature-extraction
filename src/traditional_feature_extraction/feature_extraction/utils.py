"""
Consolidates all the natural language feature information we explore in 
analyzing and characterizing the complexity of each HRI dataset.

Each function expects a pandas dataframe with relevantly defined
column names. Then each function outputs a pandas dataframe.
"""

import pandas as pd
import spacy
import stanza


def drop_na(df: pd.DataFrame, nl_column: str) -> pd.DataFrame:
    return df[df[nl_column].notnull()]


def get_num_unique_values(df: pd.DataFrame, nl_column: str) -> int:
    return len(df[nl_column].unique())


def spacy_processing(
    df: pd.DataFrame, nl_column: str, spacy_col="spacy_parse"
) -> pd.DataFrame:
    df_copy = df.copy()
    nlp = spacy.load("en_core_web_sm")

    def parse(raw_text: str) -> dict:
        doc = nlp(raw_text)
        return doc.to_json()

    df_copy.loc[:, spacy_col] = df_copy.loc[:, nl_column].apply(parse)
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


def get_constituency_parse_tree(
    df: pd.DataFrame, nl_column: str, parse_tree_column="constit_parse_tree"
) -> pd.DataFrame:
    df_copy = df.copy()
    nlp = stanza.Pipeline(lang="en", processors="tokenize,pos,constituency")

    def parse(raw_text: str) -> dict:
        doc = nlp(raw_text)
        return doc.sentences[0].constituency

    df_copy.loc[:, parse_tree_column] = df_copy.loc[:, nl_column].apply(parse)
    return df_copy


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
