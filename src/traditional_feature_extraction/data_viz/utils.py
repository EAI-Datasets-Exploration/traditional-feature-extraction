"""
Helper visualization functions. 
"""
import os
import matplotlib.pyplot as plt
import pandas as pd
import re
import seaborn as sns
import spacy
import string
from wordcloud import WordCloud


def get_total_instructions(fp: str) -> str:
    df = pd.read_csv(fp)
    num_data = len(df)
    return f"dataset length: {num_data}"


def get_unique_ds(fp: str, word_len_column="nl_instructions") -> str:
    df = pd.read_csv(fp)
    unique_values = df[word_len_column].nunique()
    return f"num unique values: {unique_values}"


def get_ds_name(fp: str) -> str:
    match = re.search(r"results\/([^\/]+)\/", fp)

    if match:
        folder_name = match.group(1)
    else:
        print("No file name found.")
        raise ValueError
    return folder_name


def get_word_len_hist(fp: str, word_len_column="word_len") -> None:
    """
    This function only saves the word counts for a single dataset of interest
    """
    result_dir = os.path.split(fp)[0]
    df = pd.read_csv(fp)

    # Just a little hack to make the plot readable with many values,
    # should try to find a better way to do this
    unique_values = df[word_len_column].nunique()
    width = max(10, unique_values * 0.35)
    plt.figure(figsize=(width, 6))

    ds_name = get_ds_name(fp)
    title = f"Sequence Length Histogram for {ds_name}"

    plot = sns.countplot(x=word_len_column, data=df).set_title(title)
    fig = plot.get_figure()
    fig.savefig(result_dir + "/seq_len_hist.pdf")
    plt.clf()


def get_word_len_hist_summary(results_files: list, truncation_limit=30):
    """
    This function saves the counts of all datsets and summarizes them in
    a single chart.
    """
    fp = results_files[0]
    result_dir = os.path.split(os.path.split(fp)[0])[0]

    df_list = []
    for file in results_files:
        curr_df = pd.read_csv(file)
        curr_df = (
            curr_df.word_len.value_counts(normalize=True)
            .reset_index(name="Count")
            .rename(columns={"index": "word_len"})
        )
        curr_df["ds_name"] = get_ds_name(file)
        df_list.append(curr_df)

    result = pd.concat(df_list).reset_index(drop=True)

    def truncate(x, truncation_limit=truncation_limit):
        if x > truncation_limit:
            x = truncation_limit
        return x

    result["word_len"] = result["word_len"].apply(truncate)

    title = "Sequence Lengths for EAI Datasets"
    ylabel = "Percentage of Commands"
    xlabel = "Number of Words"

    sns.barplot(result, x="word_len", y="Count", hue="ds_name", errorbar=None).set(
        title=title, ylabel=ylabel, xlabel=xlabel
    )
    plt.savefig(result_dir + "/seq_len_hist.pdf")
    plt.clf()


def viz_parse_tree(indx: int, fp: str, sentence: str) -> None:
    """
    fp here should be the results_dir location
    """
    nlp = spacy.load("en_core_web_sm")
    doc = nlp(sentence)
    svg = spacy.displacy.render(doc, style="dep", jupyter=False)

    file_name = "-".join([w.text for w in doc if not w.is_punct]) + ".svg"
    output_path = fp + "/" + str(indx) + "_" + file_name
    with open(output_path, "w", encoding="utf-8") as f:
        f.write(svg)


def get_common_parses(
    fp: str,
    dep_parse_col="dep_parse_tree",
    nl_column="nl_instructions",
    number_of_trees=25,
) -> None:
    result_dir = os.path.split(fp)[0]
    df = pd.read_csv(fp)
    index_keys = df[dep_parse_col].unique()
    index_values = range(len(index_keys))
    index_dict = {index_keys[i]: index_values[i] for i in range(len(index_keys))}
    df["mapped_dep"] = df[dep_parse_col].map(index_dict)

    out_path = result_dir + "/" + "dep_parse_data"
    if not os.path.exists(out_path):
        os.makedirs(out_path, exist_ok=True)

    top_indexes = df["mapped_dep"].value_counts().head(number_of_trees).index.tolist()

    for value in top_indexes:
        rows = df.loc[df["mapped_dep"] == value].iloc[0]
        original_sent = rows[nl_column]
        viz_parse_tree(value, out_path, original_sent)

    plot = sns.countplot(x="mapped_dep", data=df).set_title("Dep Parse Histogram")

    fig = plot.get_figure()
    fig.savefig(out_path + "/dep_parse_hist.pdf")
    plt.clf()


def get_word_cloud(fp: str, column: str, bigrams=True) -> None:
    result_dir = os.path.split(fp)[0]
    df = pd.read_csv(fp)
    text = " ".join(
        (df[column]).astype(str)
    )  # ensures all column values will be interpreted as string
    text = text.translate(
        str.maketrans("", "", string.punctuation)
    )  # removes punctuation from string, if it exists
    wordcloud = WordCloud().generate(text)
    wordcloud = WordCloud(
        background_color="white",
        max_words=100,
        max_font_size=40,
        relative_scaling=0.5,
        collocations=bigrams,
    ).generate(text)

    ds_name = get_ds_name(fp)
    title = f"{column} Word Cloud for {ds_name} "

    plt.title(title, fontsize=13)
    wordcloud_svg = wordcloud.to_svg()

    with open(
        result_dir + "/" + column + "_word_cloud.svg", "w+", encoding="utf-8"
    ) as f:
        f.write(wordcloud_svg)
