"""
This file reads from a dataset's *_results.csv and 
visualizes different natural language features
"""

from argparse import ArgumentParser
from data_viz.utils import (
    get_word_len_hist,
    get_common_parses,
    get_word_cloud,
)
from data_viz.similarity_report import generate_report


def main(params):
    results_path = params.results_path

    get_word_len_hist(results_path)
    get_word_cloud(results_path, column="nl_instructions")
    get_word_cloud(results_path, column="nouns", bigrams=False)
    get_word_cloud(results_path, column="verbs", bigrams=False)

    get_common_parses(results_path)

    generate_report(results_path, use_trad_sim=True, use_neural_sim=True)


if __name__ == "__main__":
    parser = ArgumentParser()
    """ 
    results_path should be the absolute path to the *_results.csv file
    """
    parser.add_argument("--results_path", default="None", type=str, required=True)

    args = parser.parse_args()

    main(args)
