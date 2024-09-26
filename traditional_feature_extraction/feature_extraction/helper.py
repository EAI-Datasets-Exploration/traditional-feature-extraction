"""
For now, these functions just provide the full list spacy uses
to do dependency and pos tagging.
"""
import spacy


def print_spacy_pos_tags():
    nlp = spacy.load("en_core_web_sm")
    return nlp.get_pipe("tagger").labels


def print_spacy_dep_parse_tags():
    nlp = spacy.load("en_core_web_sm")
    return nlp.get_pipe("parser").labels


if __name__ == "__main__":
    print(f"list of spacy pos tags: {print_spacy_pos_tags()}")
    print(f"list of spacy dependency tags: {print_spacy_dep_parse_tags()}")
