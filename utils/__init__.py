import json
import spacy
from pathlib import Path

from preprocess import COQA_PATH


def load_json_file(file_path):
    with open(file_path, 'r') as file:
        return json.load(file)


def tokenize_sentences(text):
    # Load the spaCy model
    nlp = spacy.load("en_core_web_sm")

    # Process the text
    doc = nlp(text)

    # Extract sentences
    sentences = [sent.text for sent in doc.sents]

    return sentences