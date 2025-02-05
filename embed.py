import spacy

# pip install https://github.com/explosion/spacy-experimental/releases/download/v0.6.1/en_coreference_web_trf-3.4.0a2-py3-none-any.whl
nlp_coref = spacy.load("en_coreference_web_trf")
nlp = spacy.load("en_core_web_sm")

PRONOUNS = ["he", "she", "it", "they", "them", "him", "her", "his", "hers", "its", "their", "theirs", "himself", "herself", "itself", "themselves", "this", "that", "these", "those", "who", "whom", "whose", "which", "what", "where", "when", "why", "how", "all", "any", "anybody", "anyone", "anything", "both", "each", "other", "either", "everyone", "everybody", "everything", "few", "many", "neither", "nobody", "none", "no one", "nothing", "one", "some", "somebody", "someone", "something", "such", "myself", "yourself", "yourselves", "himself", "herself", "itself", "ourselves", "themselves", "whoever", "whomever", "whichever", "whatever", "you", "I", "me", "we", "us", "my", "your", "mine", "yours", "our", "ours", "their", "theirs", "his", "her", "hers", "its", "whose", "what", "which", "that", "this", "these", "those", "all", "any", "anybody", "anyone", "anything", "both", "each", "other", "either", "everyone", "everybody", "everything", "few", "many", "neither", "nobody", "none", "no one", "nothing", "one", "some", "somebody", "someone", "something", "such", "myself", "yourself", "yourselves", "himself", "herself", "itself", "ourselves", "themselves", "whoever", "whomever", "whichever", "whatever", "you", "I", "me", "we", "us", "my", "your", "mine", "yours", "our", "ours", "their", "theirs", "his", "her", "hers", "its", "whose", "what", "which", "that", "this", "these", "those", "all", "any", "anybody", "anyone", "anything", "both", "each", "other", "either", "everyone", "everybody", "everything", "few", "many", "neither", "nobody", "none", "no one", "nothing", "one", "some", "somebody", "someone", "something", "such", "myself", "yourself", "yourselves", "himself", "herself", "itself", "ourselves", "themselves", "whoever", "whomever", "whichever", "whatever", "you", "I", "me", "we", "us", "my", "your", "mine", "yours", "our", "ours", "their", "theirs", "his", "her", "hers", "its", "whose", "what", "which", "that", "this", "these", "those", "all", "any", "anybody", "anyone", "anything", "both", "each", "other", "either", "name"]


class Entity:
    def __init__(self, text: str, start: int=-1, end: int=-1, start_char: int=-1, end_char: int=-1):
        self.text = text
        self.start = start
        self.end = end
        self.start_char = start_char
        self.end_char = end_char

    def __str__(self):
        return f"{self.text} ({self.start}-{self.end})"

    def __repr__(self):
        return f"{self.text} ({self.start}-{self.end})"

    def __eq__(self, other):
        return (str(self.text).lower() in str(other.text).lower() or str(other.text).lower() in str(self.text).lower()) and (self.start <= other.start <= self.end or other.start <= self.start <= other.end)

# Entity Embedding
def entities(text: str) -> list:
    # NER or heuristic to recognize entities
    doc_coref = nlp_coref(text)
    doc = nlp(text)
    es = [Entity(ent.text, ent.start, ent.end, ent.start_char, ent.end_char) for ent in doc.ents]
    clusters = [val for key, val in doc_coref.spans.items()]
    clusters = [([Entity(x, x.start, x.end, x.start_char, x.end_char) for x in cluster]) for cluster in clusters]
    for ent in es:
        if all(ent not in cluster for cluster in clusters):
            clusters.append([ent])
    return clusters


if __name__ == "__main__":
    from preprocess import COQA_DEV_PATH
    from preprocess.coqa_data import parse_coqa
    from utils import load_json_file
    coqa_data = load_json_file(COQA_DEV_PATH)['data']

    coqa_conversations = parse_coqa(coqa_data)
    print(len(coqa_conversations))

    # print(entities(coqa_conversations[0].context))
