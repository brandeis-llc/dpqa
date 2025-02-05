# file to create entity graphs
from embed import entities, Entity, EntityGraph
import penman

import spacy

# load in data
from preprocess import COQA_DEV_PATH, COQA_PATH
from preprocess.coqa_data import parse_coqa
from utils import load_json_file

# from holographic.amr import get_variables, get_vocab

nlp = spacy.load("en_core_web_sm")


def entities_from_text(conversation, graphs):

    coref_groups = entities(conversation)
    # load in AMR graphs

    ents = []
    # doc = nlp(conversation.context)
    sentence_tokens = []
    k = 1
    for graph in graphs:
        while f"snt{k}" not in graph.metadata:
            k += 1
        toks = []
        for token in nlp(graph.metadata[f"snt{k}"]):
            toks.append(token.text)
        sentence_tokens.append(toks)
        k += 1
    #sentence_tokens = [[token.text for token in nlp(graph.metadata[f"snt{i+1}"])] for i, graph in enumerate(graphs)]

    # [[token.text for token in sent] for sent in doc.sents]

    offset = 0
    # print(coref_groups)
    # print(len(coref_groups))
    entity_groups = {i: [] for i, group in enumerate(coref_groups)}
    node_groups = {i: [] for i, group in enumerate(coref_groups)}
    for i, graph in enumerate(graphs):
        alignments = []
        for triple in graph.triples + graph.attributes():
            if triple in penman.surface.alignments(graph):
                value = triple[2][1:-1] if triple[2].startswith("\"") else triple[2]
                alignments.append((penman.surface.alignments(graph)[triple].indices[0], value))

        variables = get_variables(graph)
        seen = []

        for triple in graph.triples + graph.attributes():
            seeing = 1
            if triple[1] == ":instance":
                continue
            if triple[0] in seen and triple[2] in seen:
                continue
            elif triple[0] in seen:
                seen.append(triple[2])
                seeing = 2
            elif triple[2] in seen:
                seen.append(triple[0])
                seeing = 0

            if triple[0] in variables:
                parent = variables[triple[0]]
            else:
                parent = triple[0]

            if triple[2] in variables:
                child = variables[triple[2]]
            else:
                child = triple[2]

            # remove quotes from around parent and child
            parent = parent[1:-1] if parent[0].startswith("\"") else parent
            child = child[1:-1] if child[0].startswith("\"") else child

            if seeing == 1 or seeing == 2:
                ents.append(parent)
            if seeing == 0 or seeing == 1:
                ents.append(child)

            # get alignment of child
            for alignment in alignments:
                if seeing == 1 or seeing == 2:
                    if child == alignment[1]:
                        child_align = alignment[0]+1
                        # print("Child:", child, "Alignment:", offset+child_align, "Sentence:", i+1)
                        possible_ent = Entity(child, start=offset+child_align, end=offset+child_align+1)
                        groups = [possible_ent in x for x in coref_groups]
                        if any(groups):
                            if ((parent, triple[1], child), triple, i+1) not in entity_groups[groups.index(True)]:
                                entity_groups[groups.index(True)].append(((parent, triple[1], child), triple, i+1))
                                node_groups[groups.index(True)].append((child, triple[2]))
                            # print("Parent:", parent, "Child:", child, "Alignment:", offset+child_align, "Coref Group:", groups.index(True)+1)
                            # print()
                if seeing == 0 or seeing == 1:
                    if parent == alignment[1]:
                        parent_align = alignment[0]+1
                        # print("Parent:", parent, "Alignment:", offset+parent_align, "Sentence:", i+1)
                        possible_ent = Entity(parent, start=offset+parent_align, end=offset+parent_align+1)
                        groups = [possible_ent in x for x in coref_groups]
                        if any(groups):
                            if ((parent, triple[1], child), triple, i + 1) not in entity_groups[groups.index(True)]:
                                entity_groups[groups.index(True)].append(((parent, triple[1], child), triple, i + 1))
                                node_groups[groups.index(True)].append((parent, triple[0]))
                            # print("Parent:", parent, "Child:", child, "Alignment:", offset+parent_align, "Coref Group:", groups.index(True)+1)
                            # print()
                    # print(triple.alignment)

        offset += len(sentence_tokens[i])  # add the length of the i-th sentence
    # currently returns groups of entities that are coreferential
    return entity_groups, coref_groups, node_groups


if __name__ == "__main__":
    coqa_data = load_json_file(COQA_DEV_PATH)['data']

    coqa_conversations = parse_coqa(coqa_data)
    # print(len(coqa_conversations))

    entity_groups, coref_groups, name_groups = entities_from_text(coqa_conversations[1].context)
