from typing import List
from attrs import define, field

from utils import load_json_file
from preprocess import COQA_PATH


@define
class CoQAQABase:
    id: int
    question: str
    answer: str
    rationale: str
    rationale_start: int
    rationale_end: int


@define
class CoQAQA(CoQAQABase):
    prev_qa: List[CoQAQABase]


@define
class CoQAConv:
    id: str
    source: str
    filename: str
    context: str
    qas: List[CoQAQA] = field(factory=list)


def parse_coqa(data):
    conversations = []
    for instance in data:
        conv_id = instance['id']
        source = instance['source']
        filename = instance['filename']
        context = instance['story']
        prev_qa = []
        qas = []
        for i, question in enumerate(instance['questions']):
            q_id = question['turn_id']
            q_text = question['input_text']
            answer = instance['answers'][i]
            a_id = answer['turn_id']
            assert q_id == a_id
            a_text = answer['input_text']
            rationale = answer['span_text']
            rationale_start = answer['span_start']
            rationale_end = answer['span_end']
            qa_base = CoQAQABase(id=q_id, question=q_text, answer=a_text, rationale=rationale,
                                 rationale_start=rationale_start, rationale_end=rationale_end)
            prev_qa_copy = prev_qa.copy()
            qa = CoQAQA(id=q_id, question=q_text, answer=a_text, rationale=rationale,
                        rationale_start=rationale_start, rationale_end=rationale_end, prev_qa=prev_qa_copy)
            prev_qa.append(qa_base)
            qas.append(qa)
        coqa_conv = CoQAConv(id=conv_id, source=source, filename=filename, context=context, qas=qas)
        conversations.append(coqa_conv)

    return conversations


if __name__ == '__main__':
    dev_path = COQA_PATH.joinpath('coqa-dev-v1.0.json')
    coqa_data = load_json_file(dev_path)['data']

    coqa_conversations = parse_coqa(coqa_data)
    print(coqa_conversations)