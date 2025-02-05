from typing import List
from attrs import define, field

from utils import load_json_file
from preprocess import QUAC_PATH


@define
class QuACAnswer:
    text: str
    start: int


@define
class QuACQABase:
    id: str
    followup: str
    yesno: str
    question: str
    answers: List[QuACAnswer]


@define
class QuACQA(QuACQABase):
    prev_qa: List[QuACQABase] = field(factory=list)


@define
class QuACPara:
    id: str
    context: str
    qas: List[QuACQA] = field(factory=list)


@define
class QuACSec:
    title: str
    background: str
    section_title: str
    paragraphs: List[QuACPara] = field(factory=list)


def parse_quac(data):
    conversations = []
    for instance in data['data']:
        title = instance['title'] # entity e in the paper
        background = instance['background'] # b in the paper
        section_title = instance['section_title'] # t in the paper
        paragraphs = instance['paragraphs']
        quac_paras = []
        for paragraph in paragraphs:
            context = paragraph['context'] # s in the paper
            qas = paragraph['qas']
            para_id = paragraph['id']
            prev_qa = []
            quac_qas = []
            for qa in qas:
                qa_id = qa['id']
                followup = qa['followup']
                yesno = qa['yesno']
                question = qa['question']
                quac_answers = []
                for answer in qa['answers']:
                    answer_text = answer['text']
                    answer_start = answer['answer_start']
                    quac_answer = QuACAnswer(answer_text, answer_start)
                    quac_answers.append(quac_answer)
                quac_qa_base = QuACQABase(qa_id, followup, yesno, question, quac_answers)
                prev_qa_copy = prev_qa.copy()
                quac_qa = QuACQA(qa_id, followup, yesno, question, quac_answers, prev_qa_copy)
                quac_qas.append(quac_qa)
                prev_qa.append(quac_qa_base)
            quac_para = QuACPara(para_id, context, quac_qas)
            quac_paras.append(quac_para)
        quac_sec = QuACSec(title, background, section_title, quac_paras)
        conversations.append(quac_sec)
    return conversations


if __name__ == '__main__':
    train_path = QUAC_PATH.joinpath('train_v0.2.json')
    quac_data = load_json_file(train_path)

    quac_conversations = parse_quac(quac_data)