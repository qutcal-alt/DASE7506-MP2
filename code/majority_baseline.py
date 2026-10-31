"""Untrained valid-answer plurality; ties use the first supplied candidate."""
from collections import Counter
from common import parse_answer


def predict(bank,arrays,checkpoint=None):
    selected=[]
    for row in bank:
        answers=[parse_answer(c['text']) for c in row['candidates']]
        counts=Counter(a for a in answers if a is not None)
        if not counts:selected.append(0);continue
        maximum=max(counts.values())
        selected.append(next(i for i,a in enumerate(answers) if a is not None and counts[a]==maximum))
    return selected
