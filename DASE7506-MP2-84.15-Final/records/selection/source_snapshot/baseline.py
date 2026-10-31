"""Frozen 83.08% verifier: text, embeddings and arithmetic-provenance features."""
import json
from pathlib import Path
import numpy as np
from common import parse_answer
from features import build_features
from quantity_features import extract


def design(bank,arrays):
    text,names,_=build_features(bank,arrays)
    # Frozen selected recipe uses the raw provided embedding, then standardises it.
    e=np.asarray(arrays['embeddings'],dtype=np.float64)
    q=extract(bank)
    delta=q-q.mean(1,keepdims=True)
    provenance=np.concatenate([q,delta,delta/(q.std(1,keepdims=True)+1e-6)],-1).astype(np.float64)
    return np.concatenate([text,e,provenance],-1)


def predict(bank,arrays,checkpoint):
    if not bank:return []
    root=Path(checkpoint)
    config=json.loads((root/'config.json').read_text())
    member=config['components'][0]
    if len(config['components'])!=1 or member['normalised'] or not member['relative']:
        raise ValueError('This package supports the frozen raw-embedding provenance model')
    x=design(bank,arrays)
    with np.load(root/member['file'],allow_pickle=False) as z:
        scores=((x-z['mu'])/z['sd'])@z['W']+z['b'][0]
    valid=np.array([[parse_answer(c['text']) is not None for c in row['candidates']] for row in bank])
    return np.where(valid,scores,-1e30).argmax(1).astype(int).tolist()
