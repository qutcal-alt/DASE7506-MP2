"""CPU NumPy inference for a frozen GPU-trained bounded correction network."""
import json
from pathlib import Path
import numpy as np
from common import parse_answer
from baseline import design
from rich_features import extract as graph_extract
from expression_features import extract as expression_extract,augment


def manual_features(base,graph,expression):
    return np.concatenate([base[:,:,:148],base[:,:,1172:],graph,augment(expression,True)],-1).astype(np.float32)


def network_input(manual,anchor):
    a=np.maximum(anchor,-20.).astype(np.float32)
    return np.concatenate([manual,a[:,:,None],(a-a.mean(1,keepdims=True))[:,:,None]],-1)


def forward(raw,state,bound):
    z=np.clip((raw-state['mu'])/state['sd'],-8,8).astype(np.float32)
    h=np.maximum(z@state['enc.weight'].T+state['enc.bias'],0)
    return (bound*np.tanh((h@state['out.weight'].T+state['out.bias'])[...,0])).astype(np.float32)


def probability(scores,valid):
    scores=np.where(valid,scores,-1e30);p=np.exp(scores-scores.max(1,keepdims=True))*valid
    return p/np.maximum(p.sum(1,keepdims=True),1e-30)


def anchor_scores(base,expression,valid,states):
    p=[]
    for x,state in zip([base,np.concatenate([base,augment(expression,True)],-1)],states):
        p.append(probability(((x-state['mu'])/state['sd'])@state['W']+state['b'][0],valid))
    return np.log(np.maximum(.75*p[0]+.25*p[1],1e-30))


def predict(bank,arrays,checkpoint):
    if not bank:return []
    root=Path(checkpoint);config=json.loads((root/'config.json').read_text())
    base=design(bank,arrays);g,_=graph_extract(bank);e=expression_extract(bank)
    valid=np.array([[parse_answer(c['text']) is not None for c in row['candidates']] for row in bank])
    states=[]
    for name in config['anchor_files']:
        with np.load(root/name,allow_pickle=False) as z:states.append(dict(z))
    anchor=anchor_scores(base,e,valid,states)
    raw=network_input(manual_features(base,g,e),anchor)
    corrections=[]
    for name in config['members']:
        with np.load(root/name,allow_pickle=False) as z:corrections.append(forward(raw,dict(z),config['bound']))
    residual=np.mean(corrections,0)
    scores=anchor+config['residual_weight']*residual
    return np.where(valid,scores,-1e30).argmax(1).astype(int).tolist()
