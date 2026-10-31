"""Small tree corrections; CPU inference depends only on NumPy and input text."""
import json
from pathlib import Path
import numpy as np
from common import parse_answer
from baseline import design
from rich_features import extract as graph_extract
from expression_features import extract as expression_extract
from gpu_residual_submission import manual_features,network_input,forward,anchor_scores


def tree_input(manual,conditions,anchor,block):
    raw=network_input(manual,anchor);parts=[raw]
    if block in ('conditions','relative'):parts.append(conditions)
    if block=='relative':
        graph=manual[:,:,202:236];delta=graph-graph.mean(1,keepdims=True)
        parts.extend([delta,delta/(graph.std(1,keepdims=True)+1e-6)])
    return np.nan_to_num(np.concatenate(parts,-1),nan=0.,posinf=1e8,neginf=-1e8).astype(np.float32)


def tree_forward(x,state):
    shape=x.shape[:2];flat=x.reshape(-1,x.shape[-1]);roots=state['roots']
    nodes=np.broadcast_to(roots,(len(flat),len(roots))).copy();row=np.arange(len(flat))[:,None]
    for _ in range(int(state['max_depth'])):
        features=state['feature'][nodes];active=features>=0
        if not active.any():break
        left=flat[row,np.maximum(features,0)]<=state['threshold'][nodes]
        children=np.where(left,state['left'][nodes],state['right'][nodes])
        nodes=np.where(active,children,nodes)
    assert (state['feature'][nodes]<0).all()
    return state['value'][nodes].sum(1).reshape(shape)


def predict(bank,arrays,checkpoint):
    if not bank:return []
    root=Path(checkpoint);config=json.loads((root/'config.json').read_text())
    base=design(bank,arrays);g,c=graph_extract(bank);e=expression_extract(bank)
    manual=manual_features(base,g,e)
    valid=np.array([[parse_answer(candidate['text']) is not None for candidate in row['candidates']] for row in bank])
    states=[]
    for name in config['anchor_files']:
        with np.load(root/name,allow_pickle=False) as z:states.append(dict(z))
    anchor=anchor_scores(base,e,valid,states);score=anchor.copy()
    if config['reference_weight']:
        raw=network_input(manual,anchor);residual=[]
        for name in config['reference_members']:
            with np.load(root/name,allow_pickle=False) as z:residual.append(forward(raw,dict(z),1.))
        score+=config['reference_weight']*np.mean(residual,0)
    for part in config.get('trees',[]):
        with np.load(root/part['file'],allow_pickle=False) as z:state=dict(z)
        score+=part['weight']*tree_forward(tree_input(manual,c,anchor,part['block']),state)
    return np.where(valid,score,-1e30).argmax(1).astype(int).tolist()
