"""Compare tied-rank feature views without repeating text preprocessing."""
import json
from pathlib import Path
import numpy as np
from common import parse_answer
from baseline import design
from rich_features import extract as graph_extract
from expression_features import extract as expression_extract
from gpu_residual_submission import manual_features,network_input,forward,anchor_scores,probability
from tree_correction_submission import tree_input,tree_forward


def average_ranks(values):
    lower=(values[:,None,:]<values[:,:,None]).sum(-1)
    equal=(values[:,None,:]==values[:,:,None]).sum(-1)
    return lower+(equal-1)*.5


def rank_view(base,strength):
    changed=base.copy()
    for j in range(16):
        # The original text block has 100 raw features and 16 (z,delta,rank) triplets.
        delta=base[:,:,101+3*j];index=102+3*j
        changed[:,:,index]=(1-strength)*base[:,:,index]+strength*average_ranks(delta)
    return changed


def load_states(root,config):
    names=set(config['anchor_files']+config['reference_members']+[p['file'] for p in config['trees']])
    states={}
    for name in names:
        with np.load(Path(root)/name,allow_pickle=False) as z:states[name]=dict(z)
    return states


def score_features(base,graph,conditions,expression,valid,config,states):
    manual=manual_features(base,graph,expression)
    anchor=anchor_scores(base,expression,valid,[states[n] for n in config['anchor_files']]);score=anchor.copy()
    if config['reference_weight']:
        raw=network_input(manual,anchor)
        score+=config['reference_weight']*np.mean([forward(raw,states[n],1.) for n in config['reference_members']],0)
    for part in config['trees']:
        raw=tree_input(manual,conditions,anchor,part['block'])
        score+=part['weight']*tree_forward(raw,states[part['file']])
    return score


def mixed_score(original,corrected,valid,weight,mode):
    if mode=='probability':return (1-weight)*probability(original,valid)+weight*probability(corrected,valid)
    return (1-weight)*original+weight*corrected


def predict(bank,arrays,checkpoint):
    if not bank:return []
    root=Path(checkpoint);config=json.loads((root/'config.json').read_text());states=load_states(root,config)
    base=design(bank,arrays);g,c=graph_extract(bank);e=expression_extract(bank)
    valid=np.array([[parse_answer(candidate['text']) is not None for candidate in row['candidates']] for row in bank])
    order=config['order'];changed=rank_view(base,order['strength'])
    corrected=score_features(changed,g,c,e,valid,config,states)
    if order['weight']==1:score=corrected
    else:
        original=score_features(base,g,c,e,valid,config,states)
        score=mixed_score(original,corrected,valid,order['weight'],order['mode'])
    return np.where(valid,score,-1e30).argmax(1).astype(int).tolist()
