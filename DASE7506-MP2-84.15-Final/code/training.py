"""Train/validation-only helpers for the frozen-recipe experiments."""
import os
for key in ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS'):os.environ[key]='4'
from pathlib import Path
from collections import Counter,defaultdict
import numpy as np
from scipy.optimize import minimize
from scipy.special import logsumexp
from common import DATA,load_inputs,read_jsonl,parse_answer
from baseline import design
from rich_features import extract


def load(split):
    if split not in ('train','validation'):raise ValueError('Development loader excludes test.')
    bank,arrays=load_inputs(split)
    refs=read_jsonl(DATA/f'gsm8k_{split}.jsonl')
    assert len(bank)==len(refs)
    assert all(b['id']==r['id'] and b['question']==r['question'] for b,r in zip(bank,refs))
    y=np.array([[int((a:=parse_answer(c['text'])) is not None and a==parse_answer(r['answer'])) for c in b['candidates']]
                for b,r in zip(bank,refs)])
    answers=[[parse_answer(c['text']) for c in b['candidates']] for b in bank]
    counts=[Counter(a for a in row if a is not None) for row in answers]
    valid=np.array([[a is not None for a in row] for row in answers])
    hard=np.array([len(c)>1 for c in counts])
    minority=np.array([c[next(a for a,correct in zip(row,yy) if correct)]<max(c.values())
                       for c,row,yy in zip(counts,answers,y)])
    graph,conditions=extract(bank)
    return dict(bank=bank,arrays=arrays,y=y,valid=valid,hard=hard,minority=minority,
                fit_mask=y.sum(1)<4,base=design(bank,arrays),graph=graph,conditions=conditions)


def folds(data,seed):
    rng=np.random.RandomState(seed);strata=defaultdict(list)
    for i in range(len(data['y'])):strata[(int(data['y'][i].sum()),bool(data['hard'][i]))].append(i)
    result=np.zeros(len(data['y']),dtype=int)
    for values in strata.values():
        values=np.array(values);rng.shuffle(values)
        result[values]=np.arange(len(values))%3
    return result


def fit(x,y,l2=.006,penalty_ratio=1.,base_dim=1226):
    n,k,d=x.shape
    mu=x.reshape(-1,d).mean(0);sd=x.reshape(-1,d).std(0)+1e-8
    z=(x-mu)/sd;target=y/y.sum(1,keepdims=True)
    penalty=np.full(d,l2);penalty[base_dim:]*=penalty_ratio
    def objective(w):
        scores=z@w;p=np.exp(scores-logsumexp(scores,axis=1,keepdims=True))
        loss=np.mean(logsumexp(scores,axis=1)-(target*scores).sum(1))+float((penalty*w)@w)
        gradient=np.einsum('nk,nkf->f',(p-target)/n,z)+2*penalty*w
        return loss,gradient
    result=minimize(objective,np.zeros(d),jac=True,method='L-BFGS-B',options={'maxiter':350})
    return dict(W=result.x,b=np.array([0.]),mu=mu,sd=sd),dict(success=bool(result.success),iterations=int(result.nit),loss=float(result.fun))


def logits(x,state):return ((x-state['mu'])/state['sd'])@state['W']+state['b'][0]


def probability(scores,valid):
    s=np.where(valid,scores,-1e30);p=np.exp(s-s.max(1,keepdims=True))*valid
    return p/np.maximum(p.sum(1,keepdims=True),1e-30)


def metrics(prob,data):
    choice=np.where(data['valid'],prob,-1e30).argmax(1)
    correct=data['y'][np.arange(len(choice)),choice].astype(bool)
    return dict(correct=int(correct.sum()),n=len(choice),disagreement_correct=int(correct[data['hard']].sum()),
                disagreement_n=int(data['hard'].sum()),minority_correct=int(correct[data['minority']].sum()),
                minority_n=int(data['minority'].sum()))
