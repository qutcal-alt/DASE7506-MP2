"""Bounded question-only arithmetic search, scored against supplied final answers.

Each question quantity is an occurrence, usable at most once in a program. This
is a soft feasibility feature, not a semantic solver or an answer-generation API.
"""
import math
import re
from fractions import Fraction
import numpy as np
from common import parse_answer
from rich_features import quantities

DIM=22
LIMIT=128


def value_key(value):return round(float(value),6)


def program_tables(question,targets):
    mentions=quantities(question)
    n=len(mentions)
    if not 1<=n<=5:return None,n
    question=question.replace(',','').lower();table={}
    for i,(position,number,kind) in enumerate(mentions):
        value=float(number);aliases={value:True};tail=question[position:position+35]
        if kind=='ratio' and abs(value)>1:aliases[value/100]=False
        if re.search(r'\bcents?\b',tail):aliases[value/100]=False
        if re.search(r'\bminutes?\b',tail):aliases[value/60]=False
        table[1<<i]={value_key(v):np.array([1.,float(raw),0.,0.,0.,0.]) for v,raw in aliases.items() if abs(v)<=1e9}
    target_values=[v for v in targets if v is not None and math.isfinite(v)]
    for size in range(2,n+1):
        for mask in range(1,1<<n):
            if mask.bit_count()!=size:continue
            results={};sub=(mask-1)&mask
            while sub:
                other=mask^sub
                if other and sub<other:
                    for a,ca in table[sub].items():
                        for b,cb in table[other].items():
                            total=ca[0]*cb[0];raw=ca[1]*cb[1]
                            choices=[(a+b,2),(a-b,3),(b-a,3),(a*b,4)]
                            if abs(b)>1e-9:choices.append((a/b,5))
                            if abs(a)>1e-9:choices.append((b/a,5))
                            for v,op in choices:
                                if not math.isfinite(v) or abs(v)>1e9:continue
                                key=value_key(v)
                                if key not in results:results[key]=np.zeros(6)
                                record=results[key];record[0]+=total;record[1]+=raw;record[op]+=total
                sub=(sub-1)&mask
            if len(results)>LIMIT:
                common=sorted(results,key=lambda v:(-results[v][0],abs(v)))[:LIMIT//2]
                nearest=sorted(results,key=lambda v:min((abs(v-t)/(1+abs(t)) for t in target_values),default=abs(v)))[:LIMIT//2]
                selected=list(dict.fromkeys(common+nearest))
                results={v:results[v] for v in selected}
            table[mask]=results
    return table,n


def row_features(question,texts):
    answers=[]
    for text in texts:
        a=parse_answer(text)
        try:answers.append(float(Fraction(a)) if a is not None else None)
        except (ValueError,OverflowError):answers.append(None)
    tables,n=program_tables(question,answers)
    if tables is None:return np.zeros((4,DIM),dtype=np.float32)
    full=(1<<n)-1;all_values=tables[full];den=max(1,sum(v[0] for v in all_values.values()))
    flags=[float(bool(re.search(pattern,question.lower()))) for pattern in
           (r'\bremain|\bleft\b|\bafter\b',r'\beach\b|\bper\b|\bevery\b',
            r'\bequal|\bshare|\bdivide',r'\bat least\b|\bminimum\b|\bneed',
            r'\bwhole\b|\bfull\b|\bat most\b|\bcan .*buy')]
    rows=[]
    for answer in answers:
        if answer is None or abs(answer)>1e9:rows.append([0.]*DIM);continue
        key=value_key(answer);support=[mask.bit_count() for mask,states in tables.items() if key in states]
        r=all_values.get(key,np.zeros(6));p=r[0]/den
        floor_support=ceil_support=False
        if abs(answer-round(answer))<1e-9:
            floor_support=any(math.floor(v+1e-5)==round(answer) for v in all_values)
            ceil_support=any(math.ceil(v-1e-5)==round(answer) for v in all_values)
        features=[1.,float(bool(support)),float(key in all_values),
            min(support,default=0)/n,max(support,default=0)/n,
            len(support)/max(1,(1<<n)-1),math.log1p(r[0]),p,
            r[1]/max(1,r[0]),*[r[i]/max(1,r[0]) for i in (2,3,4,5)],
            float(floor_support),float(ceil_support),
            flags[0]*r[3]/max(1,r[0]),flags[1]*r[4]/max(1,r[0]),flags[2]*r[5]/max(1,r[0]),
            flags[3]*float(ceil_support),flags[4]*float(floor_support),
            float(floor_support and not key in all_values),float(ceil_support and not key in all_values)]
        assert len(features)==DIM
        rows.append(features)
    return np.asarray(rows,dtype=np.float32)


def extract(bank,progress=None):
    rows=[]
    for i,row in enumerate(bank):
        rows.append(row_features(row['question'],[c['text'] for c in row['candidates']]))
        if progress is not None and (i+1)%500==0:progress(i+1)
    return np.asarray(rows,dtype=np.float32)


def augment(raw,relative):
    if not relative:return raw.astype(np.float64)
    delta=raw-raw.mean(1,keepdims=True)
    return np.concatenate([raw,delta,delta/(raw.std(1,keepdims=True)+1e-6)],-1).astype(np.float64)
