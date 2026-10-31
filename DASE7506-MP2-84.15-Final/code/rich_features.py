"""Occurrence-preserving arithmetic paths and soft question-condition features.

All extraction uses the question and candidate text only. Ambiguous quantities
remain alternative paths; their ancestor sets are never blindly unioned.
"""
import ast
from collections import defaultdict, namedtuple
from fractions import Fraction
import itertools
import math
import re
import numpy as np
from common import parse_answer
import quantity_features as old

Node = namedtuple('Node', 'roots unknown depth dimension bad step')
UNITS = {'dollar':'money','dollars':'money','cent':'money','cents':'money',
         'hour':'time','hours':'time','minute':'time','minutes':'time','second':'time','seconds':'time',
         'day':'time','days':'time','week':'time','weeks':'time','year':'time','years':'time',
         'mile':'length','miles':'length','foot':'length','feet':'length','meter':'length','meters':'length',
         'inch':'length','inches':'length','pound':'mass','pounds':'mass','ounce':'mass','ounces':'mass',
         'person':'person','people':'person','student':'person','students':'person','child':'person','children':'person',
         'percent':'ratio','percentage':'ratio','times':'ratio'}
STOP = set('a an the of and or to for per each is are was were has have had more less fewer than as at in on from with by will can must total remaining left new old her his their this that them then costs cost earns earned paid pay spends spent needs wants gets takes makes buys sells'.split())
FRACTIONS = {'half':Fraction(1,2),'halves':Fraction(1,2),'quarter':Fraction(1,4),
             'third':Fraction(1,3),'twice':Fraction(2),'double':Fraction(2),'dozen':Fraction(12)}
FRACTION_RE = re.compile(r'\b(?:half|halves|quarter|third|twice|double|dozen)\b',re.I)
FLAGS = {
 'share':r'\bequal(?:ly)?\b|\bshare|\bdivide|\bdistribut',
 'remaining':r'\bremain|\bleft\b|\bafter\b|\brest\b',
 'difference':r'\bmore than\b|\bless than\b|\bfewer\b|\bdifference\b',
 'total':r'\btotal\b|\baltogether\b|\bin all\b|\bcombined\b',
 'percent':r'%|\bpercent|\bdiscount|\btax\b|\bincrease|\bdecrease',
 'rate':r'\beach\b|\bper\b|\bevery\b|\brate\b',
 'fraction':r'\bhalf|\bthird|\bquarter|\bfraction|\d\s*/\s*\d',
 'at_least':r'\bat least\b|\bminimum\b|\bmust\b|\bneed',
 'at_most':r'\bat most\b|\bmaximum\b|\bcan .*\bbuy\b|\bwhole\b|\bfull\b',
 'conversion':r'\bminutes?\b.*\bhours?\b|\bhours?\b.*\bminutes?\b|\bcents?\b.*\bdollars?\b|\bfeet\b.*\binches\b',
 'money':r'\$|\bdollars?\b|\bcents?\b|\bprofit\b|\bcost\b|\bearn',
 'time':r'\bhours?\b|\bminutes?\b|\bdays?\b|\bweeks?\b',
}


def unit(text,start,end):
    if '$' in text[max(0,start-2):start]:return 'money'
    tail=text[end:end+35].lower().lstrip()
    if tail.startswith('%'):return 'ratio'
    words=re.findall(r'[a-z]+',tail)
    if words and words[0] in {'red','blue','green','white','black','large','small','big','little','extra','additional'}:
        words=words[1:]
    if words and words[0] in UNITS:return UNITS[words[0]]
    if words and words[0] not in STOP and len(words[0])>2:
        return 'item:'+words[0].rstrip('s')
    return 'unknown'


def quantities(question):
    question=question.replace(',','')
    mentions=[]
    for m in old._NUMBER.finditer(question):
        try:
            value=Fraction(m[0].replace(' ',''))
            if abs(value)<=1e12:mentions.append((m.start(),value,unit(question,m.start(),m.end())))
        except (ValueError,ZeroDivisionError):pass
    for m in old._WORD_PATTERN.finditer(question):
        a=old._WORDS[m[1].lower()];b=old._WORDS[m[2].lower()] if m[2] else 0
        values=[a+b] if a>=20 else ([a,b] if m[2] else [a])
        for v in values:mentions.append((m.start(),Fraction(v),unit(question,m.start(),m.end())))
    for m in FRACTION_RE.finditer(question):mentions.append((m.start(),FRACTIONS[m[0].lower()],'ratio'))
    return sorted(mentions,key=lambda r:r[0])


def dim(kind):
    if kind=='unknown':return None
    if kind in ('ratio','scalar'):return ()
    return ((kind,1),)


def dimensional(a,b,op):
    if a is None or b is None:return None,0
    if op in ('+','-'):
        return (a,0) if a==b else (None,1)
    powers=dict(a)
    for key,v in b:powers[key]=powers.get(key,0)+(v if op=='*' else -v)
    return tuple(sorted((k,v) for k,v in powers.items() if v)),0


def template(node,qvalues,leaves):
    value=old._value(node)
    if isinstance(node,(ast.Constant,ast.UnaryOp)) or (isinstance(node,ast.BinOp) and isinstance(node.op,ast.Div)
            and value.denominator!=1 and old._key(value) in qvalues):
        leaves.append(old._key(value));return len(leaves)-1
    if isinstance(node,ast.BinOp):
        op={ast.Add:'+',ast.Sub:'-',ast.Mult:'*',ast.Div:'/'}[type(node.op)]
        return (op,template(node.left,qvalues,leaves),template(node.right,qvalues,leaves))
    raise ValueError('Unsupported arithmetic template')


def evaluate_dimension(tree,nodes):
    if isinstance(tree,int):return nodes[tree].dimension,0
    op,left,right=tree;a,ab=evaluate_dimension(left,nodes);b,bb=evaluate_dimension(right,nodes)
    result,bad=dimensional(a,b,op)
    return result,ab+bb+bad


def requested_unit(question):
    tail=question.lower().rsplit('?',1)[0][-160:]
    m=re.search(r'how (?:many|much)\s+(?:more\s+)?([a-z]+)',tail)
    if m:
        word=m[1]
        if word in UNITS:return dim(UNITS[word])
        if word not in STOP:return dim('item:'+word.rstrip('s'))
    if re.search(r'how much.*(?:cost|pay|earn|money|spend|profit)',tail):return dim('money')
    return None


def graph(question,text,mentions=None):
    mentions=quantities(question) if mentions is None else mentions
    nq=max(1,len(mentions));sources=defaultdict(list)
    for i,(_,value,kind) in enumerate(mentions):
        node=Node(frozenset([i]),frozenset(),0,dim(kind),0,-1)
        sources[old._key(value)].append(node)
        # A percentage may occur as 5 or as 0.05 in the written derivation.
        if kind=='ratio' and abs(value)>1:sources[old._key(value/100)].append(node)
    qvalues=set(sources);computed=defaultdict(list);checked=bad_equations=unit_bad=0;last_op='';last_rhs=None
    normal=old._normalize(text);used=set();unknown_operands=0;operand_count=0;ambiguity=0
    for step,m in enumerate(list(re.finditer('=',normal))[:80]):
        left=old._parse_side(normal[:m.start()],True);right=old._parse_side(normal[m.end():],False)
        if left is None or right is None or not isinstance(left[0],ast.BinOp):continue
        checked+=1
        if not math.isclose(float(left[1]),float(right[1]),rel_tol=1e-5,abs_tol=1e-6):
            bad_equations+=1;continue
        leaves=[]
        try:tree=template(left[0],qvalues,leaves)
        except (ValueError,KeyError):continue
        if len(leaves)>12:continue
        options=[]
        for value in leaves:
            available=computed[value][-4:]+sources[value][:4]
            operand_count+=1
            if len(available)>1:ambiguity+=1
            if not available:
                constant=value in (0,1,2,3,4,7,10,12,24,60,100,1000)
                available=[Node(frozenset(),frozenset() if constant else frozenset([value]),0,() if constant else None,0,-1)]
                unknown_operands+=int(not constant)
            options.append(available)
        result=[];seen=set()
        for chosen in itertools.islice(itertools.product(*options),64):
            roots=frozenset().union(*(n.roots for n in chosen));unknown=frozenset().union(*(n.unknown for n in chosen))
            dimension,bad=evaluate_dimension(tree,chosen)
            signature=(roots,unknown,dimension,bad)
            if signature in seen:continue
            seen.add(signature)
            node=Node(roots,unknown,1+max((n.depth for n in chosen),default=0),dimension,
                      bad+sum(n.bad for n in chosen),step)
            result.append(node);used.update(roots)
        # Preserve distinct paths, including low and high coverage alternatives.
        result.sort(key=lambda n:(len(n.unknown),n.bad,len(n.roots),n.depth))
        if len(result)>12:result=result[:6]+result[-6:]
        value=old._key(right[1]);computed[value].extend(result)
        computed[value]=computed[value][-24:]
        unit_bad+=int(any(n.bad>0 for n in result));last_rhs=value
        last_op={ast.Add:'+',ast.Sub:'-',ast.Mult:'*',ast.Div:'/'}[type(left[0].op)]
    answer=parse_answer(text)
    try:av=old._key(Fraction(answer)) if answer is not None and abs(Fraction(answer))<=1e12 else None
    except (ValueError,OverflowError):av=None
    paths=computed.get(av,[])
    target=requested_unit(question)
    coverage=[len(p.roots)/nq for p in paths];unknown=[len(p.unknown) for p in paths]
    roots=[p.roots for p in paths]
    union=frozenset().union(*roots) if roots else frozenset()
    intersection=set.intersection(*(set(r) for r in roots)) if roots else set()
    matching=[p for p in paths if target is not None and p.dimension==target]
    units={kind for _,_,kind in mentions if kind!='unknown'}
    same_value=defaultdict(list)
    for _,value,kind in mentions:same_value[old._key(value)].append(kind)
    collisions=sum(len(k)>1 for k in same_value.values())
    mixed_collisions=sum(len(set(k)-{'unknown'})>1 for k in same_value.values())
    features=[
        math.log1p(len(mentions)),math.log1p(checked),bad_equations/max(1,checked),
        float(bool(paths)),max(coverage,default=0),min(coverage,default=0),
        len(union)/nq,len(intersection)/nq,(len(union)-len(intersection))/nq,
        max((p.depth for p in paths),default=0),min((p.depth for p in paths),default=0),
        min(unknown,default=0),max(unknown,default=0),len(used)/nq,
        unknown_operands/max(1,operand_count),ambiguity/max(1,operand_count),
        math.log1p(len(paths)),float(last_rhs==av and av is not None),
        unit_bad/max(1,checked),float(bool(matching)),len(matching)/max(1,len(paths)),
        max((len(p.roots)/nq for p in matching),default=0),
        min((p.bad for p in paths),default=0),max((p.bad for p in paths),default=0),
        float(av is not None and abs(av-round(av))<1e-9),
        float(av is not None and av<0),math.log1p(abs(av)) if av is not None else 0.,
        len(units),collisions/nq,mixed_collisions/nq,
        *[float(last_op==op) for op in ('+','-','*','/')],
    ]
    # Question-only flags are crossed with candidate-varying evidence.
    low=question.lower();flags=[float(bool(re.search(pattern,low))) for pattern in FLAGS.values()]
    cue=[features[3],features[4],features[11],features[19],features[24],features[25],features[26],
         features[30],features[31],features[32],features[33]]
    constraints=[flag*value for flag in flags for value in cue]
    return np.asarray(features,dtype=np.float32),np.asarray(constraints,dtype=np.float32),paths,mentions


def extract(bank):
    graphs=[];conditions=[]
    for row in bank:
        mentions=quantities(row['question']);g=[];c=[]
        for candidate in row['candidates']:
            gf,cf,_,_=graph(row['question'],candidate['text'],mentions);g.append(gf);c.append(cf)
        graphs.append(g);conditions.append(c)
    return np.asarray(graphs,dtype=np.float32),np.asarray(conditions,dtype=np.float32)


def augment(graphs,conditions,block,relative):
    raw=graphs if block=='graph' else conditions if block=='conditions' else np.concatenate([graphs,conditions],-1)
    if not relative:return raw.astype(np.float64)
    delta=raw-raw.mean(1,keepdims=True)
    return np.concatenate([raw,delta,delta/(raw.std(1,keepdims=True)+1e-6)],-1).astype(np.float64)
