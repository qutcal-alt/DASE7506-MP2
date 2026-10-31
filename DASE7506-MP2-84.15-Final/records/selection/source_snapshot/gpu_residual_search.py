"""GPU residual-network training; train-only nested base predictions and validation."""
import os
for key in ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS'):os.environ[key]='4'
os.environ['CUBLAS_WORKSPACE_CONFIG']=':4096:8'
import hashlib,json,platform,shutil,time
from pathlib import Path
import numpy as np
import scipy
import torch
from torch import nn
from training import load,fit,logits,probability,metrics
from expression_features import extract as expression_extract,augment
from gpu_residual_submission import manual_features,network_input,forward,anchor_scores

CODE=Path(__file__).resolve().parent;OUT=CODE/'runs/gpu_residual_search';FROZEN=CODE/'runs/gpu_residual_frozen'
DEVICE=torch.device('cuda:0')
torch.set_num_threads(4);torch.set_num_interop_threads(1)
torch.backends.cuda.matmul.allow_tf32=False
torch.use_deterministic_algorithms(True)


class Residual(nn.Module):
    def __init__(self,d,width,drop,bound):
        super().__init__();self.enc=nn.Linear(d,width);self.drop=nn.Dropout(drop);self.out=nn.Linear(width,1);self.bound=bound
        nn.init.zeros_(self.out.weight);nn.init.zeros_(self.out.bias)
    def forward(self,x):return self.bound*torch.tanh(self.out(self.drop(torch.relu(self.enc(x))))[...,0])


def fit_network(manual,anchor,y,valid,rows,held,spec,seed):
    torch.manual_seed(seed);torch.cuda.manual_seed_all(seed)
    raw=network_input(manual,anchor);mu=raw[rows].mean((0,1));sd=raw[rows].std((0,1)).clip(.01)
    z=np.clip((raw-mu)/sd,-8,8).astype(np.float32)
    xt=torch.from_numpy(z).to(DEVICE);at=torch.from_numpy(anchor.astype(np.float32)).to(DEVICE)
    yt=torch.from_numpy(y.astype(np.float32)).to(DEVICE);vt=torch.from_numpy(valid).to(DEVICE)
    index=torch.from_numpy(np.flatnonzero(rows)).to(DEVICE)
    held_index=torch.from_numpy(np.flatnonzero(held) if held.dtype==bool else held).to(DEVICE)
    model=Residual(raw.shape[-1],spec['width'],spec['drop'],spec['bound']).to(DEVICE)
    opt=torch.optim.AdamW(model.parameters(),lr=.001,weight_decay=.05)
    snaps={};states={}
    for epoch in range(1,61):
        model.train()
        for batch in index[torch.randperm(len(index),device=DEVICE)].split(128):
            residual=model(xt[batch]);score=(at[batch]+residual).masked_fill(~vt[batch],-1e9)
            target=yt[batch]/yt[batch].sum(1,keepdim=True)
            loss=-(target*torch.log_softmax(score,1)).sum(1).mean()+.02*residual.square().mean()
            opt.zero_grad();loss.backward();nn.utils.clip_grad_norm_(model.parameters(),5);opt.step()
        if epoch in (10,30,60):
            model.eval()
            with torch.no_grad():cuda_out=model(xt[held_index]).cpu().numpy()
            state={k:v.detach().cpu().numpy().copy() for k,v in model.state_dict().items()};state.update(mu=mu,sd=sd)
            cpu_out=forward(raw[held],state,spec['bound'])
            np.testing.assert_allclose(cpu_out,cuda_out,atol=3e-4,rtol=3e-4)
            snaps[epoch]=cpu_out;states[epoch]=state
    return snaps,states


def main():
    if not torch.cuda.is_available():raise RuntimeError('CUDA GPU is required for this experiment.')
    if FROZEN.exists():raise FileExistsError(FROZEN)
    OUT.mkdir(parents=True,exist_ok=True);tick=time.perf_counter()
    specs=[dict(name='w32_b05',width=32,drop=.3,bound=.5),dict(name='w64_b05',width=64,drop=.4,bound=.5),
           dict(name='w32_b10',width=32,drop=.3,bound=1.),dict(name='w64_b10',width=64,drop=.4,bound=1.)]
    (OUT/'predeclared.json').write_text(json.dumps(dict(specs=specs,epochs=[10,30,60],cv_seeds=[11,23,47],
        full_seeds=[11,23,47],validation_finalists=5,device=torch.cuda.get_device_name(0),
        residual_weights=[.25,.5,1.],test_references_used=False),indent=2)+'\n')
    train=load('train');n=len(train['y'])
    expression_freeze=json.loads((CODE/'runs/expression_search/freeze.json').read_text())
    assert hashlib.sha256((CODE/'expression_features.py').read_bytes()).hexdigest()==expression_freeze['source_hashes']['expression_features.py']
    with np.load(CODE/'runs/expression_search/train-input-features.npz',allow_pickle=False) as z:e=z['features'].copy()
    assert e.shape[:2]==train['y'].shape
    manual=manual_features(train['base'],train['graph'],e)
    ex=np.concatenate([train['base'],augment(e,True)],-1)
    with np.load(CODE/'runs/search/folds.npz',allow_pickle=False) as z:folds=[z['fold_0'],z['fold_1']]
    with np.load(CODE/'runs/search/oof-probabilities.npz',allow_pickle=False) as z:base_oof=z['base'].copy()
    expression_oof=[]
    for fold in folds:
        s=np.zeros_like(train['y'],dtype=np.float64)
        for k in np.unique(fold):
            active=(fold!=k)&train['fit_mask'];held=fold==k
            state,_=fit(ex[active],train['y'][active],.012,1.,train['base'].shape[-1])
            s[held]=logits(ex[held],state)
        expression_oof.append(probability(s,train['valid']))
    anchor_oof=np.log(np.maximum(.75*base_oof+.25*np.stack(expression_oof),1e-30))
    outer=folds[0];teacher={}
    for k in np.unique(outer):
        outer_train=outer!=k;inner=np.random.RandomState(930+int(k)).randint(0,2,n)
        pp=[np.zeros_like(train['y'],dtype=np.float64) for _ in range(2)]
        for j in (0,1):
            active=outer_train&(inner==j)&train['fit_mask'];scoring=outer_train&(inner!=j)
            for index,x in enumerate([train['base'],ex]):
                state,_=fit(x[active],train['y'][active],.006 if index==0 else .012,1.,train['base'].shape[-1])
                pp[index][scoring]=probability(logits(x[scoring],state),train['valid'][scoring])
        anchor=np.log(np.maximum(.75*pp[0]+.25*pp[1],1e-30));anchor[~outer_train]=anchor_oof[0][~outer_train]
        teacher[int(k)]=anchor
        print(json.dumps(dict(stage='nested_anchor',fold=int(k),elapsed=time.perf_counter()-tick)),flush=True)
    records=[];candidates=[]
    for spec in specs:
        sums={epoch:np.zeros_like(train['y'],dtype=np.float64) for epoch in (10,30,60)}
        for seed_index,seed in enumerate((11,23,47)):
            for k in np.unique(outer):
                rows=(outer!=k)&train['hard'];held=outer==k
                snaps,_=fit_network(manual,teacher[int(k)],train['y'],train['valid'],rows,held,spec,seed+100*int(k))
                for epoch in sums:sums[epoch][held]+=snaps[epoch]/3
                print(json.dumps(dict(stage='cuda_fit',spec=spec['name'],seed=seed,fold=int(k),elapsed=time.perf_counter()-tick,
                    gpu_memory_MiB=torch.cuda.max_memory_allocated()/2**20)),flush=True)
        for epoch,residual in sums.items():
            for weight in (.25,.5,1.):
                score=anchor_oof[0]+weight*residual
                row=dict(name=spec['name']+f'_e{epoch}_r{weight}',spec=spec,epoch=epoch,residual_weight=weight,
                    cv=metrics(probability(score,train['valid']),train));candidates.append(row)
        records.extend([r for r in candidates if r['spec']['name']==spec['name']])
        (OUT/'cv-models.json').write_text(json.dumps(records,indent=2)+'\n')
    candidates.sort(key=lambda r:(-r['cv']['disagreement_correct'],-r['cv']['minority_correct'],r['spec']['width'],r['name']))
    shortlist=candidates[:5];(OUT/'shortlist-before-validation.json').write_text(json.dumps(shortlist,indent=2)+'\n')
    print(json.dumps(dict(stage='shortlist',candidates=shortlist)),flush=True)
    val=load('validation');ve=expression_extract(val['bank']);vmanual=manual_features(val['base'],val['graph'],ve)
    anchor_paths=[CODE/'runs/expression_frozen/base.npz',CODE/'runs/expression_frozen/expression.npz'];astates=[]
    for p in anchor_paths:
        with np.load(p,allow_pickle=False) as z:astates.append(dict(z))
    va=anchor_scores(val['base'],ve,val['valid'],astates);vraw=network_input(vmanual,va)
    train_anchor=np.log(np.maximum((.75*base_oof+.25*np.stack(expression_oof)).mean(0),1e-30))
    trained={};vpred={}
    for name in {r['spec']['name'] for r in shortlist}:
        spec=next(s for s in specs if s['name']==name);member_states=[];member_residuals={epoch:[] for epoch in (10,30,60)}
        for seed in (11,23,47):
            _,states=fit_network(manual,train_anchor,train['y'],train['valid'],train['hard'],np.arange(8),spec,seed)
            member_states.append(states)
            for epoch in member_residuals:member_residuals[epoch].append(forward(vraw,states[epoch],spec['bound']))
        trained[name]=member_states
        for epoch in member_residuals:vpred[(name,epoch)]=np.mean(member_residuals[epoch],0)
    for row in shortlist:
        score=va+row['residual_weight']*vpred[(row['spec']['name'],row['epoch'])]
        row['validation']=metrics(probability(score,val['valid']),val)
    shortlist.sort(key=lambda r:(-r['validation']['correct'],-r['cv']['disagreement_correct'],r['spec']['width'],r['name']))
    selected=shortlist[0];(OUT/'validation-finalists.json').write_text(json.dumps(shortlist,indent=2)+'\n')
    FROZEN.mkdir(parents=True)
    for name,path in zip(['anchor_base.npz','anchor_expression.npz'],anchor_paths):shutil.copyfile(path,FROZEN/name)
    members=[]
    for seed,states in zip((11,23,47),trained[selected['spec']['name']]):
        name=f'residual_{seed}.npz';np.savez(FROZEN/name,**states[selected['epoch']]);members.append(name)
    config=dict(anchor_files=['anchor_base.npz','anchor_expression.npz'],members=members,
                bound=selected['spec']['bound'],residual_weight=selected['residual_weight'])
    (FROZEN/'config.json').write_text(json.dumps(config,indent=2)+'\n')
    sources=['gpu_residual_submission.py','gpu_residual_search.py','expression_features.py','rich_features.py','baseline.py',
             'features.py','quantity_features.py','arith.py','training.py']
    freeze=dict(selected=selected,test_evaluated=False,search_seconds=time.perf_counter()-tick,
        environment=dict(python=platform.python_version(),numpy=np.__version__,scipy=scipy.__version__,torch=torch.__version__,
            cuda=torch.version.cuda,gpu=torch.cuda.get_device_name(0),peak_gpu_memory_MiB=torch.cuda.max_memory_allocated()/2**20),
        source_hashes={name:hashlib.sha256((CODE/name).read_bytes()).hexdigest() for name in sources},
        asset_hashes={p.name:hashlib.sha256(p.read_bytes()).hexdigest() for p in FROZEN.iterdir()})
    (OUT/'freeze.json').write_text(json.dumps(freeze,indent=2)+'\n')
    print(json.dumps(dict(stage='frozen',**freeze)),flush=True)


if __name__=='__main__':main()
