"""Evaluate tied-rank correction on reconstructed train-only CV models."""
import os
for key in ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS','NUMEXPR_NUM_THREADS'):os.environ[key]='4'
os.environ['CUBLAS_WORKSPACE_CONFIG']=':4096:8'
import hashlib,itertools,json,platform,shutil,time
from pathlib import Path
import numpy as np
import scipy
import torch
import lightgbm as lgb
from training import load,fit,probability,metrics
from baseline import design
from common import load_inputs
from expression_features import augment
from gpu_residual_search import fit_network
from gpu_residual_submission import manual_features
from tree_correction_search import fit_tree,export
from tree_bagging_search import fit_seeded
from tree_correction_submission import tree_input
from order_submission import average_ranks,rank_view,load_states,score_features,mixed_score

CODE=Path(__file__).resolve().parent;OUT=CODE/'runs/order_search';FROZEN=CODE/'runs/order_frozen'
STRENGTHS=(.25,.5,.75,1.)


def main():
    if OUT.exists() or FROZEN.exists():raise FileExistsError(OUT)
    # Average ranks equal the average of all 24 permutation-specific ordinal ranks.
    examples=np.asarray([[1.,1.,1.,1.],[0.,1.,0.,2.],[3.,1.,2.,2.]])
    views=[]
    for perm in itertools.permutations(range(4)):
        p=np.asarray(perm);views.append(np.argsort(np.argsort(examples[:,p],axis=1),axis=1)[:,np.argsort(p)])
    np.testing.assert_allclose(average_ranks(examples),np.mean(views,0),atol=0,rtol=0)
    OUT.mkdir();tick=time.perf_counter()
    (OUT/'predeclared.json').write_text(json.dumps(dict(strengths=STRENGTHS,score_weights=[.25,.5,.75,1.],
        modes=['logit','probability'],new_validation_finalists=5,reference_also_compared=True,
        training_recipe='Same frozen linear/GPU/tree recipes, excluded outer folds',test_references_used=False),indent=2)+'\n')
    data=load('train');n=len(data['y']);base=data['base']
    with np.load(CODE/'runs/correction_cache/train.npz',allow_pickle=False) as z:cache=dict(z)
    e=cache['expression'];ex=np.concatenate([base,augment(e,True)],-1)
    manual=manual_features(base,data['graph'],e);outer=cache['fold_0']
    root=CODE/'runs/tree_bagging_frozen';config=json.loads((root/'config.json').read_text())
    original=np.zeros_like(data['y'],dtype=np.float64);corrected={a:np.zeros_like(original) for a in STRENGTHS}
    for k in np.unique(outer):
        folder=OUT/f'cv_models/fold_{k}';folder.mkdir(parents=True)
        active=(outer!=k)&data['fit_mask'];rows=(outer!=k)&data['hard'];held=outer==k
        for x,name,l2 in zip((base,ex),config['anchor_files'],(.006,.012)):
            state,_=fit(x[active],data['y'][active],l2,1.,1226);np.savez(folder/name,**state)
        teacher=cache[f'teacher_{k}']
        for seed,name in zip((11,23,47),config['reference_members']):
            _,states=fit_network(manual,teacher,data['y'],data['valid'],rows,np.arange(8),
                                 dict(width=32,drop=.3,bound=1.),seed+100*int(k))
            np.savez(folder/name,**states[30])
        raw=tree_input(manual,data['conditions'],teacher,'conditions')
        model=fit_tree(raw,teacher,data['y'],data['valid'],rows,dict(loss='ce',leaves=7))
        np.savez(folder/'original_trees.npz',**export(model,160))
        for seed in (73,101,137):
            model=fit_seeded(raw,teacher,data['y'],data['valid'],rows,dict(leaves=15,min_leaf=100,l2=20.),seed)
            np.savez(folder/f'trees_{seed}.npz',**export(model,160))
        (folder/'config.json').write_text(json.dumps(config,indent=2)+'\n');states=load_states(folder,config)
        b=base[held];g=data['graph'][held];c=data['conditions'][held];ee=e[held];valid=data['valid'][held]
        original[held]=score_features(b,g,c,ee,valid,config,states)
        for a in STRENGTHS:corrected[a][held]=score_features(rank_view(b,a),g,c,ee,valid,config,states)
        print(json.dumps(dict(stage='cv_fold',fold=int(k),seconds=time.perf_counter()-tick)),flush=True)
    np.savez(OUT/'oof-scores.npz',original=original,**{f'a{a}':s for a,s in corrected.items()})
    reference=metrics(probability(original,data['valid']),data);assert reference['disagreement_correct']==1777,reference
    candidates=[]
    for a in STRENGTHS:
        for weight in (.25,.5,.75,1.):
            for mode in ('logit','probability'):
                score=mixed_score(original,corrected[a],data['valid'],weight,mode)
                candidates.append(dict(name=f'a{a}_w{weight}_{mode}',strength=a,weight=weight,mode=mode,
                    cv=metrics(score,data)))
    candidates.sort(key=lambda r:(-r['cv']['disagreement_correct'],-r['cv']['minority_correct'],r['strength']*r['weight'],r['name']))
    (OUT/'cv-models.json').write_text(json.dumps(candidates,indent=2)+'\n');shortlist=candidates[:5]
    (OUT/'shortlist-before-validation.json').write_text(json.dumps(shortlist,indent=2)+'\n')
    bank,arrays=load_inputs('validation');vb=design(bank,arrays)
    with np.load(CODE/'runs/correction_cache/validation.npz',allow_pickle=False) as z:val=dict(z)
    vg=val['manual'][:,:,202:236];vc=val['conditions'];ve=val['expression'];states=load_states(root,config)
    vo=score_features(vb,vg,vc,ve,val['valid'],config,states)
    assert metrics(vo,val)['correct']==755
    views={a:score_features(rank_view(vb,a),vg,vc,ve,val['valid'],config,states) for a in {r['strength'] for r in shortlist}}
    for row in shortlist:
        score=mixed_score(vo,views[row['strength']],val['valid'],row['weight'],row['mode'])
        row['validation']=metrics(score,val)
    shortlist.append(dict(name='reference_392',strength=0.,weight=0.,mode='logit',cv=reference,validation=metrics(vo,val)))
    shortlist.sort(key=lambda r:(-r['validation']['correct'],-r['cv']['disagreement_correct'],r['strength']*r['weight'],r['name']))
    selected=shortlist[0];(OUT/'validation-finalists.json').write_text(json.dumps(shortlist,indent=2)+'\n')
    FROZEN.mkdir()
    for p in root.iterdir():
        if p.name!='config.json':shutil.copyfile(p,FROZEN/p.name)
    config['order']={key:selected[key] for key in ('strength','weight','mode')}
    (FROZEN/'config.json').write_text(json.dumps(config,indent=2)+'\n')
    sources=['order_search.py','order_submission.py','tree_bagging_search.py','tree_correction_search.py','tree_correction_submission.py',
             'gpu_residual_search.py','gpu_residual_submission.py','training.py','expression_features.py','rich_features.py',
             'baseline.py','features.py','quantity_features.py','arith.py']
    freeze=dict(selected=selected,test_evaluated=False,search_seconds=time.perf_counter()-tick,
        environment=dict(python=platform.python_version(),numpy=np.__version__,scipy=scipy.__version__,lightgbm=lgb.__version__,
            torch=torch.__version__,cuda=torch.version.cuda,gpu=torch.cuda.get_device_name(0),threads=4),
        source_hashes={name:hashlib.sha256((CODE/name).read_bytes()).hexdigest() for name in sources},
        asset_hashes={p.name:hashlib.sha256(p.read_bytes()).hexdigest() for p in FROZEN.iterdir()})
    (OUT/'freeze.json').write_text(json.dumps(freeze,indent=2)+'\n');print(json.dumps(selected,indent=2),flush=True)


if __name__=='__main__':main()
