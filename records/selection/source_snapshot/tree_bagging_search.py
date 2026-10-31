"""Stabilise shallow residual trees with three training seeds, train/val only."""
import os
for key in ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS','NUMEXPR_NUM_THREADS'):os.environ[key]='4'
import hashlib,json,platform,shutil,time
from pathlib import Path
import numpy as np
import lightgbm as lgb
from training import probability,metrics
from tree_correction_search import export
from tree_correction_submission import tree_input,tree_forward

CODE=Path(__file__).resolve().parent;OUT=CODE/'runs/tree_bagging_search';FROZEN=CODE/'runs/tree_bagging_frozen'
SEEDS=(73,101,137);ROUNDS=(20,60,160)


def fit_seeded(x,anchor,y,valid,rows,spec,seed):
    target=y[rows]/y[rows].sum(1,keepdims=True);vv=valid[rows]
    first=[True]
    def objective(pred,dataset):
        score=pred.reshape(-1,4)
        if first[0]:np.testing.assert_allclose(score,anchor[rows],atol=1e-12,rtol=1e-12);first[0]=False
        p=probability(score,vv)
        return (p-target).ravel(),np.maximum(p*(1-p),1e-6).ravel()
    params=dict(objective=objective,num_leaves=spec['leaves'],min_data_in_leaf=spec['min_leaf'],lambda_l2=spec['l2'],
        learning_rate=.03,feature_fraction=.8,verbosity=-1,num_threads=4,seed=seed,deterministic=True,force_col_wise=True,
        max_bin=63,min_sum_hessian_in_leaf=1.,boost_from_average=False)
    dataset=lgb.Dataset(x[rows].reshape(-1,x.shape[-1]),label=y[rows].ravel(),init_score=anchor[rows].ravel(),free_raw_data=True)
    return lgb.train(params,dataset,num_boost_round=max(ROUNDS))


def main():
    if OUT.exists() or FROZEN.exists():raise FileExistsError(OUT)
    OUT.mkdir();tick=time.perf_counter()
    specs=[dict(name=f'l{leaves}_m{minimum}_r{l2}',leaves=leaves,min_leaf=minimum,l2=l2,block='conditions')
           for leaves,minimum,l2 in [(7,60,10.),(15,60,10.),(7,40,5.),(7,100,20.),(15,100,20.),(7,60,30.)]]
    (OUT/'predeclared.json').write_text(json.dumps(dict(specs=specs,seeds=SEEDS,rounds=ROUNDS,weights=[.25,.5,1.],
        beta=[.25,.5,.75,1.],new_validation_finalists=5,reference_also_compared=True,test_references_used=False),indent=2)+'\n')
    with np.load(CODE/'runs/correction_cache/train.npz',allow_pickle=False) as z:train=dict(z)
    with np.load(CODE/'runs/correction_cache/validation.npz',allow_pickle=False) as z:val=dict(z)
    with np.load(CODE/'runs/tree_correction_search/oof.npz',allow_pickle=False) as z:old=z['conditions_ce_l7_160'].copy()
    old_cv=.0625*train['reference_oof']+.1875*old
    with np.load(CODE/'runs/tree_correction_frozen/trees.npz',allow_pickle=False) as z:old_state=dict(z)
    vx=tree_input(val['manual'],val['conditions'],val['anchor'],'conditions')
    old_val=.0625*val['reference']+.1875*tree_forward(vx,old_state)
    outer=train['fold_0'];candidates=[];oof={}
    for spec in specs:
        scores={r:np.zeros_like(train['y'],dtype=np.float64) for r in ROUNDS}
        for k in np.unique(outer):
            anchor=train[f'teacher_{k}'];raw=tree_input(train['manual'],train['conditions'],anchor,'conditions')
            rows=(outer!=k)&train['hard'];held=outer==k
            for seed in SEEDS:
                model=fit_seeded(raw,anchor,train['y'],train['valid'],rows,spec,seed)
                for r in ROUNDS:
                    state=export(model,r);actual=tree_forward(raw[held],state)
                    expected=model.predict(raw[held].reshape(-1,raw.shape[-1]),num_iteration=r,raw_score=True).reshape(-1,4)
                    np.testing.assert_allclose(actual,expected,atol=1e-10,rtol=1e-10);scores[r][held]+=actual/len(SEEDS)
        for r,residual in scores.items():
            oof[f"{spec['name']}_{r}"]=residual
            for weight in (.25,.5,1.):
                for beta in (.25,.5,.75,1.):
                    score=train['anchor_oof'][0]+(1-beta)*old_cv+beta*weight*residual
                    candidates.append(dict(name=f"{spec['name']}_r{r}_w{weight}_b{beta}",spec=spec,rounds=r,
                        old_weight=1-beta,new_weight=weight*beta,cv=metrics(probability(score,train['valid']),train)))
        print(json.dumps(dict(stage='bagged_cv',spec=spec,seconds=time.perf_counter()-tick)),flush=True)
    candidates.sort(key=lambda r:(-r['cv']['disagreement_correct'],-r['cv']['minority_correct'],r['rounds'],r['name']))
    (OUT/'cv-models.json').write_text(json.dumps(candidates,indent=2)+'\n');np.savez(OUT/'oof.npz',**oof)
    shortlist=candidates[:5];(OUT/'shortlist-before-validation.json').write_text(json.dumps(shortlist,indent=2)+'\n')
    models={};states={};raw=tree_input(train['manual'],train['conditions'],train['train_anchor'],'conditions')
    for row in shortlist:
        spec=row['spec'];name=spec['name'];key=(name,row['rounds'])
        if name not in models:models[name]=[fit_seeded(raw,train['train_anchor'],train['y'],train['valid'],train['hard'],spec,seed) for seed in SEEDS]
        if key not in states:
            exported=[export(model,row['rounds']) for model in models[name]]
            prediction=np.mean([tree_forward(vx,state) for state in exported],0)
            expected=np.mean([model.predict(vx.reshape(-1,vx.shape[-1]),num_iteration=row['rounds'],raw_score=True).reshape(-1,4) for model in models[name]],0)
            np.testing.assert_allclose(prediction,expected,atol=1e-10,rtol=1e-10);states[key]=(exported,prediction)
        score=val['anchor']+row['old_weight']*old_val+row['new_weight']*states[key][1]
        row['validation']=metrics(probability(score,val['valid']),val)
    shortlist.append(dict(name='reference_392',old_weight=1.,new_weight=0.,rounds=0,
        cv=metrics(probability(train['anchor_oof'][0]+old_cv,train['valid']),train),
        validation=metrics(probability(val['anchor']+old_val,val['valid']),val)))
    shortlist.sort(key=lambda r:(-r['validation']['correct'],-r['cv']['disagreement_correct'],r['rounds'],r['name']))
    selected=shortlist[0];(OUT/'validation-finalists.json').write_text(json.dumps(shortlist,indent=2)+'\n')
    FROZEN.mkdir();root=CODE/'runs/gpu_residual_frozen';names=['anchor_base.npz','anchor_expression.npz']
    rw=.0625*selected['old_weight'];members=['residual_11.npz','residual_23.npz','residual_47.npz'] if rw else []
    for name in [*names,*members]:shutil.copyfile(root/name,FROZEN/name)
    parts=[]
    if selected['old_weight']:
        np.savez(FROZEN/'original_trees.npz',**old_state);parts.append(dict(file='original_trees.npz',block='conditions',weight=.1875*selected['old_weight']))
    if selected['new_weight']:
        for seed,state in zip(SEEDS,states[(selected['spec']['name'],selected['rounds'])][0]):
            name=f'trees_{seed}.npz';np.savez(FROZEN/name,**state)
            parts.append(dict(file=name,block='conditions',weight=selected['new_weight']/len(SEEDS)))
    (FROZEN/'config.json').write_text(json.dumps(dict(anchor_files=names,reference_members=members,reference_weight=rw,trees=parts),indent=2)+'\n')
    sources=['tree_bagging_search.py','tree_correction_search.py','tree_correction_submission.py','gpu_residual_submission.py',
             'expression_features.py','rich_features.py','baseline.py','features.py','quantity_features.py','arith.py','training.py']
    freeze=dict(selected=selected,test_evaluated=False,search_seconds=time.perf_counter()-tick,
        environment=dict(python=platform.python_version(),numpy=np.__version__,lightgbm=lgb.__version__,threads=4),
        source_hashes={name:hashlib.sha256((CODE/name).read_bytes()).hexdigest() for name in sources},
        asset_hashes={p.name:hashlib.sha256(p.read_bytes()).hexdigest() for p in FROZEN.iterdir()})
    (OUT/'freeze.json').write_text(json.dumps(freeze,indent=2)+'\n');print(json.dumps(selected,indent=2),flush=True)


if __name__=='__main__':main()
