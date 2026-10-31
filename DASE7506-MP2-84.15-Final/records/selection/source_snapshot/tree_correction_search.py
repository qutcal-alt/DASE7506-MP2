"""Nested train-only tree residuals, selected on validation before official test.

LightGBM custom objective and export interface:
https://lightgbm.readthedocs.io/en/v4.6.0/pythonapi/lightgbm.train.html
"""
import os
for key in ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS','NUMEXPR_NUM_THREADS'):os.environ[key]='4'
import hashlib,json,platform,shutil,time
from pathlib import Path
import numpy as np
import lightgbm as lgb
from training import metrics,probability
from tree_correction_submission import tree_input,tree_forward

CODE=Path(__file__).resolve().parent;CACHE=CODE/'runs/correction_cache'
OUT=CODE/'runs/tree_correction_search';FROZEN=CODE/'runs/tree_correction_frozen'
ROUNDS=(20,60,160)


def export(model,iterations):
    feature=[];threshold=[];left=[];right=[];value=[];roots=[];depths=[]
    def visit(node,depth):
        index=len(feature);feature.append(-1);threshold.append(0.);left.append(-1);right.append(-1)
        value.append(node.get('leaf_value',0.));depths.append(depth)
        if 'split_feature' in node:
            assert node['decision_type']=='<=' and not node.get('missing_type')=='NaN'
            feature[index]=node['split_feature'];threshold[index]=node['threshold']
            left[index]=visit(node['left_child'],depth+1);right[index]=visit(node['right_child'],depth+1)
        return index
    for tree in model.dump_model(num_iteration=iterations)['tree_info']:roots.append(visit(tree['tree_structure'],0))
    return dict(feature=np.asarray(feature,np.int32),threshold=np.asarray(threshold,np.float64),left=np.asarray(left,np.int32),
        right=np.asarray(right,np.int32),value=np.asarray(value,np.float64),roots=np.asarray(roots,np.int32),
        max_depth=np.array(max(depths,default=0),np.int32))


def fit_tree(x,anchor,y,valid,rows,spec):
    yy=y[rows].astype(np.float64);vv=valid[rows];target=yy/yy.sum(1,keepdims=True)
    first=[True]
    def objective(predictions,dataset):
        score=predictions.reshape(-1,4)
        if first[0]:
            np.testing.assert_allclose(score,anchor[rows],atol=1e-12,rtol=1e-12);first[0]=False
        if spec['loss']=='ce':
            p=probability(score,vv);gradient=p-target;hessian=np.maximum(p*(1-p),1e-6)
        else:
            pairs=yy[:,:,None]*(1-yy[:,None,:])*vv[:,None,:];pairs/=np.maximum(pairs.sum((1,2),keepdims=True),1)
            diff=score[:,:,None]-score[:,None,:];q=1/(1+np.exp(np.clip(diff,-40,40)))
            z=pairs*q;gradient=-z.sum(2)+z.sum(1)
            z=pairs*q*(1-q);hessian=np.maximum(z.sum(2)+z.sum(1),1e-6)
        return gradient.ravel(),hessian.ravel()
    params=dict(objective=objective,num_leaves=spec['leaves'],learning_rate=.03,min_data_in_leaf=60,
        lambda_l2=10.,feature_fraction=.8,verbosity=-1,num_threads=4,seed=73,deterministic=True,force_col_wise=True,
        max_bin=63,min_sum_hessian_in_leaf=1.,boost_from_average=False)
    dataset=lgb.Dataset(x[rows].reshape(-1,x.shape[-1]),label=y[rows].ravel(),init_score=anchor[rows].ravel(),free_raw_data=True)
    return lgb.train(params,dataset,num_boost_round=max(ROUNDS))


def main():
    if FROZEN.exists() or OUT.exists():raise FileExistsError('Use a fresh research output directory.')
    OUT.mkdir();tick=time.perf_counter()
    manifest=json.loads((CACHE/'manifest.json').read_text())
    for name,h in manifest['files'].items():assert hashlib.sha256((CACHE/name).read_bytes()).hexdigest()==h
    with np.load(CACHE/'train.npz',allow_pickle=False) as z:train=dict(z)
    with np.load(CACHE/'validation.npz',allow_pickle=False) as z:val=dict(z)
    specs=[dict(name=f'{block}_{loss}_l{leaves}',block=block,loss=loss,leaves=leaves)
           for block in ('manual','conditions','relative') for loss in ('ce','pair') for leaves in (7,15)]
    (OUT/'predeclared.json').write_text(json.dumps(dict(specs=specs,rounds=ROUNDS,tree_weights=[.25,.5,1.],
        beta=[.25,.5,.75,1.],new_validation_finalists=5,reference_also_compared=True,test_references_used=False),indent=2)+'\n')
    outer=train['fold_0'];candidates=[];oof={}
    for spec in specs:
        scores={r:np.zeros_like(train['y'],dtype=np.float64) for r in ROUNDS}
        for k in np.unique(outer):
            anchor=train[f'teacher_{k}'];x=tree_input(train['manual'],train['conditions'],anchor,spec['block'])
            rows=(outer!=k)&train['hard'];held=outer==k
            model=fit_tree(x,anchor,train['y'],train['valid'],rows,spec)
            for r in ROUNDS:
                state=export(model,r);actual=tree_forward(x[held],state)
                expected=model.predict(x[held].reshape(-1,x.shape[-1]),num_iteration=r,raw_score=True).reshape(-1,4)
                np.testing.assert_allclose(actual,expected,rtol=1e-10,atol=1e-10)
                scores[r][held]=actual
        for r,residual in scores.items():
            oof[f"{spec['name']}_{r}"]=residual
            for weight in (.25,.5,1.):
                for beta in (.25,.5,.75,1.):
                    reference_weight=.25*(1-beta);tree_weight=weight*beta
                    p=probability(train['anchor_oof'][0]+reference_weight*train['reference_oof']+tree_weight*residual,train['valid'])
                    candidates.append(dict(name=f"{spec['name']}_r{r}_w{weight}_b{beta}",spec=spec,rounds=r,
                        reference_weight=reference_weight,tree_weight=tree_weight,cv=metrics(p,train)))
        print(json.dumps(dict(stage='tree_cv',spec=spec,seconds=time.perf_counter()-tick)),flush=True)
    candidates.sort(key=lambda r:(-r['cv']['disagreement_correct'],-r['cv']['minority_correct'],r['rounds'],r['name']))
    (OUT/'cv-models.json').write_text(json.dumps(candidates,indent=2)+'\n');np.savez(OUT/'oof.npz',**oof)
    shortlist=candidates[:5];(OUT/'shortlist-before-validation.json').write_text(json.dumps(shortlist,indent=2)+'\n')
    models={};states={}
    for row in shortlist:
        spec=row['spec'];name=spec['name']
        if name not in models:
            x=tree_input(train['manual'],train['conditions'],train['train_anchor'],spec['block'])
            models[name]=fit_tree(x,train['train_anchor'],train['y'],train['valid'],train['hard'],spec)
        key=(name,row['rounds'])
        if key not in states:
            state=export(models[name],row['rounds']);vx=tree_input(val['manual'],val['conditions'],val['anchor'],spec['block'])
            actual=tree_forward(vx,state)
            expected=models[name].predict(vx.reshape(-1,vx.shape[-1]),num_iteration=row['rounds'],raw_score=True).reshape(-1,4)
            np.testing.assert_allclose(actual,expected,rtol=1e-10,atol=1e-10);states[key]=(state,actual)
        residual=states[key][1]
        p=probability(val['anchor']+row['reference_weight']*val['reference']+row['tree_weight']*residual,val['valid'])
        row['validation']=metrics(p,val)
    baseline=dict(name='reference_391',reference_weight=.25,tree_weight=0.,rounds=0,
        cv=manifest['reference_oof'],validation=manifest['reference_validation'])
    shortlist.append(baseline)
    shortlist.sort(key=lambda r:(-r['validation']['correct'],-r['cv']['disagreement_correct'],r['rounds'],r['name']))
    (OUT/'validation-finalists.json').write_text(json.dumps(shortlist,indent=2)+'\n');selected=shortlist[0]
    FROZEN.mkdir();reference=CODE/'runs/gpu_residual_frozen'
    names=['anchor_base.npz','anchor_expression.npz']
    reference_members=['residual_11.npz','residual_23.npz','residual_47.npz'] if selected['reference_weight'] else []
    for name in [*names,*reference_members]:shutil.copyfile(reference/name,FROZEN/name)
    config=dict(anchor_files=names,reference_members=reference_members,reference_weight=selected['reference_weight'],trees=[])
    if selected['tree_weight']:
        state=states[(selected['spec']['name'],selected['rounds'])][0];np.savez(FROZEN/'trees.npz',**state)
        config['trees']=[dict(file='trees.npz',block=selected['spec']['block'],weight=selected['tree_weight'])]
    (FROZEN/'config.json').write_text(json.dumps(config,indent=2)+'\n')
    sources=['tree_correction_search.py','tree_correction_submission.py','gpu_residual_submission.py',
             'expression_features.py','rich_features.py','baseline.py','features.py','quantity_features.py','arith.py','training.py']
    freeze=dict(selected=selected,test_evaluated=False,search_seconds=time.perf_counter()-tick,
        environment=dict(python=platform.python_version(),numpy=np.__version__,lightgbm=lgb.__version__,threads=4),
        source_hashes={name:hashlib.sha256((CODE/name).read_bytes()).hexdigest() for name in sources},
        asset_hashes={p.name:hashlib.sha256(p.read_bytes()).hexdigest() for p in FROZEN.iterdir()})
    (OUT/'freeze.json').write_text(json.dumps(freeze,indent=2)+'\n')
    print(json.dumps(freeze,indent=2),flush=True)


if __name__=='__main__':main()
