"""Rebuild linear teachers, GPU residuals and the fixed tree/phrase recipe.

Only the supplied training inputs/references and fixed fold assignments are read.
No training prediction cache or validation/test reference is required.
"""
import os
for key in ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS','NUMEXPR_NUM_THREADS'):os.environ[key]='4'
os.environ['CUBLAS_WORKSPACE_CONFIG']=':4096:8'
import argparse,hashlib,json,platform,time
from pathlib import Path
import numpy as np
import scipy
import torch
from training import load,fit,logits,probability
from expression_features import extract,augment
from model_training import fit_network,fit_tree,fit_seeded,export
from gpu_residual_submission import manual_features

CODE=Path(__file__).resolve().parent


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output',type=Path,default=CODE/'runs/retrained_bundle')
    args=parser.parse_args()
    if args.output.exists():raise FileExistsError('Use a new output directory.')
    if not torch.cuda.is_available():raise RuntimeError('Use the CUDA PyTorch training environment.')
    tick=time.perf_counter();data=load('train')
    expr=extract(data['bank'],progress=lambda n:print(json.dumps(dict(stage='expression_features',groups=n)),flush=True))
    base=data['base'];extended=np.concatenate([base,augment(expr,True)],-1)
    manual=manual_features(base,data['graph'],expr)
    with np.load(CODE/'training_assets/folds.npz',allow_pickle=False) as z:folds=[z['fold_0'],z['fold_1']]
    assert all(f.shape==(len(data['y']),) for f in folds)
    probabilities=[];linear=[];optimisation=[]
    for index,x in enumerate((base,extended)):
        l2=.006 if index==0 else .012;member=[]
        for fold in folds:
            p=np.zeros_like(data['y'],dtype=np.float64)
            for k in np.unique(fold):
                active=(fold!=k)&data['fit_mask'];held=fold==k
                state,record=fit(x[active],data['y'][active],l2,1.,1226)
                p[held]=probability(logits(x[held],state),data['valid'][held]);optimisation.append(record)
            member.append(p)
        probabilities.append(np.stack(member))
        state,record=fit(x[data['fit_mask']],data['y'][data['fit_mask']],l2,1.,1226)
        linear.append(state);optimisation.append(record)
        print(json.dumps(dict(stage='linear_teacher',teacher=index,seconds=time.perf_counter()-tick)),flush=True)
    anchor=np.log(np.maximum((.75*probabilities[0]+.25*probabilities[1]).mean(0),1e-30))
    spec=dict(name='w32_b10',width=32,drop=.3,bound=1.)
    args.output.mkdir(parents=True)
    for name,state in zip(('anchor_base.npz','anchor_expression.npz'),linear):np.savez(args.output/name,**state)
    names=[]
    for seed in (11,23,47):
        _,states=fit_network(manual,anchor,data['y'],data['valid'],data['hard'],np.arange(8),spec,seed)
        name=f'residual_{seed}.npz';np.savez(args.output/name,**states[30]);names.append(name)
        print(json.dumps(dict(stage='gpu_residual',seed=seed,seconds=time.perf_counter()-tick)),flush=True)
    # Fixed shallow-tree recipe, using training OOF anchors already computed above.
    from tree_correction_submission import tree_input
    import lightgbm as lgb
    recipe=json.loads((CODE/'training_assets/recipe.json').read_text())
    raw=tree_input(manual,data['conditions'],anchor,'conditions')
    original=fit_tree(raw,anchor,data['y'],data['valid'],data['hard'],dict(loss='ce',leaves=7))
    np.savez(args.output/'original_trees.npz',**export(original,160))
    bag_spec=dict(leaves=15,min_leaf=100,l2=20.)
    for seed in (73,101,137):
        model=fit_seeded(raw,anchor,data['y'],data['valid'],data['hard'],bag_spec,seed)
        np.savez(args.output/f'trees_{seed}.npz',**export(model,160))
    config=json.loads((CODE/'runs/model_bundle/config.json').read_text())
    (args.output/'config.json').write_text(json.dumps(config,indent=2)+'\n')
    record=dict(training_split='train',training_groups=len(data['y']),linear_fit_groups=int(data['fit_mask'].sum()),
        residual_fit_groups=int(data['hard'].sum()),seeds=[11,23,47],teacher_folds=2,folds_per_assignment=3,
        loss='uniform-positive CE plus 0.02 residual mean square',adamw_lr=.001,adamw_weight_decay=.05,
        selected_epoch=30,computed_epoch_snapshots=[10,30,60],batch_size=128,gradient_clip=5,
        spec=spec,seconds=time.perf_counter()-tick,python=platform.python_version(),numpy=np.__version__,
        scipy=scipy.__version__,torch=torch.__version__,cuda=torch.version.cuda,gpu=torch.cuda.get_device_name(0),
        peak_gpu_memory_MiB=torch.cuda.max_memory_allocated()/2**20,linear_optimisation=optimisation,
        assets={p.name:hashlib.sha256(p.read_bytes()).hexdigest() for p in args.output.iterdir()})
    record['postprocessing']=config['order']
    record['tree_recipe']=recipe
    record['lightgbm']=lgb.__version__
    # Metadata lives outside the inference asset directory.
    metadata=args.output.parent/(args.output.name+'-training.json')
    metadata.write_text(json.dumps(record,indent=2)+'\n')
    print(json.dumps(record,indent=2),flush=True)


if __name__=='__main__':main()
