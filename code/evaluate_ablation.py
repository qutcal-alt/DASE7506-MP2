"""Score predeclared fixed-weight removals; never select or change the final method.

Features are shared between variants to reduce repeated work. These timings are
not fresh-process evaluation measurements. Every score uses the official scorer.
"""
import os
for key in ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS','NUMEXPR_NUM_THREADS'):os.environ[key]='4'
import argparse,copy,json,subprocess,sys
from pathlib import Path
import numpy as np
from common import PROTOCOL,load_inputs,parse_answer,validate_choices
from baseline import design
from rich_features import extract as graph_extract
from expression_features import extract as expression_extract
from order_submission import load_states,score_features,rank_view,mixed_score
from majority_baseline import predict as majority

CODE=Path(__file__).resolve().parent


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--checkpoint',type=Path,default=Path('runs/model_bundle'))
    parser.add_argument('--output-dir',type=Path,default=Path('../records/ablations'))
    args=parser.parse_args();root=(CODE/args.checkpoint).resolve();out=(CODE/args.output_dir).resolve();out.mkdir(parents=True,exist_ok=True)
    core=json.loads((root/'config.json').read_text());states=load_states(root,core)
    variants={name:copy.deepcopy(core) for name in ('full','no_neural','no_trees','no_rank_adjustment','linear_only')}
    variants['no_neural']['reference_weight']=0.
    variants['no_trees']['trees']=[]
    variants['no_rank_adjustment']['order']['weight']=0.
    variants['linear_only']['reference_weight']=0.;variants['linear_only']['trees']=[]
    (out/'predeclared-variants.json').write_text(json.dumps(dict(variants=variants,majority_tie='first supplied valid candidate',
        fit_weights=False,choose_final_method=False,shared_features=True),indent=2)+'\n')
    summary={}
    for split in ('validation','test'):
        bank,arrays=load_inputs(split);base=design(bank,arrays);graph,conditions=graph_extract(bank);expr=expression_extract(bank)
        valid=np.array([[parse_answer(c['text']) is not None for c in row['candidates']] for row in bank])
        choices={'majority':majority(bank,arrays)}
        for name,config in variants.items():
            original=score_features(base,graph,conditions,expr,valid,config,states);order=config['order']
            if order['weight']:
                adjusted=score_features(rank_view(base,order['strength']),graph,conditions,expr,valid,config,states)
                score=mixed_score(original,adjusted,valid,order['weight'],order['mode'])
            else:score=original
            choices[name]=np.where(valid,score,-1e30).argmax(1).astype(int).tolist()
        summary[split]={}
        for name,selected in choices.items():
            ids=[row['id'] for row in bank];validate_choices(bank,ids,selected)
            folder=out/name;folder.mkdir(exist_ok=True);predictions=folder/f'{split}-predictions.json';result=folder/f'{split}-score.json'
            predictions.write_text(json.dumps(dict(protocol=PROTOCOL,split=split,ids=ids,selected=selected),indent=2)+'\n')
            process=subprocess.run([sys.executable,str(CODE/'score_predictions.py'),'--split',split,'--predictions',str(predictions),
                '--output',str(result)],cwd=CODE,capture_output=True,text=True,encoding='utf-8')
            (folder/f'{split}-scoring.log').write_text(process.stdout+process.stderr,encoding='utf-8')
            if process.returncode:raise RuntimeError(process.stderr)
            summary[split][name]=json.loads(result.read_text())
        print(json.dumps(dict(split=split,scores=summary[split]),indent=2),flush=True)
    (out/'summary.json').write_text(json.dumps(dict(fixed_weights=True,no_final_model_selection=True,results=summary),indent=2)+'\n')


if __name__=='__main__':main()
