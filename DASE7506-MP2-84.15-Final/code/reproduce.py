"""Official data checks and fresh-process prediction plus independent scoring."""
import argparse,hashlib,json,os,platform,subprocess,sys,time,zipfile
from pathlib import Path
from importlib.metadata import version

CODE=Path(__file__).resolve().parent


def cpu_name():
    if sys.platform=='win32':
        try:
            import winreg
            with winreg.OpenKey(winreg.HKEY_LOCAL_MACHINE,r'HARDWARE\DESCRIPTION\System\CentralProcessor\0') as key:
                return winreg.QueryValueEx(key,'ProcessorNameString')[0].strip()
        except OSError:pass
    return platform.processor() or platform.machine()


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--checkpoint',type=Path,default=Path('runs/model_bundle'))
    parser.add_argument('--output-dir',type=Path,default=Path('runs/reproduction'))
    args=parser.parse_args();checkpoint=(CODE/args.checkpoint).resolve()
    output=(CODE/args.output_dir).resolve();output.mkdir(parents=True,exist_ok=True)
    env=dict(os.environ,PYTHONIOENCODING='utf-8',PYTHONDONTWRITEBYTECODE='1')
    for key in ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS','NUMEXPR_NUM_THREADS'):env[key]='4'
    manifest=json.loads((CODE/'PACKAGE_MANIFEST.json').read_text())
    hashes={name:hashlib.sha256((CODE.parent/name).read_bytes()).hexdigest()==value
            for name,value in manifest['files'].items() if name!='code/submission.py'}
    assert all(hashes.values()),hashes
    historical=CODE.parent/'records/selection'
    original=json.loads((historical/'order-freeze-original.json').read_text())
    for name,expected in original['source_hashes'].items():
        assert hashlib.sha256((historical/'source_snapshot'/name).read_bytes()).hexdigest()==expected,name
    if checkpoint==CODE/'runs/model_bundle':
        frozen=json.loads((CODE/'training_assets/inference-integrity.json').read_text())
        for name,expected in frozen['asset_hashes'].items():
            assert hashlib.sha256((checkpoint/name).read_bytes()).hexdigest()==expected,name
        for name,expected in frozen['source_hashes'].items():
            assert hashlib.sha256((CODE/name).read_bytes()).hexdigest()==expected,name
    records=[];scores={}
    def run(name,args):
        tick=time.perf_counter();command=[sys.executable,*map(str,args)]
        p=subprocess.run(command,cwd=CODE,env=env,capture_output=True,text=True,encoding='utf-8')
        elapsed=time.perf_counter()-tick
        (output/(name+'.log')).write_text(p.stdout+p.stderr,encoding='utf-8')
        records.append(dict(step=name,command=command,wall_seconds=elapsed,exit_code=p.returncode))
        print(p.stdout,end='',flush=True)
        if p.returncode:raise RuntimeError(p.stderr)
        return elapsed
    run('check-data',['check_data.py'])
    for split in ('validation','test'):
        predictions=output/(split+'-predictions.json');score=output/(split+'-score.json')
        pt=run(split+'-predict',['predict.py','--method','submission.py','--checkpoint',checkpoint,'--split',split,'--output',predictions])
        st=run(split+'-score',['score_predictions.py','--split',split,'--predictions',predictions,'--output',score])
        scores[split]={**json.loads(score.read_text()),'prediction_process_seconds':pt,
                       'scoring_process_seconds':st,'complete_cpu_seconds':pt+st}
    uncompressed=0;assets={}
    for path in checkpoint.iterdir():
        if path.suffix not in ('.npz','.json') or path.name=='training.json':continue
        assets[path.name]=dict(bytes=path.stat().st_size,sha256=hashlib.sha256(path.read_bytes()).hexdigest())
        if path.suffix=='.npz':
            with zipfile.ZipFile(path) as z:uncompressed+=sum(x.file_size for x in z.infolist())
        else:uncompressed+=path.stat().st_size
    assert uncompressed<=16*2**20
    result=dict(scores=scores,assets=assets,uncompressed_inference_bytes=uncompressed,
        official_file_hash_checks=hashes,processes=records,
        source_sha256={name:hashlib.sha256((CODE/name).read_bytes()).hexdigest() for name in ('submission.py', 'order_submission.py', 'baseline.py', 'features.py', 'arith.py', 'quantity_features.py', 'rich_features.py', 'expression_features.py', 'gpu_residual_submission.py', 'tree_correction_submission.py', 'model_training.py', 'training.py', 'train_export.py')},
        environment=dict(python=platform.python_version(),numpy=version('numpy'),threads=4,platform=platform.platform(),
                         local_cpu=cpu_name()))
    (output/'summary.json').write_text(json.dumps(result,indent=2,ensure_ascii=False)+'\n',encoding='utf-8')
    print(json.dumps(scores,indent=2),flush=True)


if __name__=='__main__':main()
