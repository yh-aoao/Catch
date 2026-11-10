"""Hold-out seeds for conservative 10 cm closer launch, with all yaw retained."""
from concurrent.futures import ProcessPoolExecutor,as_completed
import importlib.util
import json
from pathlib import Path
import copy
ROOT=Path(__file__).resolve().parents[1]
spec=importlib.util.spec_from_file_location('scan',ROOT/'analysis/scan_fruit_launch_20261110.py')
scan=importlib.util.module_from_spec(spec);spec.loader.exec_module(scan)
OUT=ROOT/'analysis_artifacts/fruit_joint_launch_20261110'

def run(job):
    name,distance,dr=job;p=copy.deepcopy(scan.base.base.catalog()['roll'][name]);p['distance_m']=distance
    rows=[scan.trial(name,p,s,dr) for s in range(1000,1050)]
    return scan.summarize(name,distance,[-180,180],rows,dr)

if __name__=='__main__':
    results=[];jobs=[(n,d,r) for n in ('apple','lemon') for d in ([2.,2.2],[1.9,2.1]) for r in (False,True)]
    with ProcessPoolExecutor(max_workers=3) as pool:
        for f in as_completed([pool.submit(run,j) for j in jobs]):
            result=f.result();results.append(result)
            print(json.dumps({k:v for k,v in result.items() if k!='rows'}),flush=True)
            (OUT/'holdout.json').write_text(json.dumps(results,indent=2))
