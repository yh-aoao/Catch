"""Reproduce 100 passive trials per fruit using the current catalog defaults."""
import importlib.util,json,sys
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'catch_it_copy'))
spec=importlib.util.spec_from_file_location('motion_validation',ROOT/'analysis/validate_object_benchmark_20261103.py')
m=importlib.util.module_from_spec(spec);spec.loader.exec_module(m)
if __name__=='__main__':
    profiles=m.catalog()['roll'];results=[]
    for name in ('orange','apple','lemon'):
        rows=[m.trial(name,profiles[name],'roll',seed,.002) for seed in range(100)]
        times=[r['departure']['time'] for r in rows if r['departure']]
        result=dict(object=name,trials=100,passed=sum(r['passed'] for r in rows),arrival_min=min(times) if times else None,arrival_max=max(times) if times else None,rows=rows)
        results.append(result);print(name,result['passed'],result['arrival_min'],result['arrival_max'],flush=True)
    (ROOT/'analysis_artifacts/fruit_launch_revision_20261103/validation.json').write_text(json.dumps(results,indent=2))
