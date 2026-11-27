"""Five-seed Basket evaluation using the original virtual receiving-region criterion."""
import argparse
from concurrent.futures import ThreadPoolExecutor,as_completed
import csv
from datetime import datetime
import hashlib,json,os
from pathlib import Path
import shlex,subprocess,sys
import statistics

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))


def sha(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def aggregate(cases,diameters):
    # Reuse the SAME summarizer as the original 78% evaluation.
    from basket_mobile_bridge.eval_regions import summarize
    records=[record for case in cases for record in case['records']]
    radii={case['ball_radius'] for case in cases}
    if len(radii)>1:raise ValueError('Ball radius differs between seeds')
    result=summarize(records,diameters,next(iter(radii),.033275785333333335))
    result['seeds_completed']=len(cases)
    for region in result['regions']:
        rates=[next(r['success_rate'] for r in case['summary']['regions'] if r['diameter_m']==region['diameter_m']) for case in cases]
        region['seed_rate_std']=statistics.stdev(rates) if len(rates)>1 else None
    return result


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--basket-checkpoint','--checkpoint',dest='checkpoint',type=Path,
        default=ROOT/'basket_mobile_bridge/runs/tennis_bridge_seed123/teacher_last.pth')
    p.add_argument('--seeds',nargs=5,type=int,default=[41000,42000,43000,44000,45000])
    p.add_argument('--episodes-per-seed',type=int,default=100)
    p.add_argument('--diameters',nargs='+',type=float,default=[.6,.8])
    p.add_argument('--jobs',type=int,default=5);p.add_argument('--output',type=Path)
    p.add_argument('--resume',action='store_true');p.add_argument('--dry-run',action='store_true')
    a=p.parse_args()
    import math
    if a.episodes_per_seed<1 or a.jobs<1:p.error('Counts must be positive')
    if len(set(a.seeds))!=5 or any(s<0 or s+a.episodes_per_seed>=2**32 for s in a.seeds):p.error('Supply five distinct valid seeds')
    ordered=sorted(a.seeds)
    if any(y-x<a.episodes_per_seed for x,y in zip(ordered,ordered[1:])):p.error('Episode seed ranges must not overlap')
    if not a.diameters or any(not math.isfinite(d) or d<=0 for d in a.diameters):p.error('Diameters must be positive and finite')
    a.diameters=sorted(set(a.diameters));checkpoint=a.checkpoint.resolve()
    if not checkpoint.is_file():p.error('Missing checkpoint: '+str(checkpoint))
    output=(a.output or ROOT/'evaluation_results'/datetime.now().strftime('basket_regions_5seed_%Y%m%d_%H%M%S_%f')).resolve()
    paths={Path(__file__).resolve()}
    for folder in ['basket_mobile_bridge','basket_mobile_curriculum','basket_autonomous','gym_dcmm','configs']:
        paths.update((ROOT/folder).rglob('*.py'))
    sources={str(path.relative_to(ROOT)):sha(path) for path in sorted(paths)}
    spec=dict(checkpoint=str(checkpoint),checkpoint_sha256=sha(checkpoint),seeds=a.seeds,
        episodes_per_seed=a.episodes_per_seed,diameters_m=a.diameters,sources=sources,
        criterion='Original physical hoop and episode termination; confirmed purposeful throw; first descending crossing of original tilted hoop plane; whole ball clearance; no dangerous robot collision. NO stopped-release requirement.',
        policy='deterministic mean',environment='original basket_mobile_bridge, far_mobile',device='cpu')
    tasks=[]
    for seed in a.seeds:
        folder=output/('seed_'+str(seed))
        command=[sys.executable,'-m','basket_mobile_bridge.eval_regions','--checkpoint',str(checkpoint),
            '--output',str(folder),'--episodes',str(a.episodes_per_seed),'--seed',str(seed),
            '--diameters',*[str(d) for d in a.diameters],'--device','cpu']
        tasks.append((seed,folder,command))
    print('Output:',output,'\nRequested:',5*a.episodes_per_seed,'episodes; original virtual-region criterion, no stopped-release requirement.',flush=True)
    if a.dry_run:
        for _,_,cmd in tasks:print(shlex.join(cmd))
        return 0
    if a.resume:
        if not (output/'request.json').is_file() or json.loads((output/'request.json').read_text())!=spec:
            p.error('Resume model/source/request mismatch; choose a new output directory')
    else:
        if output.exists():p.error('Output exists; use --resume with matching request or choose a new directory')
        output.mkdir(parents=True);(output/'request.json').write_text(json.dumps(spec,indent=2))
    if (output/'COMPLETE').exists():(output/'COMPLETE').unlink()
    environment=dict(os.environ,OMP_NUM_THREADS='1',OPENBLAS_NUM_THREADS='1',MKL_NUM_THREADS='1',MUJOCO_GL='egl',WANDB_MODE='disabled')
    def run(task):
        seed,folder,command=task
        if not (a.resume and (folder/'COMPLETE').is_file()):
            if folder.exists():raise RuntimeError('Incomplete seed directory exists; move it aside before resume: '+str(folder))
            print('[start]',seed,flush=True)
            with (output/('seed_'+str(seed)+'.log')).open('w') as log:
                result=subprocess.run(command,cwd=ROOT,env=environment,stdout=log,stderr=subprocess.STDOUT)
            if result.returncode:raise RuntimeError('Seed failed; inspect seed_'+str(seed)+'.log')
        summary=json.loads((folder/'summary.json').read_text());metadata=json.loads((folder/'config.json').read_text())
        records=[json.loads(line) for line in (folder/'episodes.jsonl').read_text().splitlines() if line.strip()]
        if (not summary.get('complete') or summary.get('interrupted') or len(records)!=a.episodes_per_seed
            or summary['episodes']!=a.episodes_per_seed or metadata['checkpoint_sha256']!=spec['checkpoint_sha256']):
            raise RuntimeError('Incomplete or mismatched seed '+str(seed))
        if [r['seed'] for r in records]!=list(range(seed,seed+a.episodes_per_seed)):raise RuntimeError('Wrong episode seeds')
        if metadata['diameters_m']!=a.diameters:raise RuntimeError('Wrong receiving regions')
        if not metadata['unchanged_physical_rim'] or not metadata['unchanged_episode_termination']:raise RuntimeError('Wrong physical environment')
        # This evaluator is pinned to the original tennis-ball bridge schema/physics.
        radius=.033275785333333335
        from basket_mobile_bridge.eval_regions import summarize
        checked=summarize(records,a.diameters,radius)
        if checked['regions']!=summary['regions'] or checked['original_successes']!=summary['original_successes']:
            raise RuntimeError('Summary/episode disagreement')
        (folder/'COMPLETE').write_text('All requested virtual-region episodes completed\n')
        print('[complete]',seed,[(r['diameter_m'],r['hits'],r['success_rate']) for r in summary['regions']],flush=True)
        return dict(seed=seed,summary=summary,records=records,ball_radius=radius)
    completed=[];errors=[]
    with ThreadPoolExecutor(max_workers=min(a.jobs,5)) as pool:
        futures={pool.submit(run,task):task[0] for task in tasks}
        for future in as_completed(futures):
            try:completed.append(future.result())
            except Exception as error:
                errors.append(dict(seed=futures[future],error=str(error)));print(error,file=sys.stderr,flush=True)
    completed.sort(key=lambda c:c['seed']);rows=[]
    for case in completed:
        s=case['summary']
        rows.append(dict(seed=case['seed'],metric='original_safe_hoop',diameter_m=None,successes=s['original_successes'],episodes=s['episodes'],success_rate=s['original_success_rate']))
        for region in s['regions']:
            rows.append(dict(seed=case['seed'],metric='virtual_receiving_region',diameter_m=region['diameter_m'],successes=region['hits'],episodes=s['episodes'],success_rate=region['success_rate']))
    (output/'per_seed.json').write_text(json.dumps(rows,indent=2))
    with (output/'per_seed.csv').open('w',newline='') as file:
        writer=csv.DictWriter(file,fieldnames=['seed','metric','diameter_m','successes','episodes','success_rate']);writer.writeheader();writer.writerows(rows)
    summary=aggregate(completed,a.diameters)
    unchanged=sha(checkpoint)==spec['checkpoint_sha256'] and all(sha(ROOT/path)==digest for path,digest in sources.items())
    if not unchanged:errors.append(dict(error='Model/source changed during evaluation'))
    summary.update(requested_episodes=5*a.episodes_per_seed,complete=len(completed)==5 and not errors,
        criterion=spec['criterion'],checkpoint_sha256=spec['checkpoint_sha256'])
    (output/'summary.json').write_text(json.dumps(summary,indent=2));(output/'failures.json').write_text(json.dumps(errors,indent=2))
    lines=['Basket original virtual-region multiseed evaluation',
        'Completed seeds: %d/5; completed episodes: %d/%d'%(len(completed),summary['episodes'],5*a.episodes_per_seed),
        'Primary metric does NOT require stopped release; original physical hoop unchanged.',
        'Original safe hoop score: %d/%d'%(summary['original_successes'],summary['episodes'])]
    for r in summary['regions']:lines.append('Virtual diameter %.2fm: %d/%d = %s'%(r['diameter_m'],r['hits'],summary['episodes'],format(r['success_rate'],'.2%') if r['success_rate'] is not None else 'N/A'))
    (output/'REPORT.txt').write_text('\n'.join(lines)+'\n')
    if summary['complete']:(output/'COMPLETE').write_text('Five seed quotas complete; source and checkpoint hashes unchanged\n')
    print(json.dumps(summary,indent=2),flush=True)
    return 0 if summary['complete'] else 1
if __name__=='__main__':sys.exit(main())
