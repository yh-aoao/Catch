"""Offline initial posture audit. No hardware API is imported or called."""
import ast
import hashlib
import json
from pathlib import Path
import sys
import tarfile
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from observation_adapter import ArmObservation, ROOT


def arrays(source):
    result={}
    for node in ast.parse(source).body:
        if isinstance(node,ast.Assign) and isinstance(node.value,ast.Call):
            for target in node.targets:
                if isinstance(target,ast.Name) and target.id in ('arm_joints','roll_arm_joints'):
                    result[target.id]=ast.literal_eval(node.value.args[0])
    return result


def main():
    folder=ROOT/'analysis_artifacts/xarm_arm_obs_20261123_152946'
    latest=json.loads((folder/'observations.jsonl').read_text().splitlines()[-1])
    current=np.array(latest['joints_rad']);source=ROOT/'catch_it_copy/configs/env/DcmmCfg.py'
    starts=arrays(source.read_text());target=np.array(starts['arm_joints'])
    adapter=ArmObservation();limits=adapter.model.jnt_range[adapter.ids]
    assert np.all(target>=limits[:,0]) and np.all(target<=limits[:,1])
    historical={}
    archive=ROOT/'analysis_artifacts/distill_best_multiseed_20261029_v1/historical_sources_and_assets.tar.gz'
    with tarfile.open(archive) as tar:
        for name in ('future/configs/env/DcmmCfg.py','future/configs/env/DcmmCfg_roll_645edc4.py','future/configs/env/DcmmCfg_bounce_5fe75d5f.py'):
            historical[name]=arrays(tar.extractfile(name).read().decode())
    report=dict(reference_only=True,motion_commands_sent=0,
                current_recording=str(folder),config=str(source),config_sha256=hashlib.sha256(source.read_bytes()).hexdigest(),
                current_joints_rad=current.tolist(),current_joints_deg=np.rad2deg(current).tolist(),
                initial_joints_rad=target.tolist(),initial_joints_deg=np.rad2deg(target).tolist(),
                direct_numeric_difference_deg=np.rad2deg(target-current).tolist(),
                training_limits_rad=limits.tolist(),training_limits_deg=np.rad2deg(limits).tolist(),
                current_config_initial_arrays=starts,historical_initial_arrays=historical,
                initial_within_training_limits=True,
                note='Endpoint reference only. Not an approved path or hardware joint-limit check. '
                'Joint order/sign near home was checked; full workspace and mounting calibration remain unverified.')
    figures=[]
    for q in (current,target):
        out=adapter.convert(q,len(figures)+1)
        ids=[adapter.arm_base_id]+[adapter.model.body('link'+str(i)).id for i in range(1,7)]
        figures.append(adapter.data.xpos[ids].copy())
        report['current_observation' if len(figures)==1 else 'initial_observation']=out
    output=ROOT/'analysis_artifacts/xarm_initial_pose_audit_20261123'
    output.mkdir(exist_ok=True)
    (output/'comparison.json').write_text(json.dumps(report,indent=2))
    fig=plt.figure(figsize=(12,6));allpts=np.concatenate(figures);lo=allpts.min(0);hi=allpts.max(0);mid=(lo+hi)/2;r=max(hi-lo)/2*1.2
    for i,(points,title) in enumerate(zip(figures,['Current recorded posture','Training reset posture (reference only)'])):
        ax=fig.add_subplot(1,2,i+1,projection='3d');ax.plot(*points.T,'o-',lw=4)
        for j,pt in enumerate(points):ax.text(*pt,'base' if j==0 else 'J'+str(j),fontsize=9)
        ax.set_xlim(mid[0]-r,mid[0]+r);ax.set_ylim(mid[1]-r,mid[1]+r);ax.set_zlim(mid[2]-r,mid[2]+r)
        ax.set_box_aspect((1,1,1));ax.view_init(20,-50);ax.set_title(title);ax.set_xlabel('X (m)');ax.set_ylabel('Y (m)');ax.set_zlabel('Z (m)')
    fig.suptitle('Nominal arm link origins only: no chassis/hand collision or transition-path validation')
    fig.tight_layout();fig.savefig(output/'posture_comparison.png',dpi=150)
    print(json.dumps({k:report[k] for k in ('current_joints_deg','initial_joints_deg','training_limits_deg','direct_numeric_difference_deg')},indent=2))
    print('Saved:',output)


if __name__=='__main__':main()
