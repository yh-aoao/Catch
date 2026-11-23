"""Offline SDK encoder feedback versus project MuJoCo FK; no hardware connection."""
import argparse,json
from pathlib import Path
import numpy as np
import mujoco
from scipy.spatial.transform import Rotation
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
root=Path(__file__).resolve().parents[2]
p=argparse.ArgumentParser();p.add_argument('recording',type=Path);args=p.parse_args();folder=args.recording
rows=[json.loads(l) for l in (folder/'states.jsonl').read_text().splitlines()]
assert all(r.get('joint_query_is_real') is True for r in rows)
t=np.array([r['elapsed_s'] for r in rows]);q=np.array([r['joints_rad'] for r in rows]);pose=np.array([r['pose_mm_rad'] for r in rows]);dt=np.diff(t)
model=mujoco.MjModel.from_xml_path(str(root/'catch_it_copy/assets/urdf/xarm6_right.xml'));data=mujoco.MjData(model);bid=mujoco.mj_name2id(model,mujoco.mjtObj.mjOBJ_BODY,'link6')
rotation=Rotation.from_euler('z',90,degrees=True)
xyz=[];angles=[]
for angles_q,sdk_pose in zip(q,pose):
 data.qpos[:6]=angles_q;mujoco.mj_forward(model,data)
 xyz.append(data.xpos[bid]*1000)
 sim_r=Rotation.from_matrix(data.xmat[bid].reshape(3,3))
 sdk_r=rotation*Rotation.from_euler('xyz',sdk_pose[3:])
 angles.append((sim_r.inv()*sdk_r).magnitude()*180/np.pi)
xyz=np.array(xyz);difference=xyz-rotation.apply(pose[:,:3]);norm=np.linalg.norm(difference,axis=1)
step=np.max(np.abs(np.diff(q,axis=0)),axis=1)*180/np.pi
moving=np.r_[False,step/dt>.1]
quiet=np.convolve(moving.astype(int),np.ones(11,dtype=int),'same')==0
stats=lambda values:dict(zip(['median','p95','max'],np.percentile(values,[50,95,100]).tolist()))
report=dict(samples=len(rows),explicit_actual_joint_query=True,
 loop_interval_ms=stats(dt*1000),query_duration_ms=stats([(r['host_monotonic_end']-r['host_monotonic_begin'])*1000 for r in rows]),
 joint_ranges_deg=np.ptp(np.rad2deg(q),axis=0).tolist(),max_step_deg=np.max(abs(np.diff(np.rad2deg(q),axis=0)),axis=0).tolist(),
 error_warn_codes=[list(x) for x in sorted({tuple(r['error_warn']) for r in rows})],
 state_counts={str(x):sum(r['state']==x for r in rows) for x in sorted({r['state'] for r in rows})},
 quiet_samples=int(quiet.sum()),fk_comparison=dict(model='assets/urdf/xarm6_right.xml',body='link6',
 transform_tested='Rz(+90deg): [x,y,z] -> [-y,x,z]; orientation assumed SDK extrinsic xyz RPY',
 quiet_position_error_mm=stats(norm[quiet]),quiet_orientation_error_deg=stats(np.array(angles)[quiet]),
 quiet_mean_position_difference_mm=difference[quiet].mean(0).tolist(),
 limitation='Small near-home sweep, nominal model only. SDK TCP pose and actual joints are sequential and may refer to different instants. '
 'This supports a coordinate hypothesis, not a calibrated whole-workspace transform or an independently measured physical TCP accuracy.'),
 last5s_joint_peak_to_peak_deg=np.ptp(np.rad2deg(q[t>t[-1]-5]),axis=0).tolist())
(folder/'actual_feedback_analysis.json').write_text(json.dumps(report,indent=2))
fig,axs=plt.subplots(4,2,figsize=(13,11),sharex=True)
for j in range(6):
 ax=axs.flat[j];ax.plot(t,np.rad2deg(q[:,j]),lw=1);ax.set_ylabel('J{} (deg), actual'.format(j+1));ax.grid(alpha=.25)
ax=axs.flat[6]
for j,s in enumerate('XYZ'):ax.plot(t,pose[:,j]-pose[0,j],label=s)
ax.set_ylabel('SDK TCP delta (mm)');ax.legend();ax.grid(alpha=.25)
ax=axs.flat[7];ax.plot(t,norm,label='Position difference');ax.scatter(t[quiet][::10],norm[quiet][::10],s=3,label='Quiet samples');ax.legend();ax.set_ylabel('FK difference (mm)');ax.grid(alpha=.25)
for ax in axs[-1]:ax.set_xlabel('Time (s)')
fig.suptitle('Actual joint feedback; nominal FK compared after +90 deg Z rotation\nSDK pose and joints queried sequentially, not synchronized')
fig.tight_layout();fig.savefig(folder/'actual_feedback_analysis.png',dpi=150)
print(json.dumps(report,indent=2))
