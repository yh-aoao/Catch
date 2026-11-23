"""Offline plots/statistics for the first xArm connectivity recording."""
import json
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
root=Path(__file__).resolve().parents[2]
p=root/'analysis_artifacts/xarm_readonly_20261123_151837'
rows=[json.loads(l) for l in (p/'states.jsonl').read_text().splitlines()]
t=np.array([r['elapsed_s'] for r in rows]);q=np.rad2deg([r['joints_rad'] for r in rows]);pose=np.array([r['pose_mm_rad'] for r in rows]);dt=np.diff(t)
fig,axs=plt.subplots(4,2,figsize=(13,11),sharex=True)
for i in range(6):
 ax=axs.flat[i];ax.plot(t,q[:,i],linewidth=1);ax.set_ylabel('J{} (deg)'.format(i+1));ax.grid(alpha=.25)
ax=axs.flat[6]
for i,name in enumerate('XYZ'):ax.plot(t,pose[:,i]-pose[0,i],label=name)
ax.legend();ax.set_ylabel('TCP delta (mm)');ax.set_xlabel('Time (s)');ax.grid(alpha=.25)
ax=axs.flat[7];ax.plot(t,[r['state'] for r in rows]);ax.set_ylabel('Controller state');ax.set_xlabel('Time (s)');ax.grid(alpha=.25)
fig.suptitle('xArm SDK recording: default joint query is_real=False\nConnectivity diagnostic; NOT verified measured joint feedback')
fig.tight_layout();fig.savefig(p/'feedback_analysis.png',dpi=150)
idx=int(np.argmax(np.max(np.abs(np.diff(q,axis=0)),axis=1)))
report=dict(samples=len(rows),loop_interval_ms_percentiles=dict(zip(['min','median','p95','max'],np.percentile(dt*1000,[0,50,95,100]).tolist())),query_duration_ms_percentiles=dict(zip(['median','p95','max'],np.percentile([(r['host_monotonic_end']-r['host_monotonic_begin'])*1000 for r in rows],[50,95,100]).tolist())),joint_ranges_deg=np.ptp(q,axis=0).tolist(),states={str(s):sum(r['state']==s for r in rows) for s in sorted({r['state'] for r in rows})},error_warn_codes=[list(s) for s in sorted({tuple(r['error_warn']) for r in rows})],largest_joint_jump=dict(time_s=[float(t[idx]),float(t[idx+1])],delta_deg=(q[idx+1]-q[idx]).tolist()),limitation='Original recorder used SDK default is_real=False; do not treat differences as measured motion velocities. Pose and joints queried sequentially; no hardware timestamp synchronization.')
(p/'feedback_analysis.json').write_text(json.dumps(report,indent=2))
print(json.dumps(report,indent=2))
