"""Validate that passive isolation preserves the production object's motion."""
import importlib.util
from pathlib import Path
import json
import copy
import sys
import numpy as np
import mujoco
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT/'catch_it_copy'))
spec=importlib.util.spec_from_file_location('audit',ROOT/'analysis/audit_roll_launch_20261111.py')
audit=importlib.util.module_from_spec(spec);spec.loader.exec_module(audit)

if __name__=='__main__':
    rows=[]
    for name in audit.catalog(audit.CAT)['roll']:
        for mode in ('fixed','dr'):
            env=audit.make_object_benchmark(object_benchmark='roll_v1',benchmark_catalog=audit.CAT,
                benchmark_options={'domain_randomization':'physical_dr_v1' if mode=='dr' else 'off'},
                object_motion='roll',task='Catching',test_object=name,viewer=False,render_mode=None,
                object_eval=False,benchmark_slot=0,benchmark_num_envs=1,benchmark_seed=62000)
            try:
                env.reset(seed=70000);src=env.unwrapped.Dcmm
                types=src.model.geom_contype.copy();friction=src.model.pair_friction.copy()
                model,data=audit.passive_model(env)
                assert np.array_equal(types,src.model.geom_contype)
                assert np.array_equal(friction,src.model.pair_friction)
                reference=copy.copy(src.model);ref=mujoco.MjData(reference)
                for d in (data,ref):
                    d.qpos[:]=src.data.qpos;d.qvel[:]=src.data.qvel;d.ctrl[:]=0.
                    mujoco.mj_forward(model if d is data else reference,d)
                bid=model.body('object').id;q=int(model.jnt_qposadr[model.body_jntadr[bid]])
                error=0.
                for _ in range(100):
                    mujoco.mj_step(model,data);mujoco.mj_step(reference,ref)
                    error=max(error,float(np.max(np.abs(data.qpos[q:q+7]-ref.qpos[q:q+7]))))
                assert error<1e-9,(name,mode,error)
                rows.append(dict(object=name,mode=mode,steps=100,max_object_qpos_error=error,
                    table_edge_y=float(data.geom_xpos[model.geom('table_surface').id,1]-model.geom_size[model.geom('table_surface').id,1])))
            finally:env.close()
    (audit.OUT/'passive_equivalence.json').write_text(json.dumps(rows,indent=2))
    print('PASS',len(rows),'production/passive object motion comparisons')
