# distillation_aux_future

Independent C-series variant: `future`, on the B6 position-auxiliary baseline.
See [shared design, tests and commands](../visual_module_experiments/README.md).
Run: `python -m distillation_aux_future.run train --stage catch --device cuda:0 --gl-backend egl --iterations 10000 --eval-every 2000 --episodes 300 --num-envs 4 --skip-teacher-baseline --output distillation_aux_future/runs/catch_10000_seed123`
