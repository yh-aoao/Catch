# distillation_aux_belief

Independent C-series variant: `belief`, on the B6 position-auxiliary baseline.
See [shared design, tests and commands](../visual_module_experiments/README.md).
Run: `python -m distillation_aux_belief.run train --stage catch --device cuda:0 --gl-backend egl --iterations 10000 --eval-every 2000 --episodes 300 --num-envs 4 --skip-teacher-baseline --output distillation_aux_belief/runs/catch_10000_seed123`
