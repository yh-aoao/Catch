Visual Bridge distillation (new experiment; no changes to the old package)

Teachers: Throw 515.51 (five shapes), Roll 403.15 (seven objects),
Bounce 16.15 (five balls), Basket mobile_bridge teacher_last (tennis ball).
Teacher content SHA256 is pinned in configs/catch.json.

Student: Top/Wrist depth + robot36 + previous20 + Catch/Basket command.
No object ID, motion ID, true basket position, parking reference, or bridge phase
enters the student. Basket position is a training-only auxiliary target.
Future Stable is retained for Catch. Basket has a direct full20 action head.
Teacher phase is used only by the teacher and to balance imitation loss.

Basket directly uses the Bridge environment and its full normalized controls:
base2*.3, arm6*.025, hand12*.15, then the same integrated joint controller.
No legacy hold-target label synthesis, phase actuator masks, or scripted braking.
The teacher's stable-arrival handoff and premature-release scoring remain.
Only far_mobile full episodes are used; no separate carry/fixed rehearsals.
Physical rim and success criterion both use diameter 0.6 m (radius 0.3 m).
Compared with the teacher's original 0.4 m rim, this changes the visible scene
and contacts; smoke checks establish compatibility, not teacher success rate.

Sampling: one complete sequence per object per update (5+7+5+1=18), task weights
25% each; 8 updates/iteration; Future Stable schedule; 10000 iterations.
Evaluate every 1000 iterations from 1000, 300 episodes/task; save each evaluated
checkpoint plus best/last/resume. num-envs is per task: 8 means 32 workers.
15 s Basket episodes require up to 375 policy steps; configured maximum >=400.

Local training (from catch_it_copy in dcmm):
python -m distillation_visual_bridge.run train --output distillation_visual_bridge/runs/bridge_seed123 --device cuda:0 --gl-backend egl --egl-device-id 0 --num-envs 8 --rollout-steps 64 --iterations 10000 --eval-start 1000 --eval-every 1000 --episodes 300 --skip-teacher-baseline

Server paths: prepare_config supports --teacher-root, --bounce-checkpoint and
--basket-checkpoint; explicit task overrides take precedence over teacher-root.
It checks all files, pins hashes, imports Basket checkpoint options and validates
Bounce grasp protocol. Do not add --teacher-root again after preparing overrides.
Resume only this experiment's resume_last.pth; older schemas are rejected.
