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

Multi-GPU rendering (2026-11-27)
------------------------------
Network inference/training stays on --device. Environment rendering is independent:
  --render-devices 1 2 3
round-robins workers over EGL IDs (globally offset by task), or:
  --render-task-devices throw=1 roll=1 bounce=2 basket=3
assigns an EGL ID per task. The task map must contain all four tasks.
These options and --egl-device-id are mutually exclusive. Legacy single-device
commands still work. No CUDA_VISIBLE_DEVICES changes are made by this code.
Use EGL IDs available on the server; CUDA numbering may differ when visibility
is restricted. Worker startup prints task/object/EGL_DEVICE for inspection.

Fresh server run:
python -m distillation_visual_bridge.run train \
  --config distillation_visual_bridge/configs/server.json \
  --output distillation_visual_bridge/runs/bridge_multigpu_seed123 \
  --device cuda:0 --gl-backend egl \
  --render-task-devices throw=1 roll=1 bounce=2 basket=3 \
  --num-envs 8 --rollout-steps 64 --iterations 10000 \
  --eval-start 1000 --eval-every 1000 --episodes 300 --skip-teacher-baseline

Resume (keep the existing output directory):
python -m distillation_visual_bridge.run train \
  --resume distillation_visual_bridge/runs/bridge_seed123/resume_last.pth \
  --output distillation_visual_bridge/runs/bridge_seed123 \
  --device cuda:0 --gl-backend egl \
  --render-task-devices throw=1 roll=1 bounce=2 basket=3 \
  --allow-source-changes

Rendering placement is stored in checkpoints and used in evaluation too.
It may be changed on resume; model/teacher/sampling contracts stay checked.
Use --allow-source-changes when moving a pre-change checkpoint to the updated
source. Viewer evaluation clears inherited EGL assignments unless explicitly
specified. This change does not add DDP or concurrent sampling/training.

metrics.jsonl timing fields:
  collect_seconds: total collection wall time
  student_inference_seconds / teacher_inference_seconds: includes CPU copies
  environment_wait_seconds: main-process blocked receive time, not total worker
    physics/render time; workers overlap network inference and other workers
  collect_other_seconds: collection bookkeeping and dispatch/replay work
  update_seconds: replay sampling and 8 optimizer updates
  refresh_hidden_seconds: rebuild pending-episode GRU states
  save_seconds / evaluation_seconds: checkpoint I/O and periodic evaluation
  seconds: entire iteration, now including saving/evaluation (old rows excluded it)
CUDA is synchronized at update/refresh timing boundaries. Per-worker physics and
render timings are not measured separately. Total samples remain 2048/iteration
for 8 envs/task x 64 steps x 4 tasks, and 18 sequences/update x 8 updates.

Single/multi rendering speed benchmark
-------------------------------------
python -m distillation_visual_bridge.benchmark \
  --config distillation_visual_bridge/configs/server.json \
  --output analysis_artifacts/bridge_speed_test_01 \
  --device cuda:4 --single-render-device 5 --multi-render-devices 5 6 7 \
  --num-envs 8 --rollout-steps 64 --warmup 10 --iterations 50

Runs sequentially in fresh subprocesses: single EGL 5, then EGL 5/6/7.
Each starts from identical seeds/student initialization, separately calibrates
normalization and fills replay, discards ten training iterations, then measures
50 iterations. Same 32 workers, 2048 steps/iteration, 18 sequences/update and
8 updates are preserved. Uses real optimizers, not render-only timing. No eval
or checkpoint saves during measurement; it does not resume/modify live training.
Optional --checkpoint freezes one student snapshot for both runs (fresh Adam and
replay; this is benchmarking, not an exact training resume). Input config is frozen
in input_config.json. --repeats 2 alternates order to reduce trial ordering bias.

Outputs: comparison.json with single_seconds, multi_seconds, speedup and
 time_reduction_percent; single_01/multi_01 directories hold measured metrics,
configs and summaries. Child stdout/stderr is in single_01.log/multi_01.log.
Use a quiet server and stop competing jobs on tested resources for fair comparison.
A speedup below 1 means multi-GPU rendering was slower in this experiment.
