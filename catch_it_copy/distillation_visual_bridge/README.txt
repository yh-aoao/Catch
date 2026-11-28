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

Ablation experiments (2026-11-27)
-------------------------------
Default (no --ablation): unchanged full model.
--ablation no-position: remove BOTH object-position aux_head and basket-position
  auxiliary head and their losses; retain Future forecast, fusion and loss.
--ablation no-future: remove forecast, fusion and future loss; Catch action head
  reads GRU h directly; retain object and basket position auxiliary supervision.
--ablation none: remove both position heads and Future branches/losses.

Basket uses its original direct action head in all variants; Future was only
connected to Catch even in the full model. Position outputs were supervision-only.
Disabled modules are absent from state_dict/optimizer, rather than only loss
weights being zero. Metrics keep disabled losses at zero for consistent tables.
Auxiliary label collection/replay format stays compatible and is not a policy input.
Teacher files, scene physics, command, sampling, task weights and eval stay fixed.
Use separate output directories and fresh training for the four experiments.
Ablation choice is stored in checkpoint config and restored automatically for
resume/eval; changing it on resume or eval is rejected. Original full model state
keys and default forward/loss results are unchanged (verified against HEAD).

Use the usual command with these output/flag pairs:
full:        --output distillation_visual_bridge/runs/bridge_seed123
no-position: --output distillation_visual_bridge/runs/bridge_no_position_seed123 --ablation no-position
no-future:   --output distillation_visual_bridge/runs/bridge_no_future_seed123 --ablation no-future
none:        --output distillation_visual_bridge/runs/bridge_no_aux_seed123 --ablation none
All: --num-envs 32 --rollout-steps 16 --iterations 10000 --eval-start 1000
     --eval-every 1000 --episodes 300 --skip-teacher-baseline
     --config distillation_visual_bridge/configs/server.json

Resource lifecycle audit (2026-11-27)
------------------------------------
num-envs is PER TASK: 8 -> 32 persistent environment/render processes;
32 -> 128. Pools initialize sequentially; [pool] task/worker/total/pid logs make
startup growth visible. Each environment owns one reusable renderer/context;
Top/Wrist use it sequentially. Worker processes stay alive during training.
Evaluations add up to eval_num_envs (default 4) at a time, then close that pool;
training pools remain resident. Neither pool nor renderer is created every step.

Fixes:
- Snapshot/restore CUDA RNG only for active network GPU, avoiding all-device
  CUDA context initialization. Reads legacy all-GPU RNG checkpoints too.
- Keep full replay depth episodes on CPU; move only BPTT depth chunks to GPU.
  Loss metadata remains small; loss weighting/lengths and optimizer work unchanged.
- Drop last chunk prediction references after optimization.
- Pool shutdown closes pipe endpoints, joins with bounded shared deadlines,
  then terminates/kills and reaps stuck workers. Idempotent close releases parent
  references. Worker finally attempts all resource cleanup even if one close fails.
- SIGTERM of the training main process now unwinds through cleanup. SIGKILL cannot
  run Python finally blocks; no automatic termination of other user jobs is done.

metrics.jsonl adds environment_processes, cuda_allocated_mb, cuda_reserved_mb,
cuda_peak_allocated_mb and cuda_peak_reserved_mb. These CUDA counters cover
PyTorch on the network device, NOT EGL renderer allocations in worker processes.
Reserved/peak can grow as allocator high-water marks; that alone is not a leak.
Replay capacity and pending episode limits already bound storage in host RAM.
No empty_cache per-step is used, since it does not free live tensors or EGL contexts.

Sync the updated package, stop the old job cleanly, then start the same command.
To resume an existing job with the source fix, use resume_last.pth and the SAME
output directory plus --allow-source-changes; do not change its ablation/num-envs/
rollout-steps. Existing processes do not hot-reload these changes.

Basket teacher/student audit (2026-11-28)
---------------------------------------
State-only teacher smoke in exact distillation scene, 20 seeds 900123..900142:
18 reached bridge handoff, 17 geometric circle hits, 2 strict task successes.
Action conversion max error 5.96e-8. No legacy holding/braking overrides.
Teacher SHA e00e4fb57fdd3db35e2e8944f008f9d53a5925c18789549b603f56890805c552.
Prior reported 78/100 with diameter0.6 used geometric receiving-region scoring;
its stopped-hit count was 14/100. Do not equate this with strict task success.
Teacher still consumes a rule-driven carry/throw flag to select two learned heads;
student has no phase input and must infer this from vision/state/history.

Eval now records reached_throw_phase, released, release_stopped,
basket_region_hit, basket_stopped_region_hit for each Basket episode, plus counts
in its object diagnostics. Region counters are diagnostic only: existing success
field, rewards and termination criteria are unchanged. Pure teacher diagnostic:
python -m distillation_visual_bridge.audit_basket \
  --config distillation_visual_bridge/configs/server.json \
  --output analysis_artifacts/bridge_basket_teacher_audit --episodes 20
This needs CPU only, with MUJOCO_GL=disable; it does not evaluate student vision.

Fixed worker Basket reset to pass episode_seed to Gym's Generator as well as
seeding global NumPy/Python. Catch legacy reset API is unchanged. This makes
future seeded Basket comparisons reproducible; historical evaluation seeds did
not fully specify its randomized holding pose. Source changes still need explicit
--allow-source-changes when evaluating/resuming old checkpoints.

Evaluate existing students with receiving-region hit scoring:
python -m distillation_visual_bridge.run eval \
  --checkpoint distillation_visual_bridge/bridge_no_future_env8_seed123/student_002000.pth \
  --output analysis_artifacts/bridge_no_future_2000_hit_eval_01 \
  --tasks basket --basket-success-metric hit --episodes 300 --eval-num-envs 8 \
  --device cuda:0 --gl-backend egl --egl-device-id 0 --allow-source-changes

--basket-success-metric hit selects basket_region_hit, the active 0.6 m diameter
receiving-region diagnostic: confirmed intentional flight, ball clearance and safe
robot episode; no stopped-release condition. It does NOT override physical controls,
bridge handoff, episode termination or training configuration. --strict is not a
flag: use --basket-success-metric strict for the old metric. Both counts appear
in objects.tennis_ball.diagnostics (basket_region_hit vs safe_basket_success).
Evaluation uses the ablation/model settings from the checkpoint automatically;
no teacher checkpoint or server.json is needed. --eval-num-envs controls actual
parallel eval workers (unlike --num-envs, which configures training pools).

Carry-failure diagnostics and optional repair pilot (2026-11-28)
--------------------------------------------------------------
Matched-seed diagnostic modes: teacher; student; teacher-carry (teacher until
handoff, then student); teacher-base (only base from teacher); teacher-arm-hand
(only arm/hand from teacher). These oracle replacements are diagnostic only,
not deployment success rates and not environment assistance in training.
Example, from catch_it_copy:
python -m distillation_visual_bridge.diagnose_control \
 --checkpoint distillation_visual_bridge/bridge_no_future_env8_seed123/student_002000.pth \
 --teacher-checkpoint basket_mobile_bridge/runs/tennis_bridge_seed123/teacher_last.pth \
 --output analysis_artifacts/bridge_control_diagnosis --device cpu --egl-device-id 0 --episodes 5
Real depth is rendered. Teachers use privileged state only for diagnostics/labels.
Per-step JSONL records student/teacher/executed actions, ball slip/local position,
base speed, target error and stage; summary.json contains per-mode hit counts.

Optional NEW experiment flags (defaults preserve original four ablations):
 --basket-rollin episode
  Choose teacher/student once at episode start and keep that executor until
  reset. Student still runs recurrent inference and receives teacher labels on
  every observed state. The beta schedule controls episode selection probability,
  not necessarily teacher fraction by frames because episode lengths can differ.
 --basket-carry-precision-weight 10
  Extra MSE on arm6+hand12 during Basket carry, using teacher phase as loss-only
  metadata. No phase input, forced hold posture, action mask or physics overrides.
  Basket task imitation/position heads and the chosen ablation remain unchanged.
 --basket-success-metric hit
  Now also available for fresh train runs to use receiving-region hit success in
  periodic evaluation/best selection. Physical termination/rewards are unchanged.
 --init-checkpoint PATH
  Initialize student/normalizer only; optimizer, replay, iteration and beta
  schedule reset. Requires same ablation, teachers, task scenes and model; permits
  new repair settings/eval metric. Writes initialization_audit.json. This is NOT
  a continuation of the old ablation and must use a new output directory.

Repair pilot from the 2000-iteration no-future model:
python -m distillation_visual_bridge.run train \
 --config distillation_visual_bridge/configs/server.json \
 --init-checkpoint distillation_visual_bridge/bridge_no_future_env8_seed123/student_002000.pth \
 --output distillation_visual_bridge/runs/repair_carry_no_future_env8_seed123 \
 --ablation no-future --basket-rollin episode --basket-carry-precision-weight 10 \
 --basket-success-metric hit --device cuda:2 --gl-backend egl --egl-device-id 2 \
 --num-envs 8 --rollout-steps 64 --iterations 1000 \
 --eval-start 250 --eval-every 250 --episodes 300 --skip-teacher-baseline
Pilot iteration250 means 250 additional iterations after loading old weights.
Repair efficacy is not established until trained/evaluated. Do not combine pilot
scores with original four-ablation results; compare it as a separate intervention.

Fresh full-model repair run (no previous weights):
python -m distillation_visual_bridge.run train \
 --config distillation_visual_bridge/configs/server.json \
 --output distillation_visual_bridge/runs/repair_carry_full_from_scratch_env8_seed123 \
 --ablation full --basket-rollin episode --basket-carry-precision-weight 10 \
 --basket-success-metric hit \
 --device cuda:0 --gl-backend egl --egl-device-id 6 \
 --num-envs 8 --rollout-steps 64 --iterations 10000 \
 --eval-start 1000 --eval-every 1000 --episodes 300 --skip-teacher-baseline
No --resume or --init-checkpoint: random student and fresh normalization/replay/Adam.
GPU example assumes network logical0 and an accessible EGL6. Check actual node.
The hit metric is the 0.6m receiving-region standard associated with the earlier
78% teacher result, not a guarantee that every scene/seed yields exactly 78%.
Only evaluation/best-model selection changes, not physical handoff/termination.

Network CUDA device preflight now occurs before output directory creation and
teacher/model allocation. Invalid logical ordinal errors report visible device
count, valid cuda:N values, and CUDA_VISIBLE_DEVICES. No silent remapping occurs.
Example: CUDA_VISIBLE_DEVICES=6 -> usually only cuda:0 is valid for PyTorch;
--device cuda:6 is then invalid. --egl-device-id uses a separate EGL enumeration
and is NOT checked against PyTorch's filtered device count. Existing single/multi
render placement flags stay unchanged. Sync run.py/runtime.py (or full package).

2026-11-28: Optional single-head Basket throwing repair
-----------------------------------------------------
--basket-phase-weight 0.1 enables a GRU(256)->128->1 auxiliary phase
classifier. Teacher basket_phase is a training label only. Predictions do not
select, mix or gate actions, and no stage is added to deployment inputs.
--basket-throw-precision-weight 1 enables normalized-action MSE during the
throw phase. Base2, arm6 and hand12 group errors receive equal weight. It uses
the existing within-episode phase balancing (half navigation, half throwing
when both exist), then the existing per-task weight. The entire throw phase
is supervised, including post-release frames; it is not a release-only loss.
These starting weights are experimental, not verified optimal settings.
Both options default to zero, preserving existing experiments and checkpoints.
The phase classifier is independent of the position/future ablation switches.
Logs add basket_phase_loss and basket_throw_precision_loss (unweighted terms,
including existing task weighting); total loss applies the configured weights.

To transfer an existing single-head student, use --init-checkpoint in a NEW
output directory with the same ablation/teacher/model/environment contracts.
Existing action/visual/GRU/normalizer weights are copied unchanged; only a newly
enabled phase head is randomly initialized. All other missing/unexpected keys
are errors. Optimizer/replay/iteration/teacher-mix schedule start afresh, and
initialization_audit.json records new parameter keys. This is not exact resume.
Do not use --resume to add these options to an existing experiment.
Eval/viewer reconstruct the optional classifier automatically from checkpoint.
