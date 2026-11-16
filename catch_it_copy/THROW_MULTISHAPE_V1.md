# Throw multishape v1

Opt-in: `throw_generalization=multishape_v1`. Default off preserves existing sphere behavior.
Five analytical shapes, uniformly selected per environment at construction; sizes use DcmmCfg.object_size. Mass uniform 0.035–0.075 kg, free-joint damping 0.005–0.02. Ellipsoid axes remain the original fixed values. MuJoCo infers consistent uniform-solid inertia after removing the old explicit inertia. No changes to rewards, observations, actions, termination, launcher, or PPO. This restores creation-time diversity, NOT per-episode physics sampling. Each environment retains its sampled shape and physics across resets; 32 environments do not guarantee equal shape counts.

Train Track: `bash scripts/train_throw_multishape_v1.sh track device_id=0`

Train Catch: `bash scripts/train_throw_multishape_v1.sh catch checkpoint_tracking=/absolute/path/track.pth device_id=0`

Default max_agent_steps=1000000000 is an upper limit, not a recommendation to run all steps. User may override num_envs, seed, output_name and PPO options. Catch starts from the new Track; official 6-action checkpoint is incompatible with current 8-action Track and must not be used here.

Backup: ../backups/before_throw_multishape_20261116. Validation: 100 compiled minimal XML draws cover all shapes, masses and positive inertia; actual full Tracking environment construction/reset passes. No full training run performed.
