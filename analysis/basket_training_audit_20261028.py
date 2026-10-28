"""Read existing TensorBoard records without TensorFlow; do not run training."""
import argparse
import json
import struct
from collections import defaultdict
from pathlib import Path

from tensorboardX.proto import event_pb2


def read_scalars(path):
    rows = defaultdict(list)
    with path.open('rb') as stream:
        while True:
            header = stream.read(8)
            if len(header) < 8:
                break
            length = struct.unpack('<Q', header)[0]
            stream.read(4)
            payload = stream.read(length)
            stream.read(4)
            if len(payload) < length:
                break  # A still-running writer may have an incomplete tail.
            event = event_pb2.Event()
            event.ParseFromString(payload)
            for value in event.summary.value:
                if value.HasField('simple_value'):
                    rows[value.tag].append([event.step, value.simple_value])
    return dict(rows)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('run', type=Path)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    files = sorted((args.run/'tb').glob('*tfevents*'))
    if len(files) != 1:
        parser.error('Select a run with exactly one event file to avoid mixing restarts')
    rows = read_scalars(files[0])
    rewards = rows['metrics/episode_rewards_per_step']
    peak = max(rewards, key=lambda row: row[1])
    summary = dict(run=str(args.run), last_step=rewards[-1][0],
        first_reward=rewards[0], best_reward=peak, last_reward=rewards[-1],
        last_mobile_success=rows['basket_mobile/mobile_success'][-1],
        peak_mobile_success=max(rows['basket_mobile/mobile_success'], key=lambda row: row[1]),
        last_rehearsal_success=rows['basket_mobile/rehearsal_success'][-1])
    args.output.mkdir(parents=True, exist_ok=True)
    (args.output/'training_scalars.json').write_text(json.dumps(rows), encoding='utf-8')
    (args.output/'training_summary.json').write_text(json.dumps(summary, indent=2), encoding='utf-8')
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    import numpy as np
    figure, axes = plt.subplots(1, 2, figsize=(12, 4.2), constrained_layout=True)
    x, y = np.asarray(rewards).T
    axes[0].plot(x/1e6, y, label='Current episode reward', linewidth=1)
    axes[0].plot(x/1e6, np.maximum.accumulate(y), label='Historical best reward', linestyle='--')
    axes[0].set(xlabel='Environment steps (millions)', ylabel='Reward', title='31.80 was an early historical maximum')
    for key, label in [('rehearsal_success', 'Fixed-target rehearsal'), ('mobile_success', 'Mobile basket')]:
        x, y = np.asarray(rows['basket_mobile/'+key]).T
        axes[1].plot(x/1e6, y*100, label=label, linewidth=1)
    axes[1].set(xlabel='Environment steps (millions)', ylabel='Logged success statistic (%)',
                title='Separate rehearsal and mobile results', ylim=(-2, 102))
    for axis in axes:
        axis.grid(alpha=.2)
        axis.legend()
    figure.savefig(args.output/'training_curves.png', dpi=180)
    plt.close(figure)
    print(json.dumps(summary, indent=2))


if __name__ == '__main__':
    main()
