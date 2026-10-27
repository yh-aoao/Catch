"""Reproduce the YCB reset/generalization and passive-drop diagnostic figure."""
import json
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
ARTIFACTS = ROOT/'analysis_artifacts'
before = json.loads((ARTIFACTS/'ycb_generalization_20261027/summary.json').read_text())
after = json.loads((ARTIFACTS/'ycb_generalization_reset_fixed_20261027/summary.json').read_text())
stability = json.loads((ARTIFACTS/'bounce_stability_report_20261027.json').read_text())
keys = [motion+'_'+obj for motion in ('roll', 'bounce') for obj in ('default', 'tennis_ball', 'baseball')]
labels = ['Roll\nOriginal', 'Roll\nTennis', 'Roll\nBaseball', 'Bounce\nOriginal', 'Bounce\nTennis', 'Bounce\nBaseball']
fig, axes = plt.subplots(1, 2, figsize=(13, 5.2), gridspec_kw={'width_ratios': [1.35, 1]})
x = np.arange(len(keys))
for offset, results, color, label in [(-.19, before, '#9CA3AF', 'Before reset fix'),
                                     (.19, after, '#2274A5', 'After reset fix')]:
    counts = [results[key]['successes'] for key in keys]
    axes[0].bar(x+offset, counts, width=.36, color=color, label=label)
    for where, count in zip(x+offset, counts):
        axes[0].text(where, count+.3, str(count), ha='center', fontsize=9)
axes[0].set(xticks=x, xticklabels=labels, ylim=(0, 23), yticks=[0, 5, 10, 15, 20],
            ylabel='Original task success flag (episodes / 20)', title='Same checkpoints and seeds; native object physics')
axes[0].legend(frameon=False, fontsize=9)
axes[0].grid(axis='y', alpha=.2)
axes[0].set_axisbelow(True)
for timestep, color in [(.002, '#CB4335'), (.0005, '#D68910'), (.00025, '#229954')]:
    entry = next(e for e in stability['results'] if e['object'] == 'tennis_ball' and e['timestep'] == timestep)
    heights = [r['height'] for r in entry['rows']]
    peaks = [r['max_height_after_first_floor'] for r in entry['rows']]
    axes[1].plot(heights, peaks, '.-', color=color, label=f'dt = {timestep*1000:g} ms')
axes[1].plot(heights, heights, '--', color='#555555', label='Release height', linewidth=1)
axes[1].set(xlabel='Release center height (m)', ylabel='Maximum center height after floor contact (m)',
            title='Passive sphere: no robot, no initial spin', ylim=(0, 1.75))
axes[1].legend(frameon=False, fontsize=9)
axes[1].grid(alpha=.2)
for ax in axes:
    ax.spines[['top', 'right']].set_visible(False)
fig.suptitle('YCB diagnosis: integration correctness and bounce numerical stability', fontsize=14)
fig.text(.5, .015, '20 paired seeds per policy case, 2.5 s horizon. Original task metrics are not a unified strict-hold criterion.\n'
         'Passive-drop results diagnose numerics; they are not real-ball material calibration.',
         ha='center', fontsize=9, color='#444444')
fig.tight_layout(rect=[0, .095, 1, .94])
path = ARTIFACTS/'ycb_generalization_diagnosis_20261027.png'
fig.savefig(path, dpi=180, facecolor='white')
print(path)
