#!/usr/bin/env python3
"""Plot predicted flight targets with explicit unverified-dynamics labels."""
import argparse
import json
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument('--input', type=Path, required=True)
parser.add_argument('--output', type=Path, required=True)
args = parser.parse_args()
data = json.loads(args.input.read_text())
rows = data['table']
fig, ax = plt.subplots(figsize=(8, 4.5), constrained_layout=True)
for cap in (8.5, 10.):
    key = f'min_grid_height_at_speed_le_{cap:g}'
    ax.plot([r['distance_to_net_m'] for r in rows], [r[key] for r in rows], 'o-',
            label=f'Outgoing speed 8 to {cap:g} m/s; vz <= -1 m/s')
ax.axhline(1.549, color='gray', linestyle='--', label='Net top + shuttle radius')
ax.set(xlabel='Hit-to-net distance (m)', ylabel='Lowest feasible sampled hit height (m)',
       title='Locked drag model: net clearance and first landing in x=2..4 m')
ax.text(.02, .96, 'CPU free-flight prediction, 2 cm height grid\nNo robot/collision/recovery feasibility claim',
        transform=ax.transAxes, va='top', fontsize=9)
ax.legend(loc='lower right', fontsize=8)
ax.grid(alpha=.2)
args.output.parent.mkdir(parents=True, exist_ok=True)
fig.savefig(args.output, dpi=180)
plt.close(fig)
