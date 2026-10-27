"""Plot actual recorded states and the validated known-model reconstruction."""
import hashlib
import json
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

HERE = Path(__file__).resolve().parent


def main():
    record = json.loads((HERE/'ACTUAL_MODEL_CHECK.json').read_text())
    arrays = HERE/'DIAGNOSTIC_ARRAYS.npz'
    assert hashlib.sha256(arrays.read_bytes()).hexdigest() == record['diagnostic_arrays_sha256']
    fig, axes = plt.subplots(2, 2, figsize=(12, 7), sharex=True, constrained_layout=True)
    descriptions = [('failed_source69','Failed teacher', '#c43d3d'),
                    ('recovered_exploration69','Recovered exploration', '#2379aa')]
    with np.load(arrays) as data:
        t = data['clock_s']
        for prefix, label, color in descriptions:
            angles = np.rad2deg(data[prefix+'_root_rpy_rad'])
            axes[0, 0].plot(t, angles[:, 0], label=label+' roll', color=color)
            axes[0, 0].plot(t, angles[:, 1], label=label+' pitch', color=color, linestyle='--', alpha=.7)
            axes[0, 1].plot(t, np.linalg.norm(data[prefix+'_root_angular_body_rad_s'], axis=1), label=label, color=color)
            axes[1, 0].plot(t, data[prefix+'_H_world'][:, 1], label=label, color=color)
            axes[1, 1].step(t, data[prefix+'_foot_contact'].sum(1), label=label, color=color, where='post')
    axes[0, 0].set(ylabel='Root roll / pitch (degrees)', ylim=(-100, 120))
    axes[0, 1].set(ylabel='Root angular speed (rad/s)', ylim=(0, 11))
    axes[1, 0].set(ylabel='World-Y angular momentum (kg m²/s)', ylim=(-4, 6))
    axes[1, 1].set(ylabel='Observed supporting feet', yticks=range(5), ylim=(-.2, 4.4))
    for ax in axes.flat:
        ax.axvspan(2.546, 3.051, color='#c1d3dc', alpha=.3, label='Contact-flight interval')
        ax.axvline(2.7257, color='#777777', linestyle=':', linewidth=1)
        ax.axvline(3.051, color='#777777', linestyle=':', linewidth=1)
        ax.set_xlim(2.4, 4.5)
        ax.grid(alpha=.2)
    axes[0, 0].legend(fontsize=8, loc='lower left')
    axes[0, 1].legend(fontsize=8)
    for ax in axes[1]:
        ax.set_xlabel('Actual episode clock (s)')
    fig.suptitle('Actual TRAIN69: recorded posture and public-state known-model momentum\nDifferent actions; neither episode is a complete smash. No new closed-loop execution.', fontsize=12)
    output = HERE/'RECOVERY_DIAGNOSTIC.png'
    fig.savefig(output, dpi=150)
    plt.close(fig)
    receipt = dict(figure_sha256=hashlib.sha256(output.read_bytes()).hexdigest(),
        program_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        actual_model_check_sha256=hashlib.sha256((HERE/'ACTUAL_MODEL_CHECK.json').read_bytes()).hexdigest(),
        source_arrays_sha256=record['diagnostic_arrays_sha256'], new_native_execution=False)
    with (HERE/'FIGURE_RECEIPT.json').open('x') as stream:
        stream.write(json.dumps(receipt,indent=2)+'\n')
    print(json.dumps(receipt))


if __name__ == '__main__':
    main()
