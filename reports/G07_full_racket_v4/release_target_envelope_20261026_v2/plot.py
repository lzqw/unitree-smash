"""Standalone scientific plot of recorded landings and CPU-only planning."""
from datetime import datetime,timezone
import hashlib
import json
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2]
MAIN = ROOT.parent/'unitree_smash'
RUN = MAIN/'reports/G07_full_racket_v4/residual_ppo_centered_leg06_after075_intermediates_n2048_probe8_fixed_train128_01'


def read(path): return json.loads(path.read_text())
def sha(path): return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    result,sensitivity = read(HERE/'RESULT.json'),read(HERE/'EARLIER_POINT_RESULT.json')
    rows = read(RUN/'episodes.json');p = HERE/'Release_target_geometry_20261026.png'; assert not p.exists()
    proposals = result['proposals']; labels = [x['TRAIN_index'] for x in proposals]
    y = np.arange(len(labels))
    actual = np.asarray([rows[i]['first_landing_m'][0] for i in labels])
    hypothetical = np.asarray([x.get('refined_landing_m',[np.nan])[0] for x in proposals])
    fig,(left,right) = plt.subplots(1,2,figsize=(12,7),gridspec_kw={'width_ratios':[1.45,1]})
    left.axvspan(2,4,color='#e3eee4',label='Original target x interval')
    left.scatter(actual,y,color='#292d35',marker='x',s=28,label='Recorded native first landing')
    left.scatter(hypothetical,y,color='#c57829',marker='o',s=23,label='CPU candidate at recorded contact')
    for a,b,c in zip(actual,hypothetical,y):
        if np.isfinite(a) and np.isfinite(b): left.plot([a,b],[c,c],color='#c2b4a1',linewidth=.7,zorder=0)
    left.set_yticks(y,labels);left.invert_yaxis();left.set_ylabel('Original TRAIN index')
    left.set_xlabel('First landing x (m)');left.grid(axis='x',alpha=.2)
    left.set_title('Actual landings and proposed flight directions')
    left.legend(loc='upper left',fontsize=8)
    variants = sensitivity['fixed8_01_sensitivity_variants'];counts = [x['CPU_geometric_witness_count'] for x in variants]
    names = ['At recorded\ncontact position','At last earlier\nball sample','One more sample\nearlier']
    bars = right.bar(np.arange(3),counts,width=.65,color=['#627f9d','#7793ad','#8ca8bc'])
    right.bar_label(bars,labels=[f'{n}/27' for n in counts],padding=4)
    right.set_ylim(0,30);right.set_xticks(np.arange(3),names,fontsize=9)
    right.set_ylabel('Contacts with a CPU geometric candidate')
    right.set_title('Hypothetical outgoing speed 8.01 m/s')
    right.grid(axis='y',alpha=.2);right.set_axisbelow(True)
    fig.suptitle('CPU flight planning from 27 recorded airborne contacts',fontsize=14)
    fig.text(.5,.025,'No new robot execution. Reachability, racket impact and recovery remain untested. Native complete: 0/128.',ha='center',fontsize=10)
    fig.subplots_adjust(bottom=.12,top=.9,wspace=.36)
    fig.savefig(p,dpi=180);plt.close(fig)
    source = {str(path):sha(path) for path in [Path(__file__),HERE/'RESULT.json',HERE/'EARLIER_POINT_RESULT.json',RUN/'episodes.json']}
    (HERE/'PLOT_RECEIPT.json').write_text(json.dumps(dict(created_at=datetime.now(timezone.utc).isoformat(),plot=str(p),
        sha256=sha(p),source_sha256=source,new_native_execution=False,formal_acceptance=False),indent=2)+'\n')
    print(str(p),flush=True)


if __name__ == '__main__': main()
