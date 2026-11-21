"""Keep a fixed independent native forward score across online WM optimizer steps."""
import json
from pathlib import Path
import torch

from scripts.evaluate_supermimic_fresh_wm_panel import assess, check_panel, digest
from scripts.rescore_supermimic_fresh_wm_panel import body_forward_check
from quadsmash.rl.supermimic_physical_loop_loading_v1 import tensor_digest
from quadsmash.rl.supermimic_wm_candidate_gate_v1 import compare_candidate_to_anchor


def event_metrics(metrics):
    return {'event_aligned':{str(offset):{str(h):groups[f'contact{offset:+d}']
        for h,groups in metrics.items() if f'contact{offset:+d}' in groups} for offset in (-1,0,1)}}


def compare_forward(initial, candidate, horizons):
    m=dict(anchor=initial,candidate=candidate)
    body=body_forward_check(m,horizons,['body_rmse','root_angular_mae'])
    flight=body_forward_check(m,[64],['ball_velocity_rmse'])
    contact=compare_candidate_to_anchor(event_metrics(initial),event_metrics(candidate),
                                        require_contact_improvement=False)
    return dict(passed=body['passed'] and flight['passed'] and contact['passed'],
                body=body,flight=flight,contact=contact)


class OnlineForwardGuard:
    def __init__(self, spec_path, adapter, task_contract):
        spec_path=Path(spec_path)
        def read(path):
            return json.loads(path.read_text())
        spec=read(spec_path)
        def artifact(name):
            row=spec[name]
            path=spec_path.parent/row['path']
            if digest(path)!=row['sha256']:
                raise ValueError('Changed online forward guard artifact: '+name)
            return path
        self.reference_policy=spec.get('comparison_reference','block_initial')
        if self.reference_policy not in ('block_initial','formal_anchor'):
            raise ValueError('Unknown forward comparison reference')
        self.spec_sha256=digest(spec_path)
        self.panel=read(artifact('panel'))
        capture=artifact('capture_metrics')
        native=capture.parent/'native.pt'
        recorded=read(capture)
        reference=read(artifact('reference_score'))
        root=Path(__file__).resolve().parents[1]
        weights=[Path(p) for p in self.panel['weights']]
        check_panel(self.panel,root/'configs/g07_full_racket_eval_v4.json',root,weights)
        behavior=(spec_path.parent/spec['behavior_checkpoint']).resolve()
        if (str(behavior) not in self.panel['weights'] or recorded['weights']!=self.panel['weights']
            or reference['weights']!=recorded['weights'] or reference['transition']!='initial_components'
            or not reference['prospective_forward_passed'] or reference['contact_input']!='autoregressive_robot'
            or reference['runtime_world_sha256']!=tensor_digest(adapter.state_dict().items())
            or recorded['panel_sha256']!=spec['panel']['sha256'] or reference['panel_sha256']!=recorded['panel_sha256']
            or reference['data_sha256']!=recorded['data_sha256'] or digest(native)!=recorded['data_sha256']
            or reference['windows']!=recorded['windows'] or recorded['ticks']!=800
            or recorded['first_episodes']!=len(self.panel['cases'])
            or self.panel['horizons']!=[1,2,4,8,16,32,64]):
            raise ValueError('Unverified deployed WM, incomplete native forward reference, or changed panel')
        actor=torch.load(behavior,weights_only=False,mmap=True,map_location='cpu')
        if (actor['metadata']['task_contract']!=task_contract
            or tensor_digest(actor['actor'].items())!=recorded['actor_sha256']):
            raise ValueError('Held-out native task or collecting policy identity differs')
        self.data=torch.load(native,map_location='cpu',weights_only=True,mmap=True)
        self.windows=recorded['windows']
        # Recompute the reference using the training backend. CPU/GPU roundoff
        # must not penalize unchanged parameters at strict adjacent-tick bounds.
        self.initial=self.score(adapter)
        formal={int(h):v for h,v in reference['metrics']['anchor'].items()}
        self.formal=formal
        declared=self.panel['prospective_body_comparison']['uniform_no_regression']
        if declared!=['body_rmse','root_angular_mae']:
            raise ValueError('Unrecognized native forward bounds')
        body=body_forward_check(dict(anchor=formal,candidate=self.initial),self.panel['horizons'],declared)
        contact=compare_candidate_to_anchor(event_metrics(formal),event_metrics(self.initial))
        if not body['passed'] or not contact['passed']:
            raise ValueError('The deployed initial weights do not pass their native forward gate on this backend')

    @torch.no_grad()
    def score(self, adapter):
        device=next(adapter.parameters()).device
        result=assess(self.data,self.panel['cases'],None,adapter,self.panel['horizons'],device,predictors=('candidate',))
        if result['windows']!=self.windows:
            raise ValueError('Native launch/control windows changed')
        return result['metrics']['candidate']

    def check(self, adapter):
        current=self.score(adapter)
        deployed=compare_forward(self.initial,current,self.panel['horizons'])
        formal=compare_forward(self.formal,current,self.panel['horizons'])
        # The registered formal panel requires the candidate to improve the
        # contact tick as well as retain its adjacent-tick accuracy.
        formal['contact']=compare_candidate_to_anchor(event_metrics(self.formal),event_metrics(current))
        formal['passed']=formal['body']['passed'] and formal['flight']['passed'] and formal['contact']['passed']
        comparison=formal if self.reference_policy=='formal_anchor' else deployed
        return dict(comparison,comparison_reference=self.reference_policy,vs_formal_anchor=formal,
                    vs_block_initial=deployed,spec_sha256=self.spec_sha256,
                    world_sha256=tensor_digest(adapter.state_dict().items()))


def main():
    import argparse
    import os
    parser=argparse.ArgumentParser(description='Bind the online WM check to an existing native panel, capture, and initial score')
    for name in ('panel','capture-metrics','reference-score','behavior-checkpoint','output'):
        parser.add_argument('--'+name,type=Path,required=True)
    parser.add_argument('--comparison-reference',choices=('formal_anchor','block_initial'),default='formal_anchor')
    args=parser.parse_args()
    folder=args.output.parent.resolve()
    files={name:getattr(args,name).resolve(strict=True) for name in ('panel','capture_metrics','reference_score')}
    spec={name:dict(path=os.path.relpath(path,folder),sha256=digest(path)) for name,path in files.items()}
    spec['comparison_reference']=args.comparison_reference
    spec['behavior_checkpoint']=str(args.behavior_checkpoint.resolve(strict=True))
    with args.output.open('x') as out:
        json.dump(spec,out,indent=2)
        out.write('\n')


if __name__=='__main__':
    main()
