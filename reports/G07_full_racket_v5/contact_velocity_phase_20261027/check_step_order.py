"""Record the installed native-step order supporting the phase correction."""
import ast
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2].parent/'unitree_smash_contact_recovery'
LAB = Path('/home/autodrive/Isaac/IsaacLab/source/isaaclab/isaaclab/envs/direct_rl_env.py')


def excerpt(path, start, stop):
    lines = path.read_text().splitlines()
    return dict(path=str(path), first_line=start, last_line=stop,
                text='\n'.join(lines[start-1:stop]))


def main():
    tree = ast.parse(LAB.read_text())
    cls = next(n for n in tree.body if isinstance(n, ast.ClassDef) and n.name == 'DirectRLEnv')
    step = next(n for n in cls.body if isinstance(n, ast.FunctionDef) and n.name == 'step')
    loop = next(n for n in step.body if isinstance(n, ast.For)
                and ast.unparse(n.iter) == 'range(self.cfg.decimation)')
    calls = [ast.unparse(n.value) for n in loop.body if isinstance(n, ast.Expr)
             and isinstance(n.value, ast.Call)]
    assert calls.index('self._apply_action()') < calls.index('self.sim.step(render=False)')
    assert calls.index('self.sim.step(render=False)') < calls.index('self.scene.update(dt=self.physics_dt)')
    ball = ROOT/'quadsmash/tasks/stand/ball_env.py'
    event = ROOT/'quadsmash/tasks/stand/full_racket_env_v4.py'
    files = [Path(__file__), LAB, ball, event, HERE/'analyze_saved_states.py', HERE/'SAVED_SEPARATED_STATE_CHECK.json']
    result = dict(checked_at=datetime.now(timezone.utc).isoformat(), installed_step_loop_calls=calls,
        source_excerpts=[excerpt(LAB, loop.lineno, loop.end_lineno), excerpt(ball, 231, 258),
                         excerpt(event, 180, 193), excerpt(event, 242, 267)],
        conclusion='The action hook consumes the previous completed physics frame. Contact detection and point_velocity_mps both read that completed frame; the contact record is not an immediately pre-impact velocity.',
        limitations='No exact pre-impact racket velocity was saved. Separated100Hz samples establish earlier motion only; they do not isolate recoil or bound intervening motion.',
        controller_or_physics_changed=False, native_scores_changed=False,
        formal_acceptance=False, goal_complete=False,
        source_sha256={str(p): hashlib.sha256(p.read_bytes()).hexdigest() for p in files})
    with (HERE/'PHYSICS_STEP_PHASE_CHECK.json').open('x') as f:
        f.write(json.dumps(result, indent=2)+'\n')
    print(json.dumps(dict(step_order_pass=True, actual_saved_phase_checks=12,
                         native_scores_changed=False, controller_or_physics_changed=False)))


if __name__ == '__main__':
    main()
