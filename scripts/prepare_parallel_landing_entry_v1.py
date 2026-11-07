#!/usr/bin/env python3
"""Prepare, but never install, a lane-batched native evaluator patch."""
import argparse
import ast
import difflib
import hashlib
import json
from pathlib import Path


def transform(source):
    def once(old, new):
        nonlocal source
        if source.count(old) != 1:
            raise ValueError('Entry shape changed: '+old[:100])
        source = source.replace(old, new, 1)

    once("parser.add_argument('--landing-mpc-command-change-weight'",
         "parser.add_argument('--landing-mpc-solver-workers', type=int, choices=range(9), default=0)\n"
         "parser.add_argument('--landing-mpc-command-change-weight'")
    once("        for lane in torch.where(active)[0].tolist():\n            case=int(lane)",
         "        pool = getattr(self, '_landing_solver_pool', None)\n"
         "        def advance_lane(lane):\n            case=int(lane)")
    once("            if case not in self.landing_mpc_cases:continue",
         "            if case not in self.landing_mpc_cases:return")
    once("                if isinstance(self.landing_mpc, LandingSupportMPCV3):\n"
         "                    self.landing_mpc.observe(state, joint_ref, lane=lane,",
         "                if pool is None and isinstance(self.landing_mpc, LandingSupportMPCV3):\n"
         "                    self.landing_mpc.observe(state, joint_ref, lane=lane,")
    once("                    continue\n                if release_for_mode_mismatch or release_for_tracking:",
         "                    return\n                if release_for_mode_mismatch or release_for_tracking:")
    once("                if isinstance(self.landing_mpc, LandingSupportMPCV3):\n"
         "                    plan=self.landing_mpc.solve(state,joint_ref,force=force_post_touchdown,",
         "                if pool is not None:\n"
         "                    plan = yield dict(operation='observe', lane=int(lane), state=state,\n"
         "                        reference=joint_ref, control_tick=control_tick, solve=True,\n"
         "                        force=force_post_touchdown)\n"
         "                elif isinstance(self.landing_mpc, LandingSupportMPCV3):\n"
         "                    plan=self.landing_mpc.solve(state,joint_ref,force=force_post_touchdown,")
    once("                        else 'public_observation_reconstruction'),native_execution=True))\n        return action",
         "                        else 'public_observation_reconstruction'),native_execution=True))\n"
         "        pending = []\n"
         "        for lane in torch.where(active)[0].tolist():\n"
         "            continuation = advance_lane(lane)\n"
         "            try:\n"
         "                request = next(continuation)\n"
         "            except StopIteration:\n"
         "                continue\n"
         "            pending.append((continuation, request))\n"
         "        if pending:\n"
         "            # Pool errors abort the run; do not silently issue a fallback.\n"
         "            plans = pool.batch([request for _, request in pending])\n"
         "            for (continuation, _), plan in zip(pending, plans):\n"
         "                try:\n"
         "                    continuation.send(plan)\n"
         "                except StopIteration:\n"
         "                    continue\n"
         "                raise RuntimeError('A native lane requested more than one solve per tick')\n"
         "        return action")
    once("                    self.landing_mpc.reset(lane=int(lane))\n\n    def effort_override",
         "                    self.landing_mpc.reset(lane=int(lane))\n"
         "        pool = getattr(self, '_landing_solver_pool', None)\n"
         "        if pool is not None:\n"
         "            lanes = self.landing_mpc_cases if ids is None else selected\n"
         "            pool.batch([dict(operation='reset', lane=int(lane)) for lane in lanes])\n\n"
         "    def effort_override")
    once("                self.landing_mpc.reset(lane=lane)\n\n    def set_landing_mpc_native_state",
         "                self.landing_mpc.reset(lane=lane)\n"
         "        pool = getattr(self, '_landing_solver_pool', None)\n"
         "        if pool is not None:\n"
         "            pool.batch([dict(operation='reset', lane=int(lane))\n"
         "                        for lane in torch.where(newly)[0].tolist()])\n\n"
         "    def set_landing_mpc_native_state")
    once("            controller.configure_landing_mpc(limits, default)\n",
         "            controller.configure_landing_mpc(limits, default)\n"
         "        if args.landing_mpc_solver_workers:\n"
         "            from scripts.native_parallel_landing_configuration_v1 import attach_parallel_solver\n"
         "            attach_parallel_solver(controller, args.landing_mpc_solver_workers)\n")
    once("                command_change_weight=float(args.landing_mpc_command_change_weight),\n",
         "                command_change_weight=float(args.landing_mpc_command_change_weight),\n"
         "                solver_workers=int(args.landing_mpc_solver_workers),\n")
    once("    finally:\n        env._diagnostic_observer=None\n        env.close()",
         "    finally:\n"
         "        pool = getattr(controller, '_landing_solver_pool', None)\n"
         "        if pool is not None:pool.close()\n"
         "        env._diagnostic_observer=None\n        env.close()")
    compile(source, '<parallel-native-preview>', 'exec')
    tree=ast.parse(source)
    method=next(n for n in ast.walk(tree) if isinstance(n,ast.FunctionDef) and n.name=='apply_landing_mpc')
    assert sum(isinstance(n,ast.Yield) for n in ast.walk(method))==1
    return source


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--source',type=Path,required=True)
    p.add_argument('--output',type=Path,required=True)
    a=p.parse_args()
    if a.output.exists():raise ValueError('Preserve earlier preparation')
    old=a.source.read_text();new=transform(old)
    a.output.mkdir(parents=True)
    (a.output/'proposed_evaluator.py').write_text(new)
    target='scripts/evaluate_shared_target_ground64_persistent_recovery_v1.py'
    patch=''.join(difflib.unified_diff(old.splitlines(True),new.splitlines(True),
        fromfile='a/'+target,tofile='b/'+target,n=1))
    # Git accepts a bare empty context line; avoid whitespace-only file lines.
    (a.output/'parallel_entry.patch').write_text(patch.replace('\n \n','\n\n').rstrip()+'\n')
    (a.output/'preparation.json').write_text(json.dumps(dict(installed=False,
        source=str(a.source.resolve()),before_sha256=hashlib.sha256(old.encode()).hexdigest(),
        after_sha256=hashlib.sha256(new.encode()).hexdigest()),indent=2)+'\n')


if __name__=='__main__':main()
