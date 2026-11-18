from pathlib import Path
p=Path('scripts/train_supermimic_physical_block.py');s=p.read_text();out=Path(__file__).parent
helper='''def scale_wm_quotas(quotas, batch_size):
    """CLI quotas retain their documented128-window unit; scale ratios exactly.

    This only changes supervised model minibatch size, never native lanes or
    imagination/world temporal horizons. No implicit OOM retry/ratio change.
    """
    if type(batch_size) is not int or batch_size<128 or batch_size%128:
        raise ValueError('WM batch must be a positive multiple of128')
    if not quotas or any(type(v) is not int or v<0 for v in quotas.values()) or sum(quotas.values())!=128:
        raise ValueError('WM quota units must sum to128 before batch scaling')
    return {k:v*(batch_size//128) for k,v in quotas.items()}


'''
s=s.replace('def select_launch_for_start(',helper+'def select_launch_for_start(',1)
s=s.replace("    parser.add_argument('--wm-source-quota',", "    parser.add_argument('--wm-batch-size',type=int,default=128,help='Supervised WM minibatch, multiple of128; native environments remain128')\n    parser.add_argument('--wm-source-quota',",1)
s=s.replace("    if args.wm_recent_policy_window < 0", "    if args.wm_batch_size < 128 or args.wm_batch_size%128: parser.error('wm-batch-size must be a positive multiple of128')\n    if args.wm_recent_policy_window < 0",1)
s=s.replace('native_ticks_per_proposal=args.native_ticks_per_actor_update,world_updates_per_proposal=args.world_updates_per_actor_update,','native_ticks_per_proposal=args.native_ticks_per_actor_update,world_updates_per_proposal=args.world_updates_per_actor_update,wm_batch_size=args.wm_batch_size,',1)
s=s.replace("        if sum(args.wm_source_quota.values())!=128:raise ValueError('WM source quotas must sum to the fixed batch size 128')\n",'')
old="        args.wm_stage_quota=parse_quota(args.wm_stage_quota,allowed={'uniform','pre_touchdown','touchdown','post_touchdown','recovery_failure','stable_recovery'},name='wm-stage-quota')"
s=s.replace(old,old+"\n        for field in ('wm_source_quota','wm_stage_quota','wm_approach_stage_quota'):\n            setattr(args,field,scale_wm_quotas(getattr(args,field),args.wm_batch_size))",1)
s=s.replace('            wm_source_quota=args.wm_source_quota,wm_stage_quota=args.wm_stage_quota,','            wm_batch_size=args.wm_batch_size,wm_source_quota=args.wm_source_quota,wm_stage_quota=args.wm_stage_quota,',1)
s=s.replace('timebase.world_steps,128,iteration*total+offset,','timebase.world_steps,args.wm_batch_size,iteration*total+offset,',1)
s=s.replace("writer.add_scalar('wm/source/'+source,float(value)/128.,step)","writer.add_scalar('wm/source/'+source,float(value)/args.wm_batch_size,step)",1)
(out/p.name).write_text(s)
p=Path('scripts/run_supermimic_phase1_long_v1.sh');s=p.read_text()
s=s.replace('    --world-updates-per-actor-update 4','    --world-updates-per-actor-update 4 --wm-batch-size "${SM_WM_BATCH_SIZE:-512}"',1)
(out/p.name).write_text(s)
print('Staged batch configuration; production/running source unchanged')
