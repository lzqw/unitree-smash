"""Read-only task recurrence; true physical inputs isolate task-model error.

Native future task values are used only for evaluation, not state propagation.
Every rollout stops at the first terminal or episode change, including the
terminal transition. This is not full-world prediction or task performance.
"""
import argparse
import hashlib
import json
from pathlib import Path
import torch
from quadsmash.rl.supermimic_task_memory_schema_v1 import TaskMemorySchemaV1,MemoryField
from quadsmash.rl.supermimic_task_transition_model_v1 import task_model_for_schema
from quadsmash.rl.supermimic_event_hazard_v1 import advance_event_latches
from quadsmash.rl.supermimic_physical_dataset_v1 import decode_physical
from quadsmash.rl.supermimic_physical_imagination_adapter_v1 import matrix_quaternion
from quadsmash.rl.supermimic_physical_world_v2 import PhysicalWorldV2


def adapter_feedback(schema,values,validity,old_values,physical,action,previous_action,
                     present,elapsed,sim_tick,geometry,launch_s,dt,components):
    """Mirror existing adapter causal overwrites and decode/encode feedback.

    Native next physical state is the declared teacher-forced input here.
    No native future task value, timing, or event label is read.
    """
    slices={};offset=0;pieces={};masks={}
    for f in schema.fields:
        sl=slice(offset,offset+f.width);offset+=f.width;slices[f.path]=sl
        piece=values[:,sl];mask=validity[:,sl]
        if 'decode' not in components:pass
        elif f.kind=='float':
            valid=mask>.5
            piece=torch.where(valid,piece,torch.zeros_like(piece));mask=valid.float()
        else:
            mask=torch.ones_like(mask)
            if f.kind=='bool':piece=piece.clamp(0.,1.)
        pieces[f.path]=piece;masks[f.path]=mask
    def get(name,source=None):
        key='auxiliary/environment/'+name
        return pieces[key] if source is None else source[:,slices[key]]
    def put(name,value):
        key='auxiliary/environment/'+name
        pieces[key]=value.reshape(len(values),-1);masks[key]=torch.ones_like(pieces[key])
    def encoded():return torch.cat([pieces[f.path] for f in schema.fields],-1),torch.cat([masks[f.path] for f in schema.fields],-1)
    if 'command' in components:
        put('last_action',action)
        put('action_delta',(action-previous_action).square().sum(1))
    if 'clock' in components:
        for name in ('jump_clock','ball_clock'):put(name,get(name,old_values)+dt)
    active=present.bool()|(elapsed.flatten()>=launch_s-1e-6)
    launched=(~present.bool())&active
    if 'launch' in components:
        put('launch_done',active)
        put('launch_frame',torch.where(launched.reshape(-1,1),sim_tick.reshape(-1,1),get('launch_frame')))
        put('launch_signals_ready',torch.where(launched.reshape(-1,1),torch.ones_like(get('launch_signals_ready')),get('launch_signals_ready')))
    if 'geometry' not in components:return encoded()
    p=decode_physical(physical)
    g=geometry(p['joint_position'],p['joint_velocity'],p['root_position_w'],p['root_quaternion_w'],p['root_link_velocity_w'],p['root_angular_velocity_w'])
    racket=geometry.ids['Racket'];quaternion=matrix_quaternion(g['rotation'][:,racket])
    sign=(quaternion*get('previous_tool_quaternion',old_values)).sum(1,keepdim=True)>=0
    put('previous_tool_quaternion',torch.where(sign,quaternion,-quaternion))
    put('previous_tool_position',g['position'][:,racket])
    put('previous_ball_position',p['ball_position_w']);put('previous_ball_velocity',p['ball_velocity_w'])
    return encoded()


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--checkpoint',type=Path,required=True)
    parser.add_argument('--output',type=Path,required=True)
    parser.add_argument('--adapter-memory-feedback',action='store_true')
    parser.add_argument('--feedback-components',default='decode,clock,command,launch,geometry')
    args=parser.parse_args()
    if args.output.exists():raise FileExistsError(args.output)
    components=args.feedback_components.split(',')
    if set(components)-{'decode','clock','command','launch','geometry'}:raise ValueError('Unknown diagnostic component')
    torch.set_num_threads(2)
    root=Path('/home/autodrive/LZQW/experiments/supermimic_admission256_20261118_s5lbit6k')
    package=torch.load('/home/autodrive/LZQW/experiments/supermimic_frozen_closure_20261117_223243/packed/data.pt',map_location='cpu',weights_only=True)
    d=package['tensors']
    schema=TaskMemorySchemaV1([MemoryField(**dict(f,shape=tuple(f['shape']))) for f in package['task_schema']['fields']])
    checkpoint=torch.load(args.checkpoint,map_location='cpu',weights_only=True)
    model=task_model_for_schema(checkpoint['schema'],schema,d['torque'].shape[-1])
    model.load_state_dict(checkpoint['model'],strict=True)
    model.eval()
    physical=PhysicalWorldV2(package['source_manifest']['task_contract']['asset']['sha256']).eval() if args.adapter_memory_feedback else None
    dt=package['source_manifest']['task_contract']['control_dt_s']
    substeps=round(dt/package['source_manifest']['task_contract']['physics_dt_s'])
    lanes=json.loads((root/'fixed_split.json').read_text())['validation_lanes']
    names=('jump','hit','cross','target','recovery','body','frame','fault')
    results={}
    with torch.no_grad():
        for lead in (1,16,32):
            samples=[]
            for event,name in enumerate(names):
                for lane in lanes:
                    edges=(d['next_event_state'][:,lane,event]>d['event_state'][:,lane,event]).nonzero().flatten()
                    for tick in edges.tolist():
                        start=max(0,tick-lead)
                        episode=d['episode'][tick,lane]
                        while start<tick and d['episode'][start,lane]!=episode:start+=1
                        end=start
                        while end<min(start+64,len(d['state'])) and d['episode'][end,lane]==episode:
                            end+=1
                            if d['terminated'][end-1,lane] or d['timeout'][end-1,lane]:break
                        if not start<=tick<end:raise RuntimeError('Event lies beyond episode-safe window')
                        samples.append((start,lane,end,tick,event))
            t,l,end,event_tick,event_id=torch.tensor(samples).unbind(1)
            values=d['values'][t,l];validity=d['valid'][t,l];paid=d['event_state'][t,l]
            predictions=[];native=[];masks=[]
            for step in range(64):
                active=t+step<end;at=torch.minimum(t+step,end-1)
                result=model.forward_values(d['state'][at,l],d['next_state'][at,l],d['targets'][at,l],d['previous_action'][at,l],d['feet'][at,l],d['elapsed'][at,l],values,validity)
                next_values=result['numeric'];next_valid=result['valid_logits'].sigmoid()
                if args.adapter_memory_feedback:
                    next_values,next_valid=adapter_feedback(schema,next_values,next_valid,values,d['next_state'][at,l],
                        d['final_action'][at,l],d['previous_action'][at,l],d['present'][at,l],d['elapsed'][at,l],
                        at*substeps+package['template']['auxiliary']['timing']['_sim_step_counter'],physical.geometry,physical.launch_s,dt,components)
                values=torch.where(active[:,None],next_values,values)
                validity=torch.where(active[:,None],next_valid,validity)
                paid=torch.where(active[:,None],advance_event_latches(paid,result['hazards']),paid)
                predictions.append(paid.clone());native.append(d['next_event_state'][at,l]);masks.append(active)
            predicted=torch.stack(predictions);truth=torch.stack(native);valid=torch.stack(masks)
            rows=[]
            for i,(start,lane,stop,tick,event) in enumerate(samples):
                probability=predicted[:stop-start,i,event]
                crossings=(probability>=.5).nonzero().flatten()
                first=start+int(crossings[0]) if len(crossings) else None
                paid_start=d['event_state'][start,lane]
                eligible=(paid_start==0)[None]&valid[:,i,None]
                false_paid=(predicted[:,i]>=.5)&(truth[:,i]<.5)&eligible
                rows.append(dict(event=names[event],lane=lane,episode=int(d['episode'][start,lane]),start=start,stop=stop,
                    true_tick=tick,predicted_tick=first,timing_error=None if first is None else first-tick,
                    probability_at_event=float(probability[tick-start]),probability_at_end=float(probability[-1]),
                    false_paid_lane_event_steps=int(false_paid.sum()),
                    complete_native_terminal_included=bool(d['terminated'][stop-1,lane] or d['timeout'][stop-1,lane])))
            summary={}
            for name in names:
                selected=[r for r in rows if r['event']==name]
                errors=[r['timing_error'] for r in selected if r['timing_error'] is not None]
                summary[name]=dict(count=len(selected),missed=sum(r['predicted_tick'] is None for r in selected),
                    exact=sum(e==0 for e in errors),early=sum(e<0 for e in errors),late=sum(e>0 for e in errors),
                    mean_probability_at_event=sum(r['probability_at_event'] for r in selected)/len(selected) if selected else None)
            results[str(lead)]=dict(summary=summary,rows=rows)
    output=dict(scope=__doc__,checkpoint=str(args.checkpoint),checkpoint_sha256=hashlib.sha256(args.checkpoint.read_bytes()).hexdigest(),
        task_model_schema=model.schema,heldout_lanes=lanes,actor_updates=0,model_updates=0,diagnostic_threshold=.5,adapter_memory_feedback=args.adapter_memory_feedback,
        feedback_components=components if args.adapter_memory_feedback else [],
        limitation='Event-centered windows overlap; counts are not independent episodes or a general false-positive rate.',results=results)
    args.output.write_text(json.dumps(output,indent=2)+'\n')
    print(json.dumps({lead:r['summary'] for lead,r in results.items()},indent=2))


if __name__=='__main__':main()
