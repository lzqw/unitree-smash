"""Native-only Approach sampling labels. Never inputs to actor/model rollout."""
import torch

APPROACH_QUOTA = dict(uniform=24, approach_active=32, hit_prepare=24,
                     contact_release=24, approach_failure=16, post_hit=8)
FIELDS = ('approach_gate', 'racket_distance', 'had_contact', 'released',
          'qualified', 'frame_hit', 'attack')

def active_gate(gate, threshold=.25, minimum_batch_lanes=4):
    gate = gate.detach().reshape(-1)
    if not len(gate):return False
    return bool((gate > threshold).sum() >= min(minimum_batch_lanes, len(gate)))

@torch.no_grad()
def native_approach_metadata(row, actor, geometry):
    before, after = row['before'], row['after']
    result = {}
    for prefix, record in (('', before), ('next_', after)):
        task = record['task']; p = record['physical']
        n = len(p['joint_position']); device = p['joint_position'].device
        unknown = torch.full((n,), float('nan'), device=device)
        def field(name):
            return task[name].float() if name in task else unknown.clone()
        contact = field('legal_initial_contact') * field('string_hit') * field('airborne_string_hit')
        velocity = task.get('outgoing_velocity')
        attack = (contact * field('released') * (velocity.norm(dim=-1)>=5.) * (velocity[:,0]>0.)) if velocity is not None else unknown.clone()
        if prefix == '':
            obs = row['observation']; obs = obs['policy'] if isinstance(obs,dict) else obs
            result['approach_gate'] = actor.approach_gate(obs).reshape(-1).detach().cpu()
            kin = geometry(p['joint_position'],p['joint_velocity'],p['root_position_w'],
                           p['root_quaternion_w'],p['root_link_velocity_w'],p['root_angular_velocity_w'])
            result['racket_distance'] = (kin['position'][:,geometry.ids['Racket']]-p['ball_position_w']).norm(dim=-1).cpu()
        for name,value in dict(had_contact=field('had_contact'),released=field('released'),
                               qualified=contact,frame_hit=field('frame_hit'),attack=attack).items():
            result[prefix+name] = value.detach().cpu()
    return result

def approach_window_masks(metadata, valid, steps, *, gate_threshold=.25, min_gate_steps=4, near_distance=.30):
    """Unknown historical fields stay unknown; no terminal miss is invented.

    Near miss requires a *completed* same-episode suffix in replay. Before
    completion the close/no-hit sample remains hit_prepare, not a failure.
    """
    gate=metadata['approach_gate']; near=metadata['racket_distance']<=near_distance
    pre=(metadata['had_contact']<.5)
    def any_window(x): return x.unfold(0,steps,1).any(-1)&valid
    contact=(metadata['next_had_contact']>.5)&(metadata['had_contact']<.5)
    release=(metadata['next_released']>.5)&(metadata['released']<.5)
    illegal=((metadata['next_frame_hit']>.5)&(metadata['frame_hit']<.5)) | (contact&(metadata['next_qualified']<.5))
    bad_out=release & (metadata['next_qualified']>.5)&(metadata['next_attack']<.5)
    miss=metadata.get('completed_miss',torch.zeros_like(pre)) & near & pre
    return dict(approach_active=((gate>gate_threshold).unfold(0,steps,1).sum(-1)>=min(min_gate_steps,steps))&valid,
                hit_prepare=any_window(near & pre), contact_release=any_window(contact|release),
                approach_failure=any_window(illegal|bad_out|miss),post_hit=any_window(metadata['released']>.5),
                near_miss=any_window(miss),illegal_contact=any_window(illegal),invalid_outgoing=any_window(bad_out))

def masked_lane_mean(value, weights):
    weights=weights.detach().reshape(-1).to(value)
    value=value.reshape(len(weights),-1).mean(-1)
    return (value*weights).sum()/weights.sum().clamp_min(1.)

def gradient_parts(terms, parameters):
    """Diagnostics only; does not accumulate .grad or step an optimizer."""
    result={}; vectors={}
    for name,term in terms.items():
        grads=torch.autograd.grad(term,parameters,retain_graph=True,allow_unused=True) if term.requires_grad else [None]*len(parameters)
        vector=torch.cat([(g.detach() if g is not None else torch.zeros_like(p)).flatten() for p,g in zip(parameters,grads)])
        if not torch.isfinite(vector).all(): raise FloatingPointError('Nonfinite '+name+' gradient')
        vectors[name]=vector; result[name+'_gradient_norm']=float(vector.norm())
    a,b=vectors['task'],vectors['tracking']
    result['task_tracking_cosine']=float(torch.dot(a,b)/(a.norm()*b.norm()).clamp_min(1e-20))
    return result
