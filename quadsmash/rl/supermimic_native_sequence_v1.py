"""Read-only packed input trace for native task-state substep replay.

Only wraps existing calls, invokes each original exactly once, and records its
actual inputs. No history/observer/reward call is introduced. This does not
capture a PhysX cache or claim that model events are real contacts.
"""
import inspect
from types import SimpleNamespace
import torch

JUMP_INPUTS = ('flight_candidate', 'takeoff_time', 'flight_apex_rise',
               'flight_clear_time', 'valid_jump_landed', 'flight_qualified',
               'landing_time', 'body_ground_seen')


class NativeTaskSequenceV1:
    schema = 'native-task-call-sequence-v1'

    def __init__(self, env):
        self.env = env
        self.capture_reference_targets = False
        self.capacity = int(env.cfg.decimation) + 8
        self.active = False
        self.streams, self.order = {}, []
        self.originals = []
        for obj, method, label in (
            (env.contact_recovery, 'metric', 'recovery.metric'),
            (env.contact_recovery, 'contact', 'recovery.contact'),
            (env.rally, 'advance', 'rally.advance'),
            (env.rally, 'launch', 'rally.launch'),
            (env.contact_events, 'update', 'contact.update')):
            original = getattr(obj, method)
            signature = inspect.signature(original)
            def wrapped(*args, _original=original, _signature=signature, _label=label, _obj=obj, **kwargs):
                if self.active:
                    bound = _signature.bind(*args, **kwargs)
                    bound.apply_defaults()
                    values = dict(bound.arguments)
                    jump = values.pop('jump', None)
                    if jump is not None:
                        values.update({'jump/' + k: getattr(jump, k) for k in JUMP_INPUTS})
                    if _label == 'rally.advance':
                        # FullRacketEnv sets in_play when launch signals become
                        # ready, outside RallyRules.advance. Record that input.
                        values['external_in_play'] = _obj.in_play
                    if self.capture_reference_targets and _label == 'recovery.contact':
                        self.record_racket_target(values)
                    self.record(_label, values)
                return _original(*args, **kwargs)
            self.originals.append((obj, method, original))
            setattr(obj, method, wrapped)

    def begin(self):
        if self.active:
            raise RuntimeError('Unfinished native task trace')
        self.streams, self.order = {}, []
        self.reference_targets = []
        if self.capture_reference_targets:
            self.start_frame = self.env._sim_step_counter
        self.active = True

    @torch.no_grad()
    def record_racket_target(self, values):
        env = self.env
        ids = torch.where(values['legal_air_strings_touching'] &
                          (values['time'] == values['rally_contact_time']))[0]
        if not len(ids):
            return
        frame = env._last_physics_frame - self.start_frame
        if not 1 <= frame <= env.cfg.decimation:
            raise ValueError('Native contact outside current control interval')
        body = env.robot.data
        racket = env.racket_body_id
        features = torch.cat((body.body_link_pos_w[ids, racket] - env.shuttle.data.root_pos_w[ids],
                              body.body_link_lin_vel_w[ids, racket], env.first_tool_normal[ids],
                              body.body_link_ang_vel_w[ids, racket]), -1)
        self.reference_targets.append(dict(lanes=ids, episode=env.episode_index[ids], features=features,
                                           substep=frame, offset_s=frame*env.physics_dt))

    @torch.no_grad()
    def record(self, name, values):
        stream = self.streams.setdefault(name, dict(count=0, tensors={}, literals={}))
        index = stream['count']
        if index >= self.capacity:
            raise RuntimeError('Unexpected native task calls per control tick: ' + name)
        for key, value in values.items():
            if isinstance(value, torch.Tensor):
                if key not in stream['tensors']:
                    stream['tensors'][key] = torch.empty((self.capacity, *value.shape),
                                                        dtype=value.dtype, device=value.device)
                stream['tensors'][key][index].copy_(value)
            elif value is None or type(value) in (int, float, bool):
                stream['literals'].setdefault(key, []).append(value)
            else:
                raise TypeError('Unsupported native trace input: ' + name + '/' + key)
        stream['count'] += 1
        self.order.append((name, index))

    def finish(self):
        if not self.active:
            raise RuntimeError('Missing native task trace begin')
        self.active = False
        extra = {'racket_reference_targets': self.reference_targets} if self.capture_reference_targets else {}
        return dict(schema=self.schema, order=self.order, **extra,
            streams={name: dict(count=s['count'], literals=s['literals'],
                tensors={key: value[:s['count']].clone() for key,value in s['tensors'].items()})
                for name,s in self.streams.items()})

    def close(self):
        for obj, name, method in self.originals:
            setattr(obj, name, method)
        self.originals.clear()


@torch.no_grad()
def replay_task_sequence(trace, rally, recovery, contacts):
    """Original task functions on isolated copies; not model inference."""
    if trace['schema'] != NativeTaskSequenceV1.schema:
        raise ValueError('Unknown native event trace')
    objects = {'rally': rally, 'recovery': recovery, 'contact': contacts}
    for name, index in trace['order']:
        stream = trace['streams'][name]
        values = {key: tensor[index].clone() for key,tensor in stream['tensors'].items()}
        values.update({key: seq[index] for key,seq in stream['literals'].items()})
        if name.startswith('recovery.'):
            values['jump'] = SimpleNamespace(**{key: values.pop('jump/' + key) for key in JUMP_INPUTS})
        if name == 'rally.advance':
            rally.in_play.copy_(values.pop('external_in_play'))
        if name == 'rally.launch':
            # FullRacketEnvV4.initialize_rally_launch resets contact episodes
            # immediately before launch, with the identical ids and with no
            # intervening task calls. This deterministic transition is derived
            # from that original method, not from interpolated boundaries.
            contacts.reset(values['ids'])
        group, method = name.split('.')
        getattr(objects[group], method)(**values)
