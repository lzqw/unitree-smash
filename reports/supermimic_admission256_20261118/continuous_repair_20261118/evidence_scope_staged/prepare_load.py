from pathlib import Path
p=Path('quadsmash/rl/supermimic_action_ball_v1.py');s=p.read_text()
a=s.index('    def load_state_dict(self,state_dict,strict=True,assign=False):',s.index('class MaterialInertiaActionBallV4'))
b=s.index('    def mechanical_features',a)
s=s[:a]+'''    def _load_from_state_dict(self,state_dict,prefix,local_metadata,strict,missing_keys,unexpected_keys,error_msgs):
        # Parent adapter.load_state_dict recursively calls this hook; overriding
        # only public load_state_dict would leave resumed mechanical buffers unchecked.
        expected=self.state_dict()
        for key,value in expected.items():
            if key.startswith(('inertia.','impulse_geometry.')) or key in ('component_material','mechanical_scale'):
                saved=state_dict.get(prefix+key)
                if saved is None or not torch.equal(value.detach().cpu(),saved.detach().cpu()):
                    raise ValueError('Immutable mechanical contract differs: '+prefix+key)
        super()._load_from_state_dict(state_dict,prefix,local_metadata,strict,missing_keys,unexpected_keys,error_msgs)

'''+s[b:]
Path(__file__).with_name(p.name).write_text(s)
