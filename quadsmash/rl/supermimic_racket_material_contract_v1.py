"""Read the actual native component materials; never mutate a physics asset.

The strings' material binding does not apply to sibling frame/clamp shapes.
The receipt is native evidence, not a success label or an actor input.
"""
import hashlib,json,math
from pathlib import Path
import torch
from quadsmash.physics.full_racket_v4 import parts,string_strips

ROOT=Path(__file__).resolve().parents[2]
SCHEMA='native-racket-component-materials-v1'

def digest(record):
 return hashlib.sha256(json.dumps(record,sort_keys=True,separators=(',',':')).encode()).hexdigest()

def component_names():
 config=json.loads((ROOT/'configs/full_racket_v4_r3.json').read_text())
 components=[p for p in parts(config) if p['collision']]
 return ([f'Collider/strip_{i:02d}' for i,_ in enumerate(string_strips(config))]+
         [p['name'] for p in components if p['kind']=='box']+
         [p['name'] for p in components if p['kind'] in ('capsule','cylinder')])

def combine(a,b,ma,mb):
 modes=('average','min','multiply','max')
 if ma not in modes or mb not in modes:raise ValueError('Unknown material combine rule')
 mode=modes[max(modes.index(ma),modes.index(mb))]
 return {'average':lambda:(a+b)/2,'min':lambda:min(a,b),'multiply':lambda:a*b,'max':lambda:max(a,b)}[mode]()

def validate_contract(c,asset_sha256):
 if c.get('schema')!=SCHEMA or c.get('asset_sha256')!=asset_sha256:raise ValueError('Native racket material asset/schema mismatch')
 if c.get('sha256')!=digest({k:v for k,v in c.items() if k!='sha256'}):raise ValueError('Native material contract digest mismatch')
 if c.get('component_names')!=component_names():raise ValueError('Native material component order mismatch')
 if not c.get('native_tensor_evidence') or len(c['effective'])!=len(c['component_names']):raise ValueError('Missing measured material evidence')
 for row in c['effective']:
  if len(row)!=3 or not all(math.isfinite(v) and v>=0 for v in row) or row[2]>1:raise ValueError('Invalid measured material values')
 return c

def contract_from_native_receipt(directory):
 directory=Path(directory);path=directory/'material_contract.json';tensor_path=directory/'native_material_tensors.pt'
 r=json.loads(path.read_text());arrays=torch.load(tensor_path,map_location='cpu',weights_only=True)
 if not r['robot_all_lanes_equal'] or not r['shuttle_all_lanes_equal']:raise ValueError('Nonuniform original128 material contract')
 for name in ('robot','shuttle'):
  x=arrays[name]
  if x.ndim!=3 or x.shape[0]!=128 or x.shape[-1]!=3 or not torch.isfinite(x).all() or not torch.equal(x,x[:1].expand_as(x)):raise ValueError('Invalid measured '+name+' material tensor')
 ball=torch.unique(arrays['shuttle'][0],dim=0)
 if ball.shape!=(1,3):raise ValueError('Expected one native ball material')
 default=r['default_scene_material'];prefix='/World/envs/env_0/Robot/Racket/'
 rows={x['path'].removeprefix(prefix):x for x in r['racket_collision_bindings']}
 names=component_names()
 if set(names)!=set(rows):raise ValueError('Native collider set differs from preview geometry')
 cfg=json.loads((ROOT/'configs/shuttle_physics_v2.json').read_text())
 torch.testing.assert_close(ball[0],torch.tensor([cfg['friction'],cfg['friction'],cfg['restitution']]),atol=1e-7,rtol=0)
 effective=[];evidence=[]
 for name in names:
  row=rows[name]
  if row['bound_material'] is None:
   mat=default;values=[mat['static_friction'],mat['dynamic_friction'],mat['restitution']]
   fm=mat['friction_combine_mode'];em=mat['restitution_combine_mode']
  else:
   mat=row['attributes'];values=[mat['physics:staticFriction'],mat['physics:dynamicFriction'],mat['physics:restitution']]
   fm=mat['physxMaterial:frictionCombineMode'];em=mat['physxMaterial:restitutionCombineMode']
  value=torch.tensor(values,dtype=arrays['robot'].dtype)
  if not torch.isclose(arrays['robot'][0],value,rtol=0,atol=1e-7).all(-1).any():raise ValueError('USD material absent from native tensor: '+name)
  effective.append([combine(float(ball[0,j]),values[j],'average' if j<2 else 'max',fm if j<2 else em) for j in range(3)])
  evidence.append(dict(component=name,binding=row['bound_material'],native_properties=values))
 c=dict(schema=SCHEMA,asset_sha256=r['task_contract']['asset']['sha256'],component_names=names,effective=effective,
        native_tensor_evidence=dict(receipt_sha256=hashlib.sha256(path.read_bytes()).hexdigest(),tensor_sha256=hashlib.sha256(tensor_path.read_bytes()).hexdigest(),num_envs=128),
        component_evidence=evidence,scope='Measured original materials; model features only; no physics mutation')
 c['sha256']=digest(c);return validate_contract(c,c['asset_sha256'])
