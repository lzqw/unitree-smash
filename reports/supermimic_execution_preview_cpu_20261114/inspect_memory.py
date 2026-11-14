"""Read-only initial controller-memory inventory, CPU, no native steps."""
import json,hashlib,types
from pathlib import Path
import torch
from quadsmash.control.g07_shared_target_residual_v1 import make_g07_residual_controller

torch.set_num_threads(2)
p=Path('/home/autodrive/LZQW/unitree_smash/reports/G07_full_racket_v5/frozen_v2_smooth_approach16_env5120_01/policy.pt')
saved=torch.load(p,map_location='cpu',weights_only=False)
c=make_g07_residual_controller(saved,2,'cpu',Path('/home/autodrive/LZQW/unitree_smash'))
seen={};out=[]
def visit(obj,path,depth=0):
 if torch.is_tensor(obj):
  out.append({'path':path,'kind':'tensor','shape':list(obj.shape),'dtype':str(obj.dtype)})
  return
 if isinstance(obj,torch.nn.Module):
  out.append({'path':path,'kind':'network','class':type(obj).__name__})
  return
 if obj is None or isinstance(obj,(bool,int,float,str)):
  out.append({'path':path,'kind':type(obj).__name__,'value':obj})
  return
 if isinstance(obj,(tuple,list)):
  out.append({'path':path,'kind':type(obj).__name__,'length':len(obj)})
  return
 if not hasattr(obj,'__dict__') or depth>40:return
 if id(obj) in seen:
  out.append({'path':path,'alias_of':seen[id(obj)]});return
 seen[id(obj)]=path
 for name,value in vars(obj).items():
  if isinstance(value,(types.FunctionType,types.MethodType,types.ModuleType,type)):continue
  visit(value,path+'.'+name,depth+1)
visit(c,'controller')
r={'source':str(p),'source_sha256':hashlib.sha256(p.read_bytes()).hexdigest(),'count':2,'native_steps':0,'scope':'Construction-time inventory only; lazy runtime states and contact caches not exhaustively captured. No training or dynamics evidence.','attributes':out}
Path(__file__).with_name('controller_memory_inventory.json').write_text(json.dumps(r,indent=2)+'\n')
print('attributes',len(out),'tensors',sum(x.get('kind')=='tensor' for x in out))
for x in out:
 if x.get('kind')=='tensor' and any(k in x['path'] for k in ('clock','time','target','plan','seen','initialized','last','phase','valid')):print(x)
