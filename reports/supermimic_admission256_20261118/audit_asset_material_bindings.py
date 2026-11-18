"""Static USD material composition audit. Does not launch Kit or PhysX."""
import os,sys,json,hashlib
from pathlib import Path
try:from pxr import Usd,UsdPhysics,UsdShade
except ImportError:
 if os.environ.get('QUADSMASH_USD_CPU_BOOTSTRAP'):raise
 libs=next((Path(sys.prefix)/'lib/python3.11/site-packages/isaacsim/extscache').glob('omni.usd.libs-*'))
 env=dict(os.environ);env['PYTHONPATH']=str(libs)+os.pathsep+env.get('PYTHONPATH','');env['LD_LIBRARY_PATH']=str(libs/'bin')+os.pathsep+str(Path(sys.prefix)/'lib')+os.pathsep+env.get('LD_LIBRARY_PATH','');env['QUADSMASH_USD_CPU_BOOTSTRAP']='1'
 os.execve(sys.executable,[sys.executable,str(Path(__file__).resolve()),*sys.argv[1:]],env)
p=Path('assets/quadsmash_full_racket_v4_r3.usd');stage=Usd.Stage.Open(str(p.resolve()));rows=[]
for prim in Usd.PrimRange(stage.GetPrimAtPath('/World/Racket')):
 if prim.HasAPI(UsdPhysics.CollisionAPI):
  material,rel=UsdShade.MaterialBindingAPI(prim).ComputeBoundMaterial('physics')
  rows.append(dict(path=str(prim.GetPath()),type=prim.GetTypeName(),bound_material=str(material.GetPath()) if material else None))
print(json.dumps(dict(asset=str(p.resolve()),sha256=hashlib.sha256(p.read_bytes()).hexdigest(),collisions=rows,root_bindings=[str(r.GetName())+':'+str(r.GetTargets()) for r in stage.GetPrimAtPath('/World/Racket').GetRelationships()]),indent=2))
