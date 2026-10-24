"""Export body-local visual meshes without altering simulated geometry."""
import json
import numpy as np


def export_robot_geometry(stage, robot, output):
    from pxr import Usd,UsdGeom,UsdPhysics
    import trimesh
    cache = UsdGeom.XformCache()
    data, entries = {}, []
    root = stage.GetPrimAtPath('/World/envs/env_0/Robot')
    for prim in Usd.PrimRange(root,Usd.TraverseInstanceProxies()):
        kind, path = prim.GetTypeName(), str(prim.GetPath())
        if kind not in ('Mesh','Cube','Sphere','Cylinder','Capsule'):
            continue
        if '/collisions/' in path.lower() or '/collision/' in path.lower():
            continue
        ancestor = prim
        while ancestor.IsValid() and not ancestor.HasAPI(UsdPhysics.RigidBodyAPI):
            ancestor = ancestor.GetParent()
        if not ancestor.IsValid() or ancestor.GetName() not in robot.body_names:
            continue
        if kind == 'Mesh':
            shape = UsdGeom.Mesh(prim)
            vertices = np.asarray(shape.GetPointsAttr().Get(),dtype=float)
            indices, counts = list(shape.GetFaceVertexIndicesAttr().Get()),shape.GetFaceVertexCountsAttr().Get()
            faces,offset = [],0
            for count in counts:
                faces += [[indices[offset],indices[offset+i],indices[offset+i+1]] for i in range(1,count-1)]
                offset += count
            faces = np.asarray(faces,dtype=np.int32)
        else:
            if kind == 'Cube':
                shape = trimesh.creation.box(extents=[float(UsdGeom.Cube(prim).GetSizeAttr().Get())]*3)
            elif kind == 'Sphere':
                shape = trimesh.creation.icosphere(subdivisions=2,radius=float(UsdGeom.Sphere(prim).GetRadiusAttr().Get()))
            else:
                usd_shape = (UsdGeom.Cylinder if kind=='Cylinder' else UsdGeom.Capsule)(prim)
                radius,length = float(usd_shape.GetRadiusAttr().Get()),float(usd_shape.GetHeightAttr().Get())
                shape = trimesh.creation.cylinder(radius=radius,height=length,sections=16)
                if kind == 'Capsule':
                    caps = []
                    for sign in [-1,1]:
                        cap = trimesh.creation.icosphere(subdivisions=1,radius=radius)
                        cap.apply_translation([0,0,sign*length/2])
                        caps.append(cap)
                    shape = trimesh.util.concatenate([shape,*caps])
                axis = usd_shape.GetAxisAttr().Get()
                if axis == 'X':shape.apply_transform(trimesh.transformations.rotation_matrix(np.pi/2,[0,1,0]))
                if axis == 'Y':shape.apply_transform(trimesh.transformations.rotation_matrix(np.pi/2,[1,0,0]))
            vertices,faces = np.asarray(shape.vertices),np.asarray(shape.faces,dtype=np.int32)
        transform = np.asarray(cache.GetLocalToWorldTransform(prim)*cache.GetLocalToWorldTransform(ancestor).GetInverse())
        vertices = (np.c_[vertices,np.ones(len(vertices))]@transform)[:,:3].astype(np.float32)
        key = f'm{len(entries)}'
        data[key+'_vertices'],data[key+'_faces'] = vertices,faces
        color = UsdGeom.Gprim(prim).GetDisplayColorAttr().Get()
        entries.append(dict(key=key,body=robot.body_names.index(ancestor.GetName()),path=path,
                            color=list(color[0]) if color else None))
    if not entries:raise RuntimeError('No renderable robot geometry')
    np.savez_compressed(output/'geometry.npz',**data)
    (output/'geometry.json').write_text(json.dumps(dict(meshes=entries,body_names=robot.body_names),indent=2)+'\n')


def collision_proxy_corners(stage,robot):
    """Conservative body-local collider bounds for a labelled robot safety gate."""
    from pxr import Usd,UsdGeom,UsdPhysics
    import itertools
    transforms = UsdGeom.XformCache()
    indices,points,paths = [],[],[]
    colliders={}
    for prim in Usd.PrimRange(stage.GetPrimAtPath('/World/envs/env_0/Robot'),Usd.TraverseInstanceProxies()):
        if not prim.HasAPI(UsdPhysics.CollisionAPI):continue
        if prim.GetTypeName() in ('Xform','Scope'):
            shapes=[child for child in Usd.PrimRange(prim,Usd.TraverseInstanceProxies())
                    if child.GetTypeName() in ('Mesh','Cube','Sphere','Cylinder','Capsule')]
            if not shapes:raise ValueError(f'Collision group has no shape: {prim.GetPath()}')
        else:shapes=[prim]
        for shape in shapes:colliders[str(shape.GetPath())]=shape
    for prim in colliders.values():
        ancestor=prim
        while ancestor.IsValid() and not ancestor.HasAPI(UsdPhysics.RigidBodyAPI):ancestor=ancestor.GetParent()
        if not ancestor.IsValid() or ancestor.GetName() not in robot.body_names:continue
        # The fixture is part of the robot. Only actual racket components get
        # the racket follow-through exception, despite sharing one rigid body.
        if ancestor.GetName()=='Racket' and prim.GetName()!='Clamp':continue
        # Collision prims may use purpose=guide. BBoxCache purpose filtering
        # then returns an empty range with finite +/-FLT_MAX sentinels. Read
        # actual shape data rather than turning that range into giant corners.
        kind=prim.GetTypeName()
        if kind=='Mesh':
            vertices=np.asarray(UsdGeom.Mesh(prim).GetPointsAttr().Get(),dtype=float)
            if not len(vertices):raise ValueError(f'Empty collision mesh: {prim.GetPath()}')
            lower,upper=vertices.min(0),vertices.max(0)
        elif kind=='Cube':
            half=np.full(3,float(UsdGeom.Cube(prim).GetSizeAttr().Get())/2)
            lower,upper=-half,half
        elif kind=='Sphere':
            half=np.full(3,float(UsdGeom.Sphere(prim).GetRadiusAttr().Get()))
            lower,upper=-half,half
        elif kind in ('Cylinder','Capsule'):
            shape=(UsdGeom.Cylinder if kind=='Cylinder' else UsdGeom.Capsule)(prim)
            half=np.full(3,float(shape.GetRadiusAttr().Get()))
            axis='XYZ'.index(shape.GetAxisAttr().Get())
            half[axis]=float(shape.GetHeightAttr().Get())/2+(half[axis] if kind=='Capsule' else 0.)
            lower,upper=-half,half
        else:raise ValueError(f'Unsupported collision geometry {kind}: {prim.GetPath()}')
        if not np.isfinite([lower,upper]).all() or (lower>upper).any():
            raise ValueError(f'Invalid collision bounds: {prim.GetPath()}')
        local=np.array(list(itertools.product(*zip(lower,upper))),dtype=float)
        transform=np.asarray(transforms.GetLocalToWorldTransform(prim)*transforms.GetLocalToWorldTransform(ancestor).GetInverse())
        corners=(np.c_[local,np.ones(len(local))]@transform)[:,:3]
        if not np.isfinite(corners).all() or np.abs(corners).max()>10:
            raise ValueError(f'Invalid body-local collision corners: {prim.GetPath()}')
        points.append(corners)
        indices.append(robot.body_names.index(ancestor.GetName()));paths.append(str(prim.GetPath()))
    if not points:raise RuntimeError('No actual collider bounds for robot safety monitoring')
    return np.array(indices),np.stack(points),paths
