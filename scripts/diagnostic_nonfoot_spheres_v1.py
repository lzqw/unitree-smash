"""Extract authored nonfoot spheres for historical collision-response diagnostics.

This does not make nonfoot contact permissible for a teacher reference, and
does not claim to cover mesh/cylinder collisions or contact manifolds.
"""
import numpy as np


def nonfoot_spheres(geometry,feet):
    result=[]
    for shape in geometry.shapes:
        if shape['kind']!='Sphere' or shape['body'] in feet:continue
        transform=np.asarray(shape['transform'],float)
        if transform.shape!=(4,4) or not np.isfinite(transform).all():
            raise ValueError('Invalid authored sphere transform')
        metric=transform[:3,:3]@transform[:3,:3].T
        scale=float(np.sqrt(np.trace(metric)/3.))
        if scale<=0 or not np.allclose(metric,np.eye(3)*scale**2,atol=1e-10,rtol=0.):
            raise ValueError('Nonuniform sphere scaling unsupported')
        radius=float(shape['radius'])*scale
        if not np.isfinite(radius) or radius<=0:raise ValueError('Invalid authored sphere radius')
        result.append(dict(body=shape['body'],center_body_m=transform[3,:3].tolist(),
                           radius_m=radius,path=shape['path']))
    return result
