"""Versioned static net profile: center 1.524 m, sides 1.55 m.

The frozen benchmark width/thickness stay 2 m/0.02 m. A piecewise linear
parabolic top is a declared geometric model, not a measured net shape.
"""
PROFILE_VERSION = 'environment-local-profiled-net-v3'


def net_top_height(y, center=1.524, side=1.55, half_width=1., segments=32):
    coordinate = ((y/half_width).clamp(-1, 1)+1)*(.5*segments)
    index = coordinate.floor().clamp(0, segments-1)
    fraction = coordinate-index
    left = -1+2*index/segments
    right = left+2/segments
    return center+(side-center)*((1-fraction)*left.square()+fraction*right.square())


def net_mesh(center=1.524, side=1.55, half_width=1., thickness=.02, segments=32):
    if segments < 2 or segments % 2 or side < center:
        raise ValueError('Net needs even segments and sides at least as high as center')
    points, faces = [], []
    for i in range(segments+1):
        y = half_width*(-1+2*i/segments)
        top = center+(side-center)*(y/half_width)**2
        points.extend([(-thickness/2, y, -center/2), (thickness/2, y, -center/2),
                       (-thickness/2, y, top-center/2), (thickness/2, y, top-center/2)])
    for i in range(segments):
        j, k = 4*i, 4*(i+1)
        faces.extend([(j, k, k+2, j+2), (j+1, j+3, k+3, k+1),
                      (j, j+1, k+1, k), (j+2, k+2, k+3, j+3)])
    k = 4*segments
    faces.extend([(0, 2, 3, 1), (k, k+1, k+3, k+2)])
    return points, [tuple(reversed(face)) for face in faces]


def spawn_profiled_net(stage, path, x, center, side, segments=32):
    from pxr import UsdGeom, UsdPhysics
    root = UsdGeom.Xform.Define(stage, path)
    root.AddTranslateOp().Set((x, 0., center/2))
    root.GetPrim().SetCustomDataByKey('quadsmashNetProfile', PROFILE_VERSION)
    UsdGeom.Xform.Define(stage, path+'/geometry')
    mesh = UsdGeom.Mesh.Define(stage, path+'/geometry/mesh')
    points, faces = net_mesh(center=center, side=side, segments=segments)
    mesh.CreatePointsAttr(points)
    mesh.CreateFaceVertexCountsAttr([len(face) for face in faces])
    mesh.CreateFaceVertexIndicesAttr([i for face in faces for i in face])
    mesh.CreateSubdivisionSchemeAttr('none')
    mesh.CreateDoubleSidedAttr(True)
    UsdPhysics.CollisionAPI.Apply(mesh.GetPrim()).CreateCollisionEnabledAttr(True)
    UsdPhysics.MeshCollisionAPI.Apply(mesh.GetPrim()).CreateApproximationAttr('none')
    return mesh.GetPrim()
