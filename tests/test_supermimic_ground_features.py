import json
import numpy as np
import pytest
import torch
from quadsmash.rl.supermimic_full_body_geometry_v1 import FullBodyGeometryV1, MODEL
from quadsmash.rl.supermimic_ground_features_v1 import AuthoredGroundFeaturesV1, GEOMETRY
from scripts.authored_ground_clearance_v1 import support_witness

ASSET=json.loads(MODEL.read_text())['asset_sha256']


def model_inputs(n=2):
    generator=torch.Generator().manual_seed(83)
    joints=torch.randn(n,18,generator=generator,dtype=torch.float64)*.3
    q=torch.randn(n,4,generator=generator,dtype=torch.float64)
    root=torch.randn(n,3,generator=generator,dtype=torch.float64)
    return joints,root,q


def test_batched_all_shapes_match_existing_support_witness():
    fk=FullBodyGeometryV1(ASSET,dtype=torch.float64)
    f=AuthoredGroundFeaturesV1(fk.names,ASSET,dtype=torch.float64)
    joints,root,q=model_inputs()
    z=torch.zeros_like(root)
    bodies=fk(joints,torch.zeros_like(joints),root,q,z,z)
    got=f(bodies['position'],bodies['rotation'],.03)
    expected=[]
    for batch in range(2):
        row=[]
        for shape in json.loads(GEOMETRY.read_text())['shapes']:
            shape['transform']=np.asarray(shape['transform'])
            if shape['kind']=='Mesh':shape['points']=np.asarray(shape['points'])
            i=fk.ids[shape['body']]
            r=bodies['rotation'][batch,i].numpy();p=bodies['position'][batch,i].numpy()
            row.append(p[2]+(r@support_witness(shape,r))[2]-.03)
        expected.append(row)
    torch.testing.assert_close(got['shape_height'],torch.tensor(expected),atol=1e-10,rtol=1e-10)
    assert got['shape_height'].shape==(2,128)
    assert torch.all(got['nonfoot_height'] <= got['head_height'])


def test_translation_and_plane_height_are_not_collision_thresholds():
    fk=FullBodyGeometryV1(ASSET,dtype=torch.float64)
    f=AuthoredGroundFeaturesV1(fk.names,ASSET,dtype=torch.float64)
    joints,root,q=model_inputs()
    z=torch.zeros_like(root);body=fk(joints,joints*0,root,q,z,z)
    a=f(body['position'],body['rotation'])
    b=f(body['position']+root.new_tensor([0,0,.25]),body['rotation'])
    c=f(body['position'],body['rotation'],.25)
    torch.testing.assert_close(b['shape_height'],a['shape_height']+.25)
    torch.testing.assert_close(c['shape_height'],a['shape_height']-.25)


def test_surface_gradient_reaches_physical_coordinates():
    fk=FullBodyGeometryV1(ASSET,dtype=torch.float64)
    f=AuthoredGroundFeaturesV1(fk.names,ASSET,dtype=torch.float64)
    args=tuple(x.requires_grad_() for x in model_inputs(1))
    def forward(joints,root,q):
        z=torch.zeros_like(root)
        body=fk(joints,joints*0,root,q,z,z)
        return f(body['position'],body['rotation'])['shape_height']
    assert torch.autograd.gradcheck(forward,args,eps=1e-6,atol=2e-5,rtol=1e-4,fast_mode=True)


def test_geometry_rejects_unknown_asset():
    fk=FullBodyGeometryV1(ASSET)
    with pytest.raises(ValueError,match='current asset'):
        AuthoredGroundFeaturesV1(fk.names,'wrong')
