import math
import unittest
import numpy as np
import torch
from quadsmash.physics.full_racket_v4 import load_racket,mass_properties,parts,string_prism_gap,swept_string_prism,string_strips,string_polygon,RacketPlaneSupport
from quadsmash.physics.court_v4 import load_court,net_band_segments,top_height
from quadsmash.physics.rally_rules_v4 import RallyRulesV4


class FullRacketGeometryTest(unittest.TestCase):
    def test_rotated_aabb_empty_corner_does_not_invent_material_across_net(self):
        import itertools
        from quadsmash.physics.native import quat_rotate
        config=load_racket();model=RacketPlaneSupport(config,dtype=torch.float64)
        q=torch.tensor([[math.sqrt(.5),0.,.5,-.5]],dtype=torch.float64)
        origin=torch.zeros((1,3),dtype=torch.float64)
        exact=float(model.maximum_x(origin,q)[0])
        position=torch.tensor([[1.5-exact-.001,0.,3.]],dtype=torch.float64)
        self.assertAlmostEqual(float(model.maximum_x(position,q)[0]),1.499,places=12)
        corners=[]
        for part in parts(config):
            if part['name']=='Clamp' or not part['collision']:continue
            if part['kind']=='string_prism':
                half=np.array([config['assumptions']['string_thickness_m']/2,
                               *config['assumptions']['strings_semiaxes_yz_m']])
            else:
                half=np.abs(part['axis'])*part['length']/2+part['radius']
            corners.extend(part['center']+np.array(list(itertools.product((-1,1),repeat=3)))*half)
        old=quat_rotate(q[:,None],torch.tensor(np.array(corners)[None],dtype=torch.float64))+position[:,None]
        self.assertGreater(float(old[...,0].max()),1.51)
        self.assertGreater(float(old[...,2].min()),1.55)

    def test_plane_support_rotations_and_translations_preserve_geometry(self):
        model=RacketPlaneSupport(load_racket(),dtype=torch.float64)
        q=torch.tensor([[1.,0,0,0],[math.sqrt(.5),0,math.sqrt(.5),0]],dtype=torch.float64)
        center=torch.zeros((2,3),dtype=torch.float64)
        maximum=model.maximum_x(center,q)
        # Looking along the normal exposes exactly the frame tube radius;
        # looking along the shaft exposes the head's actual rounded top.
        a=load_racket()['assumptions']
        self.assertAlmostEqual(float(maximum[0]),max(a['frame_tube_radius_m'],a['handle_radius_m']+.0005),places=10)
        self.assertAlmostEqual(float(maximum[1]),a['head_outer_length_m']/2,places=10)
        translated=center+torch.tensor([2.,-.7,.4],dtype=torch.float64)
        torch.testing.assert_close(model.maximum_x(translated,q),maximum+2,atol=1e-12,rtol=0)

    def test_clamp_clears_source_wrist_mesh_in_flange_frame(self):
        import trimesh
        from quadsmash.control.full_racket_geometry_v4 import FullRacketGeometryV4
        geometry=FullRacketGeometryV4()
        mesh=trimesh.load(geometry.urdf.parent.parent/'meshes/link5.STL')
        offset,rotation=geometry.joints[-1]
        wrist=(np.asarray(mesh.vertices)-offset)@rotation
        config=load_racket();clamp=next(p for p in parts(config) if p['name']=='Clamp')
        from scipy.spatial.transform import Rotation
        w,x,y,z=config['body_rotation_in_flange_wxyz']
        tool_rotation=Rotation.from_quat([x,y,z,w]).as_matrix()
        import itertools
        corners=clamp['center']+np.array(list(itertools.product([-1,1],repeat=3)))*clamp['size']/2
        corners=corners@tool_rotation.T+np.array(config['string_center_in_flange_m'])
        # Separating plane remains valid for all joint6 angles about flange Z.
        self.assertGreater(corners[:,2].min()-wrist[:,2].max(),.002)

    def test_native_strips_stay_inside_string_polygon_without_enlarging_face(self):
        config=load_racket();polygon=string_polygon(config)
        edges=np.roll(polygon,-1,axis=0)-polygon
        area=0.
        for strip in string_strips(config):
            center,size=np.array(strip['center']),np.array(strip['size'])
            corners=center[None,1:]+np.array([[-1,-1],[-1,1],[1,-1],[1,1]])*size[None,1:]/2
            delta=corners[:,None,:]-polygon
            self.assertTrue((edges[:,0]*delta[...,1]-edges[:,1]*delta[...,0]>=-1e-14).all())
            area+=np.prod(size[1:])
        polygon_area=(polygon[:,0]*np.roll(polygon[:,1],-1)-polygon[:,1]*np.roll(polygon[:,0],-1)).sum()/2
        self.assertGreater(area/polygon_area,.96)
        self.assertLess(area/polygon_area,1.)

    def test_compound_mass_and_published_balance(self):
        config=load_racket();components=parts(config);properties=mass_properties(config)
        self.assertAlmostEqual(properties['mass_kg'],.153)
        bare=[p for p in components if p['name'] in ['Handle','Shaft'] or p['name'].startswith('Frame_')]
        balance=sum(p['mass']*(p['center'][2]+config['head_center_from_butt_m']) for p in bare)/sum(p['mass'] for p in bare)
        self.assertAlmostEqual(balance,.304)
        self.assertTrue(np.linalg.eigvalsh(properties['inertia_in_body_kg_m2']).min()>0)
        self.assertLess(np.linalg.norm(config['string_center_in_flange_m']),config['published']['total_length_m'])
        self.assertAlmostEqual(np.dot(config['shaft_axis_in_flange'],config['face_normal_in_flange']),0.)

    def test_string_corner_is_not_a_square_blade(self):
        points=torch.tensor([[0,0,0],[0,.09,.11],[.0506,0,0]],dtype=torch.float64)
        center=torch.zeros_like(points);q=torch.tensor([[1.,0,0,0]]*3,dtype=torch.float64)
        gap=string_prism_gap(points,center,q,load_racket(),.025)
        self.assertLess(float(gap[0]),0)
        # A point outside the ellipse, including its sphere-clearance region.
        self.assertGreater(float(string_prism_gap(points[1:2],center[1:2],q[1:2],load_racket(),0.)),0)
        self.assertAlmostEqual(float(gap[2]),.025,places=8)

    def test_convex_prism_sweep_crosses_face_but_misses_corner(self):
        start=torch.tensor([[-.2,0,0],[-.2,.09,.11]],dtype=torch.float64)
        end=start.clone();end[:,0]=.2
        center=torch.zeros_like(start);q=torch.tensor([[1.,0,0,0]]*2,dtype=torch.float64)
        hit,_,unknown=swept_string_prism(start,end,center,center,q,q,load_racket(),.001)
        self.assertEqual(hit.tolist(),[True,False]);self.assertFalse(unknown.any())

    def test_suspended_net_and_full_width(self):
        config=load_court();points=np.array([p for segment,_ in net_band_segments(config) for p in segment])
        self.assertAlmostEqual(points[:,1].min(),-3.05)
        self.assertAlmostEqual(points[:,1].max(),3.05)
        self.assertAlmostEqual(points[:,2].min(),1.524-.76)
        self.assertAlmostEqual(points[:,2].max(),1.55)
        self.assertTrue((points[:,2]>.7).all())
        heights=top_height(torch.tensor([0.,3.05],dtype=torch.float64),config)
        self.assertTrue(torch.allclose(heights,torch.tensor([1.524,1.55],dtype=torch.float64)))

    def test_net_convex_faces_have_outward_winding(self):
        for points,faces in net_band_segments(load_court()):
            vertices=np.array(points)
            volume=0.
            for face in faces:
                for j in range(1,len(face)-1):
                    a,b,c=vertices[[face[0],face[j],face[j+1]]]
                    volume+=np.dot(a,np.cross(b,c))/6
            self.assertGreater(volume,0.)


class RallyLawTest(unittest.TestCase):
    def setUp(self):
        self.events=RallyRulesV4(1,'cpu',load_court(),dtype=torch.float64)
        self.time=0.
        self.events.launch(torch.tensor([0]),torch.tensor([0.],dtype=torch.float64))

    def step(self,**overrides):
        self.time+=.01
        false=torch.tensor([False]);true=torch.tensor([True])
        fields=dict(previous_position=torch.tensor([[1.,0,1.8]],dtype=torch.float64),
            position=torch.tensor([[1.1,0,1.8]],dtype=torch.float64),velocity=torch.tensor([[10.,0,-2.]],dtype=torch.float64),
            time=torch.tensor([self.time],dtype=torch.float64),dt=.01,racket_contact_event=false,
            racket_touching=false,separated=true,contact_point=torch.tensor([[1.,0,1.8]],dtype=torch.float64),
            string_contact=true,frame_contact=false,airborne=true,ground_touch=false,robot_net_touch=false,
            body_ball_touch=false,shuttle_net_touch=false,racket_over_net=false,body_over_net=false,
            engineering_body_crossing=false)
        for name,value in overrides.items():
            if isinstance(value,bool):value=torch.tensor([value])
            elif isinstance(value,list):value=torch.tensor(value,dtype=torch.float64)
            fields[name]=value
        if 'racket_fully_own_side' not in overrides:
            fields['racket_fully_own_side']=~fields['racket_over_net']
        self.events.advance(**fields)

    def hit_and_release(self,**overrides):
        self.step(racket_contact_event=True,racket_touching=True,separated=False,**overrides)
        self.step();self.step()

    def test_rollout_inference_context_allows_next_generation_reset(self):
        ids=torch.tensor([0])
        for _ in range(2):
            with torch.inference_mode():
                self.hit_and_release()
            self.assertTrue(bool(self.events.released[0]))
            self.events.reset(ids)
            self.assertFalse(self.events.quiet_frames.is_inference())
            self.assertEqual(int(self.events.quiet_frames[0]),0)
            self.assertFalse(bool(self.events.had_contact[0]))
            self.events.launch(ids,torch.tensor([self.time],dtype=torch.float64))

    def cross_and_land(self,landing=(3.,0.,.025)):
        self.step(previous_position=[[1.51,0,1.7]],position=[[1.54,0,1.7]])
        self.step(previous_position=[[landing[0],landing[1],.04]],position=[[landing[0],landing[1],.01]],ground_touch=True)

    def test_own_side_airborne_string_hit_can_complete(self):
        self.hit_and_release();self.cross_and_land()
        self.assertTrue(self.events.complete(torch.tensor([True])))

    def test_actual_contact_point_on_opponent_side_is_fault(self):
        self.hit_and_release(contact_point=[[1.501,0,1.8]])
        self.assertTrue(self.events.wrong_side_fault);self.assertFalse(self.events.in_play)

    def test_own_side_near_net_hit_can_pass_while_center_already_on_opponent_side(self):
        self.step(previous_position=[[1.522,0,1.8]],position=[[1.5206,0,1.8]],
            contact_point=[[1.495,0,1.8]],racket_contact_event=True,racket_touching=True,separated=False)
        self.step(previous_position=[[1.5206,0,1.8]],position=[[1.530,0,1.8]],
            racket_touching=True,separated=False)
        self.step();self.step()
        self.step(previous_position=[[3.,0,.04]],position=[[3.,0,.01]],ground_touch=True)
        self.assertTrue(self.events.crossed_net)
        self.assertTrue(self.events.complete(torch.tensor([True])))

    def test_racket_followthrough_exception_does_not_extend_to_body(self):
        self.hit_and_release();self.step(racket_over_net=True)
        self.assertFalse(self.events.racket_invasion_fault)
        self.step(racket_over_net=True,body_over_net=True)
        self.assertTrue(self.events.body_over_net_fault)

    def test_prehit_racket_invasion_is_fault(self):
        self.step(racket_over_net=True)
        self.assertTrue(self.events.racket_invasion_fault)

    def test_lowering_racket_below_net_does_not_end_same_stroke_followthrough(self):
        self.hit_and_release();self.step(racket_over_net=True)
        self.step(racket_over_net=False,racket_fully_own_side=False)
        self.assertFalse(self.events.followthrough_finished)
        self.step(racket_over_net=True,racket_fully_own_side=False)
        self.assertFalse(self.events.racket_invasion_fault)

    def test_complete_return_to_own_side_ends_followthrough_exception(self):
        self.hit_and_release();self.step(racket_over_net=True)
        self.step(racket_over_net=False,racket_fully_own_side=True)
        self.assertTrue(self.events.followthrough_finished)
        self.step(racket_over_net=True,racket_fully_own_side=False)
        self.assertTrue(self.events.racket_invasion_fault)

    def test_net_touch_by_shuttle_is_not_robot_net_fault(self):
        self.hit_and_release();self.step(shuttle_net_touch=True)
        self.cross_and_land()
        self.assertTrue(self.events.shuttle_net_contact)
        self.assertTrue(self.events.complete(torch.tensor([True])))

    def test_robot_net_touch_during_play_invalidates_return(self):
        self.hit_and_release();self.step(robot_net_touch=True)
        self.cross_and_land();self.assertFalse(self.events.complete(torch.tensor([True])))

    def test_robot_net_touch_after_landing_does_not_rewrite_rally(self):
        self.hit_and_release();self.cross_and_land();self.step(robot_net_touch=True)
        self.assertFalse(self.events.robot_net_fault)
        self.assertTrue(self.events.complete(torch.tensor([True])))

    def test_ground_bounce_never_restarts_rally(self):
        self.step(ground_touch=True)
        self.hit_and_release();self.cross_and_land()
        self.assertFalse(self.events.had_contact);self.assertFalse(self.events.complete(torch.tensor([True])))

    def test_independent_second_hit_is_fault(self):
        self.hit_and_release();self.step(racket_contact_event=True,racket_touching=True,separated=False)
        self.assertTrue(self.events.double_hit_fault)

    def test_head_and_strings_in_same_stroke_are_not_double_hit(self):
        self.hit_and_release(string_contact=False,frame_contact=True)
        self.step(racket_contact_event=True,racket_touching=True,separated=False,same_stroke_frame_string=True)
        self.assertFalse(self.events.double_hit_fault)
        self.assertTrue(self.events.frame_hit);self.assertTrue(self.events.string_hit)

    def test_body_or_fixture_touch_is_fault(self):
        self.step(body_ball_touch=True)
        self.assertTrue(self.events.body_ball_fault)

    def test_legal_court_landing_is_distinct_from_target(self):
        self.hit_and_release();self.cross_and_land((8.2,2.59,.025))
        self.assertTrue(self.events.landed_legal_court);self.assertFalse(self.events.landed_target)

    def test_engineering_own_half_gate_is_separate_from_bwf_body_over_net(self):
        self.step(engineering_body_crossing=True)
        self.assertTrue(self.events.engineering_body_crossing)
        self.assertFalse(self.events.body_over_net_fault)

    def test_blocked_falling_net_shot_ends_rally(self):
        self.hit_and_release();self.step(shuttle_net_touch=True,velocity=[[-1.,0,-2.]])
        self.assertTrue(self.events.net_blocked_fault);self.assertFalse(self.events.in_play)

    def test_training_recovery_continuation_does_not_revive_failed_shuttle_rally(self):
        self.hit_and_release();self.step(shuttle_net_touch=True,velocity=[[-1.,0,-2.]])
        self.assertTrue(self.events.faulted())
        self.assertFalse(self.events.robot_rule_faulted())
        self.assertFalse(self.events.in_play)
        self.cross_and_land()
        self.assertFalse(self.events.complete(torch.tensor([True])))

    def test_shuttle_failure_cannot_hide_simultaneous_robot_net_fault(self):
        self.hit_and_release()
        self.step(shuttle_net_touch=True,robot_net_touch=True,velocity=[[-1.,0,-2.]])
        self.assertTrue(self.events.net_blocked_fault)
        self.assertTrue(self.events.robot_rule_faulted())
        self.assertTrue(self.events.faulted())

    def test_preparation_contact_is_not_an_in_play_hit(self):
        self.events.reset(torch.tensor([0]));self.hit_and_release()
        self.assertFalse(self.events.had_contact)


if __name__=='__main__':unittest.main()
