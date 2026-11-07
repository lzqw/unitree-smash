import unittest
import numpy as np
from scripts.audit_sphere_touchdown_predictions_v1 import sphere_gaps,crossing_mask


class SphereGeometryAudit(unittest.TestCase):
    def test_rotated_center_offset_not_body_origin(self):
        class Model:
            names=['foot']
            def poses(self,q):
                return [np.array([0.,0.,.022])],[np.array([[0.,0.,1.],[0.,1.,0.],[-1.,0.,0.]])]
        shape=dict(body='foot',center_body_m=[-.002,0.,0.],radius_m=.022)
        self.assertAlmostEqual(sphere_gaps(Model(),None,[shape])[0],.002)

    def test_unobserved_crossing_and_existing_support_distinguished(self):
        mask=crossing_mask([.01,.01,.01,-.001],[-.001,-.001,.005,-.002],
            np.array([False,True,False,False]),1e-4)
        np.testing.assert_array_equal(mask,[True,False,False,False])

    def test_tolerance_is_only_diagnostic(self):
        start=np.full(4,.002);end=np.full(4,.0005);contact=np.zeros(4,bool)
        self.assertFalse(crossing_mask(start,end,contact,1e-4).any())
        self.assertTrue(crossing_mask(start,end,contact,1e-3).all())
        np.testing.assert_array_equal(end,np.full(4,.0005))

    def test_malformed_geometry_is_rejected(self):
        for start,end,flags,tolerance in [([0]*3,[0]*4,np.zeros(4,bool),0),
                ([0]*4,[0]*4,[0]*4,0),([0]*4,[0]*4,np.zeros(4,bool),-1),
                ([float('nan')]*4,[0]*4,np.zeros(4,bool),0)]:
            with self.assertRaises(ValueError):crossing_mask(start,end,flags,tolerance)


if __name__=='__main__':unittest.main()
