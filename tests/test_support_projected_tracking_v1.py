import unittest
import numpy as np
from scripts.support_projected_tracking_v1 import project_support


class SupportProjectionTests(unittest.TestCase):
    def setUp(self):
        self.feet=np.array([[-1.,-1.],[1.,-1.],[1.,1.],[-1.,1.]])

    def test_inside_reference_is_unchanged(self):
        point=np.array([.1,-.2])
        np.testing.assert_array_equal(project_support(point,self.feet),point)

    def test_nearest_edge_and_corner_of_inset_polygon(self):
        np.testing.assert_allclose(project_support([2.,.2],self.feet,.02),[.98,.2],atol=1e-12)
        np.testing.assert_allclose(project_support([2.,3.],self.feet,.02),[.98,.98],atol=1e-12)

    def test_global_rotation_translation_and_foot_order(self):
        angle=.7;c,s=np.cos(angle),np.sin(angle);rotation=np.array([[c,-s],[s,c]])
        translation=np.array([3.,-2.]);point=np.array([2.,.2])
        expected=rotation@project_support(point,self.feet)+translation
        moved=self.feet[::-1]@rotation.T+translation
        np.testing.assert_allclose(project_support(rotation@point+translation,moved),expected,atol=1e-12)

    def test_no_fictitious_polygon_for_one_two_or_collinear_feet(self):
        for feet in [self.feet[:1],self.feet[:2],np.array([[-1.,0.],[0.,0.],[1.,0.]])]:
            self.assertIsNone(project_support([0.,0.],feet))

    def test_eroded_support_can_be_infeasible(self):
        self.assertIsNone(project_support([0.,0.],self.feet*.001,.01))

    def test_nonfinite_input_is_rejected(self):
        with self.assertRaises(ValueError):project_support([np.nan,0.],self.feet)


if __name__=='__main__':unittest.main()
