import math
import unittest

from scripts.summarize_precontact_prediction_v1 import phase


class IntervalClassification(unittest.TestCase):
    def test_event_at_start_already_observed(self):
        self.assertEqual(phase(3.2,320),'after_or_at_observed_nonfoot')

    def test_event_inside_prediction_interval_is_not_preimpact(self):
        self.assertEqual(phase(3.205,320),'interval_crosses_observed_nonfoot')

    def test_event_at_endpoint_is_not_preimpact(self):
        self.assertEqual(phase(3.21,320),'interval_crosses_observed_nonfoot')

    def test_later_event_leaves_interval_before_impact(self):
        self.assertEqual(phase(3.21001,320),'before_observed_nonfoot')

    def test_absence_is_explicit_not_proof_of_no_collision(self):
        self.assertEqual(phase(math.inf,320),'no_recorded_nonfoot_event')

    def test_invalid_clock_rejected(self):
        for timestamp,tick,dt in [(math.nan,320,.01),(-math.inf,320,.01),
                                   (-1,320,.01),(3.2,-1,.01),(3.2,1.5,.01),
                                   (3.2,320,0),(3.2,320,math.nan)]:
            with self.subTest(timestamp=timestamp,tick=tick,dt=dt):
                with self.assertRaises(ValueError):phase(timestamp,tick,dt)


if __name__=='__main__':unittest.main()
