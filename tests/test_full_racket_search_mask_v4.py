import unittest
import torch

from quadsmash.control.full_racket_search_mask_v4 import search_coordinate_mask, project_reference_coordinates


class SearchMaskTest(unittest.TestCase):
    def test_default_keeps_every_coordinate_and_recovery_block_is_explicit(self):
        self.assertTrue(search_coordinate_mask(37, None).all())
        indices = [3, 32, 33, 34, 35, 36]
        self.assertEqual(torch.where(search_coordinate_mask(37, indices))[0].tolist(), indices)
        for invalid in [[], [3,3], [-1], [37], [3.0], [True]]:
            with self.assertRaises(ValueError):
                search_coordinate_mask(37, invalid)

    def test_samples_and_updated_means_keep_fixed_reference_coordinates(self):
        reference = torch.arange(37, dtype=torch.float32)
        sampled = -torch.arange(4*37, dtype=torch.float32).reshape(4,37)
        before = sampled.clone()
        mask = search_coordinate_mask(37, [3,32,33,34,35,36])
        projected = project_reference_coordinates(sampled, reference, mask)
        self.assertTrue(torch.equal(projected[:,mask], sampled[:,mask]))
        self.assertTrue(torch.equal(projected[:,~mask], reference[~mask][None].repeat(4,1)))
        self.assertTrue(torch.equal(sampled, before))
        updated = project_reference_coordinates(projected.mean(0), reference, mask)
        self.assertTrue(torch.equal(updated[~mask], reference[~mask]))

    def test_unknown_mask_layout_cannot_silently_change_coordinates(self):
        with self.assertRaises(ValueError):
            project_reference_coordinates(torch.zeros(4,37),torch.zeros(32),torch.ones(32,dtype=torch.bool))


if __name__ == '__main__':
    unittest.main()
