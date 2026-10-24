import copy
import unittest
import numpy as np
import torch

from quadsmash.control.full_racket_candidate_bank_v4 import validate_candidate_bank, install_candidate_bank


class CandidateBankTest(unittest.TestCase):
    def setUp(self):
        self.names=[f'p{i}' for i in range(32)]
        self.seed={'joint_position_rad':[0.]*6}
        self.reference=np.linspace(-.2,.2,32,dtype=np.float32)
        self.bank=dict(schema='full-racket-candidate-bank-v4',asset_sha256='asset',seed=self.seed,
                       parameter_names=self.names,reference_parameters=self.reference.tolist(),
                       parameters=np.zeros((4,32)).tolist())
        self.lower,self.upper=np.full(32,-1.),np.full(32,1.)

    def validate(self,bank):
        return validate_candidate_bank(bank,self.seed,'asset',self.names,self.reference,self.lower,self.upper,6)

    def test_finite_bounded_bank_has_independent_storage_and_preserves_reference(self):
        before=copy.deepcopy(self.bank)
        rows=self.validate(self.bank)
        self.assertEqual(rows.shape,(4,32))
        rows[0,0]=.5
        self.assertEqual(self.bank,before)
        np.testing.assert_array_equal(self.reference,before['reference_parameters'])

    def test_asset_seed_parameter_order_and_reference_must_match(self):
        for key,value in [('schema','unknown'),('asset_sha256','other'),('seed',{}),
                          ('parameter_names',list(reversed(self.names))),('reference_parameters',[0.]*32)]:
            bank=copy.deepcopy(self.bank);bank[key]=value
            with self.assertRaises(ValueError):self.validate(bank)

    def test_invalid_ranges_shapes_capacity_and_nonfinite_values_are_rejected(self):
        for rows in [[],np.zeros((7,32)).tolist(),np.zeros((4,31)).tolist()]:
            bank=copy.deepcopy(self.bank);bank['parameters']=rows
            with self.assertRaises(ValueError):self.validate(bank)
        for value in [1.01,-1.01,float('nan'),float('inf')]:
            bank=copy.deepcopy(self.bank);bank['parameters'][0][5]=value
            with self.assertRaises(ValueError):self.validate(bank)

    def test_only_first_generation_is_seeded_and_two_references_are_reserved(self):
        population=torch.arange(256,dtype=torch.float32).reshape(8,32)
        before=population.clone();rows=self.validate(self.bank)
        installed=install_candidate_bank(population,rows,0)
        self.assertIs(installed,population)
        self.assertTrue(torch.equal(population[:2],before[:2]))
        self.assertTrue(torch.equal(population[6:],before[6:]))
        self.assertTrue(torch.equal(population[2:6],torch.zeros((4,32))))
        later=before.clone();install_candidate_bank(later,rows,1)
        self.assertTrue(torch.equal(later,before))
        with self.assertRaises(ValueError):install_candidate_bank(before,np.zeros((7,32)),0)


if __name__=='__main__':unittest.main()
