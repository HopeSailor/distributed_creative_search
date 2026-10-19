"""Algebra and edge-case tests on synthetic arrays only; no research data."""
import unittest
import numpy as np
from common import next_period,check_creators
from study2 import decompose,correction,neighbor_retention
from validation import agreement,correlation

class AnalysisTests(unittest.TestCase):
    def test_period_boundaries(self):
        self.assertEqual(next_period('2021H2'),'2022H1')
        self.assertEqual(next_period('2021H1',2),'2022H1')

    def test_reject_nonfocal(self):
        with self.assertRaises(ValueError): check_creators(['unrelated_account'])

    def test_translation_does_not_change_dispersion(self):
        x=np.array([[0.,0.],[1.,2.],[2.,0.]])
        r=decompose(x,x+3)
        self.assertAlmostEqual(r['DeltaV'],0)
        self.assertAlmostEqual(r['C'],0)
        self.assertAlmostEqual(r['H'],0)

    def test_corrected_identity(self):
        rng=np.random.default_rng(1)
        aa=[rng.normal(size=(5,4)) for _ in range(3)]
        bb=[rng.normal(size=(7,4)) for _ in range(3)]
        r=correction(aa,bb)
        self.assertAlmostEqual(r['C_corrected']+r['H_corrected'],r['DeltaV_corrected'])

    def test_identical_neighbors(self):
        x=np.array([[0.,0.],[1.,2.],[3.,0.],[4.,5.]])
        self.assertAlmostEqual(neighbor_retention(x,x,1),1)

    def test_perfect_coder_agreement(self):
        rows=[{'coder1':[i]*3,'coder2':[i]*3,'consensus':[i]*3,'distances':[float(i)]*3} for i in range(1,6)]
        np.testing.assert_allclose(agreement(rows)['linear_weighted_kappa'],1)
        np.testing.assert_allclose(correlation(rows),1)

if __name__=='__main__': unittest.main()
