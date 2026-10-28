"""Synthetic checks of the identities and reference rules used by the paper."""

from pathlib import Path
import sys, unittest
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from common import load_module

vm = load_module("vector_model_tests", ROOT / "03_study_1/vector_model.py")
core = load_module("collective_tests", ROOT / "04_study_2/core.py")
hist = load_module("historical_tests", ROOT / "05_study_3/historical.py")
fusion = load_module("text_fusion_tests", ROOT / "01_preprocessing/text_fusion.py")


class Methods(unittest.TestCase):
    def test_joint_distance_equal_block_weights(self):
        rng = np.random.default_rng(9)
        a = []
        b = []
        for n in [7, 13, 19]:
            x = rng.normal(size=n)
            y = rng.normal(size=n)
            a.append(x / np.linalg.norm(x))
            b.append(y / np.linalg.norm(y))
        self.assertAlmostEqual(
            np.sum((np.concatenate(a) / np.sqrt(3) - np.concatenate(b) / np.sqrt(3)) ** 2),
            np.mean([np.sum((x - y) ** 2) for x, y in zip(a, b)]),
            places=13,
        )

    def test_decomposition_and_translation_invariance(self):
        rng = np.random.default_rng(1)
        cells = {}
        names = ["a", "b", "c", "d"]
        for c in names:
            for p in ["2020H1", "2020H2"]:
                cells[c, p] = dict(mean=[rng.normal(size=7) for _ in range(3)], noise=np.zeros(3))
        d = core.decomp(cells, "2020H1", "2020H2", names)
        np.testing.assert_allclose(d[:, 2], d[:, 3] + d[:, 4], atol=1e-13)
        for value in cells.values():
            value["mean"] = [v + 7.2 for v in value["mean"]]
        np.testing.assert_allclose(core.decomp(cells, "2020H1", "2020H2", names), d, atol=1e-13)

    def test_creator_intercepts_remove_constant_offsets(self):
        rng = np.random.default_rng(7)
        d = pd.DataFrame({"creator_handle": np.repeat(["a", "b", "c"], 5)})
        blocks = {
            b: dict(rawX=rng.normal(size=(15, 8)), rawY=rng.normal(size=(15, 8))) for b in vm.BLOCKS
        }
        before = vm.estimate(vm.sufficient(d, blocks, "raw_first_FE"))["beta"]
        for values in blocks.values():
            for k in ["rawX", "rawY"]:
                values[k] += np.repeat([0, 8, -3], 5)[:, None]
        after = vm.estimate(vm.sufficient(d, blocks, "raw_first_FE"))["beta"]
        np.testing.assert_allclose(before, after, atol=1e-13)

    def test_disjoint_reference_moments_match_all_twenty_splits(self):
        from itertools import combinations

        rng = np.random.default_rng(31)
        h = rng.normal(size=(6, 8))
        x = rng.normal(size=8)
        y = rng.normal(size=8)
        mean = h.mean(0)
        v = np.sum((h - mean) ** 2) / 30
        rows = []
        for a in combinations(range(6), 3):
            b = sorted(set(range(6)) - set(a))
            xx = x - h[list(a)].mean(0)
            yy = y - h[b].mean(0)
            rows.append([xx @ xx, xx @ yy, yy @ yy])
        expected = [
            np.sum((x - mean) ** 2) + v,
            (x - mean) @ (y - mean) - v,
            np.sum((y - mean) ** 2) + v,
        ]
        np.testing.assert_allclose(np.mean(rows, 0), expected, atol=1e-13)

    def test_missing_followup_is_not_a_return(self):
        self.assertEqual(hist.outcome_status([None, None], 1), "unresolved_missing_followup")
        self.assertEqual(hist.outcome_status([0.9, None], 1), "at_least_one_observed_return")
        self.assertEqual(hist.outcome_status([1.2, 1.1], 1), "both_followups_outside")

    def test_creator_balanced_cutoff(self):
        q = hist.weighted_quantile(np.array([1.0, 2.0, 10.0]), np.array(["a", "a", "b"]), 0.5)
        self.assertEqual(q, 2.0)

    def test_ocr_confidence_and_unicode_deduplication(self):
        text, n = fusion.dedup_ocr(
            {
                "observations": [
                    dict(text=" Salt ", confidence=0.9),
                    dict(text="ＳＡＬＴ", confidence=0.9),
                    dict(text="pepper", confidence=0.2),
                    dict(text="Oil", confidence=0.8),
                ]
            }
        )
        self.assertEqual(text, "Salt Oil")
        self.assertEqual(n, 1)


if __name__ == "__main__":
    unittest.main()
