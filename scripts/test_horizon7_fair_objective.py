import math
import unittest

from analyze_horizon7_fair_waypoint_budget import normalized_row


class ObjectiveNormalizationTest(unittest.TestCase):
    def test_exported_denominator_is_not_reused(self):
        for divisor in (7.0, 7.01):
            raw = {'avg_true_objective': '-14', 'avg_true_objective_per_second': -14/divisor}
            row = normalized_row(raw, lambda r: {'objective_s': r['avg_true_objective_per_second'], 'success': .5}, 7.0)
            self.assertEqual(row['objective_s'], -2)
            self.assertEqual(row['success'], .5)

    def test_invalid_duration_and_objective_rejected(self):
        for duration in (0.0, -1.0, math.nan, math.inf):
            with self.assertRaises(ValueError):
                normalized_row({'avg_true_objective': -14}, dict, duration)
        with self.assertRaises(ValueError):
            normalized_row({'avg_true_objective': math.nan}, dict, 7.0)


if __name__ == '__main__':
    unittest.main()
