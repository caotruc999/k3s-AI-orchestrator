import unittest

import _pathsetup  # noqa: F401  (thêm modules/ vào sys.path)
from decision_policy import DecisionPolicy, DecisionThresholds


class TestDecisionPolicy(unittest.TestCase):
    def setUp(self):
        self.policy = DecisionPolicy()  # scale_down=0.8, scale_up=1.1

    def test_scale_up_when_score_at_or_above_threshold(self):
        self.assertEqual(self.policy.decide(1.1), "Scale Up")
        self.assertEqual(self.policy.decide(5.0), "Scale Up")

    def test_scale_down_when_score_below_threshold(self):
        self.assertEqual(self.policy.decide(0.79), "Scale Down")
        self.assertEqual(self.policy.decide(-1.0), "Scale Down")

    def test_keep_in_stable_band(self):
        self.assertEqual(self.policy.decide(0.8), "Keep")
        self.assertEqual(self.policy.decide(1.0), "Keep")
        self.assertEqual(self.policy.decide(1.0999), "Keep")

    def test_explain_includes_decision_and_score(self):
        text = self.policy.explain(1.35)
        self.assertIn("Scale Up", text)
        self.assertIn("1.350", text)

    def test_custom_thresholds(self):
        policy = DecisionPolicy(DecisionThresholds(scale_down=0.5, scale_up=2.0))
        self.assertEqual(policy.decide(0.6), "Keep")
        self.assertEqual(policy.decide(2.0), "Scale Up")
        self.assertEqual(policy.decide(0.4), "Scale Down")


if __name__ == "__main__":
    unittest.main()
