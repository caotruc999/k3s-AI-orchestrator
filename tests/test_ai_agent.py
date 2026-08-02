import unittest

import _pathsetup  # noqa: F401
from ai_agent import AIAgent, AgentConfig


def make_agent(**overrides):
    defaults = dict(
        min_replicas=1,
        max_replicas=3,
        scale_up_step=1,
        scale_down_step=1,
        cooldown_seconds=30,
        mode="AUTO",
    )
    defaults.update(overrides)
    return AIAgent(AgentConfig(**defaults))


class TestAIAgentModeGuards(unittest.TestCase):
    def test_set_mode_rejects_invalid_value(self):
        agent = make_agent()
        with self.assertRaises(ValueError):
            agent.set_mode("PAUSED")

    def test_apply_decision_ignored_in_manual_mode(self):
        agent = make_agent(mode="MANUAL")
        result = agent.apply_decision("Scale Up")
        self.assertEqual(result["action_taken"], "None")
        self.assertEqual(agent.current_replicas, 1)

    def test_manual_scale_rejected_in_auto_mode(self):
        agent = make_agent(mode="AUTO")
        result = agent.manual_scale("Scale Up")
        self.assertEqual(result["action_taken"], "None")
        self.assertEqual(agent.current_replicas, 1)


class TestAIAgentAutoScaling(unittest.TestCase):
    def test_scale_up_respects_max_replicas(self):
        agent = make_agent(cooldown_seconds=0)  # tắt cooldown để test riêng max_replicas
        agent.apply_decision("Scale Up")  # 1 -> 2
        agent.apply_decision("Scale Up")  # 2 -> 3 (= max_replicas)
        self.assertEqual(agent.current_replicas, 3)

        result = agent.apply_decision("Scale Up")  # đã chạm max, không tăng nữa
        self.assertEqual(result["action_taken"], "None")
        self.assertIn("max_replicas", result["reason"])
        self.assertEqual(agent.current_replicas, 3)

    def test_scale_up_blocked_during_cooldown(self):
        agent = make_agent()  # cooldown_seconds=30 (mặc định)
        agent.apply_decision("Scale Up")
        self.assertEqual(agent.current_replicas, 2)

        result = agent.apply_decision("Scale Up")
        self.assertEqual(result["action_taken"], "None")
        self.assertIn("cooldown", result["reason"].lower())
        self.assertEqual(agent.current_replicas, 2)

    def test_scale_down_respects_min_replicas(self):
        agent = make_agent(min_replicas=1, cooldown_seconds=0)
        agent.current_replicas = 1
        result = agent.apply_decision("Scale Down")
        self.assertEqual(result["action_taken"], "None")
        self.assertIn("min_replicas", result["reason"])
        self.assertEqual(agent.current_replicas, 1)

    def test_keep_never_blocked_by_cooldown(self):
        agent = make_agent(cooldown_seconds=999)
        agent.apply_decision("Scale Up")
        result = agent.apply_decision("Keep")
        self.assertEqual(result["action_taken"], "Keep")

    def test_invalid_decision_string(self):
        agent = make_agent()
        result = agent.apply_decision("Do The Thing")
        self.assertEqual(result["action_taken"], "None")
        self.assertIn("không hợp lệ", result["reason"])


class TestAIAgentManualScaling(unittest.TestCase):
    def test_manual_scale_up_and_down(self):
        agent = make_agent(mode="MANUAL", cooldown_seconds=0)

        up = agent.manual_scale("Scale Up")
        self.assertEqual(up["action_taken"], "Manual Scaled Up")
        self.assertEqual(agent.current_replicas, 2)

        down = agent.manual_scale("Scale Down")
        self.assertEqual(down["action_taken"], "Manual Scaled Down")
        self.assertEqual(agent.current_replicas, 1)

    def test_manual_keep_does_not_change_replicas(self):
        agent = make_agent(mode="MANUAL")
        result = agent.manual_scale("Keep")
        self.assertEqual(result["action_taken"], "Keep")
        self.assertEqual(agent.current_replicas, 1)


if __name__ == "__main__":
    unittest.main()
