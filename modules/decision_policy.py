# modules/decision_policy.py

from dataclasses import dataclass


@dataclass
class DecisionThresholds:
    scale_down: float = 0.8
    scale_up: float = 1.1


class DecisionPolicy:
    def __init__(self, thresholds: DecisionThresholds | None = None):
        self.thresholds = thresholds or DecisionThresholds()

    def decide(self, score: float) -> str:
        """
        Convert predicted resource pressure score into a scaling action.

        Rules:
        - score >= scale_up   -> Scale Up
        - score < scale_down   -> Scale Down
        - otherwise           -> Keep
        """
        if score >= self.thresholds.scale_up:
            return "Scale Up"
        elif score < self.thresholds.scale_down:
            return "Scale Down"
        return "Keep"

    def explain(self, score: float) -> str:
        action = self.decide(score)
        if action == "Scale Up":
            reason = f"score={score:.3f} vượt ngưỡng {self.thresholds.scale_up:.3f}"
        elif action == "Scale Down":
            reason = f"score={score:.3f} thấp hơn ngưỡng {self.thresholds.scale_down:.3f}"
        else:
            reason = (
                f"score={score:.3f} nằm trong vùng ổn định "
                f"[{self.thresholds.scale_down:.3f}, {self.thresholds.scale_up:.3f})"
            )
        return f"{action} | {reason}"


if __name__ == "__main__":
    policy = DecisionPolicy()
    test_scores = [0.55, 0.75, 0.85, 1.00, 1.12, 1.35]

    for s in test_scores:
        print(policy.explain(s))