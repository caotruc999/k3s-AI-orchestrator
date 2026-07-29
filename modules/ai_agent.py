import time
from dataclasses import dataclass


@dataclass
class AgentConfig:
    min_replicas: int = 1
    max_replicas: int = 5
    scale_up_step: int = 1
    scale_down_step: int = 1
    cooldown_seconds: int = 30
    mode: str = "AUTO"


class AIAgent:
    def __init__(self, config: AgentConfig | None = None):
        self.config = config or AgentConfig()
        self.current_replicas = self.config.min_replicas
        self.last_scale_monotonic = None

    def set_mode(self, mode: str):
        mode = mode.upper()
        if mode not in ["AUTO", "MANUAL"]:
            raise ValueError("Mode phải là AUTO hoặc MANUAL")
        self.config.mode = mode

    def cooldown_active(self) -> bool:
        if self.last_scale_monotonic is None:
            return False
        elapsed = time.monotonic() - self.last_scale_monotonic
        return elapsed < self.config.cooldown_seconds

    def apply_decision(self, decision: str) -> dict:
        decision = decision.strip()

        result = {
            "mode": self.config.mode,
            "decision": decision,
            "current_replicas_before": self.current_replicas,
            "action_taken": "None",
            "current_replicas_after": self.current_replicas,
            "reason": ""
        }

        if self.config.mode == "MANUAL":
            result["reason"] = "Hệ thống đang ở chế độ MANUAL, AI không tự scale."
            return result

        if decision == "Keep":
            result["action_taken"] = "Keep"
            result["reason"] = "Hệ thống giữ nguyên số replica."
            result["current_replicas_after"] = self.current_replicas
            return result

        if self.cooldown_active():
            result["reason"] = "Đang trong thời gian cooldown, tạm thời không scale."
            return result

        if decision == "Scale Up":
            if self.current_replicas < self.config.max_replicas:
                self.current_replicas += self.config.scale_up_step
                if self.current_replicas > self.config.max_replicas:
                    self.current_replicas = self.config.max_replicas
                self.last_scale_monotonic = time.monotonic()
                result["action_taken"] = "Scaled Up"
                result["reason"] = "AI quyết định tăng tài nguyên."
            else:
                result["reason"] = "Đã chạm max_replicas, không thể scale up thêm."

        elif decision == "Scale Down":
            if self.current_replicas > self.config.min_replicas:
                self.current_replicas -= self.config.scale_down_step
                if self.current_replicas < self.config.min_replicas:
                    self.current_replicas = self.config.min_replicas
                self.last_scale_monotonic = time.monotonic()
                result["action_taken"] = "Scaled Down"
                result["reason"] = "AI quyết định giảm tài nguyên."
            else:
                result["reason"] = "Đã chạm min_replicas, không thể scale down thêm."

        else:
            result["reason"] = f"Decision không hợp lệ: {decision}"

        result["current_replicas_after"] = self.current_replicas
        return result

    def manual_scale(self, action: str) -> dict:
        action = action.strip()

        result = {
            "mode": self.config.mode,
            "action_requested": action,
            "current_replicas_before": self.current_replicas,
            "action_taken": "None",
            "current_replicas_after": self.current_replicas,
            "reason": ""
        }

        if self.config.mode != "MANUAL":
            result["reason"] = "Chỉ được manual scale khi hệ thống đang ở chế độ MANUAL."
            return result

        if action == "Scale Up":
            if self.current_replicas < self.config.max_replicas:
                self.current_replicas += self.config.scale_up_step
                if self.current_replicas > self.config.max_replicas:
                    self.current_replicas = self.config.max_replicas
                self.last_scale_monotonic = time.monotonic()
                result["action_taken"] = "Manual Scaled Up"
                result["reason"] = "Quản trị viên tăng tài nguyên thủ công."
            else:
                result["reason"] = "Đã chạm max_replicas, không thể scale up thêm."

        elif action == "Scale Down":
            if self.current_replicas > self.config.min_replicas:
                self.current_replicas -= self.config.scale_down_step
                if self.current_replicas < self.config.min_replicas:
                    self.current_replicas = self.config.min_replicas
                self.last_scale_monotonic = time.monotonic()
                result["action_taken"] = "Manual Scaled Down"
                result["reason"] = "Quản trị viên giảm tài nguyên thủ công."
            else:
                result["reason"] = "Đã chạm min_replicas, không thể scale down thêm."

        elif action == "Keep":
            result["action_taken"] = "Keep"
            result["reason"] = "Quản trị viên giữ nguyên số replica."

        else:
            result["reason"] = f"Action không hợp lệ: {action}"

        result["current_replicas_after"] = self.current_replicas
        return result