from dataclasses import dataclass


@dataclass
class K3sResourceState:
    deployment_name: str = "edge-ai-app"
    namespace: str = "default"
    current_replicas: int = 1
    min_replicas: int = 1
    max_replicas: int = 5


class K3sClientMock:
    def __init__(self, state: K3sResourceState | None = None):
        self.state = state or K3sResourceState()

    def get_status(self) -> dict:
        return {
            "deployment_name": self.state.deployment_name,
            "namespace": self.state.namespace,
            "current_replicas": self.state.current_replicas,
            "min_replicas": self.state.min_replicas,
            "max_replicas": self.state.max_replicas,
        }

    def scale_up(self, step: int = 1) -> dict:
        before = self.state.current_replicas
        self.state.current_replicas = min(
            self.state.current_replicas + step,
            self.state.max_replicas,
        )
        after = self.state.current_replicas

        return {
            "action": "scale_up",
            "replicas_before": before,
            "replicas_after": after,
            "changed": after != before,
        }

    def scale_down(self, step: int = 1) -> dict:
        before = self.state.current_replicas
        self.state.current_replicas = max(
            self.state.current_replicas - step,
            self.state.min_replicas,
        )
        after = self.state.current_replicas

        return {
            "action": "scale_down",
            "replicas_before": before,
            "replicas_after": after,
            "changed": after != before,
        }

    def keep(self) -> dict:
        return {
            "action": "keep",
            "replicas_before": self.state.current_replicas,
            "replicas_after": self.state.current_replicas,
            "changed": False,
        }


if __name__ == "__main__":
    client = K3sClientMock()

    print("=== K3s Client Mock Test ===")
    print(client.get_status())
    print(client.scale_up())
    print(client.scale_up())
    print(client.keep())
    print(client.scale_down())
    print(client.get_status())
