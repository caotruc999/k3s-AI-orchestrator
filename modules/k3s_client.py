<<<<<<< Updated upstream
import random

def get_live_metrics():
    """
    Hàm lấy thông số realtime từ cụm K3s.
    Người 1 sẽ dùng thư viện 'kubernetes' của Python để lấy dữ liệu thật tại đây.
    """
    # CODE MOCK (Chạy tạm để test giao diện):
    mock_cpu = random.randint(55, 85)
    mock_ram = random.randint(58, 68)
    mock_active_pods = 3  # Giả lập số lượng pod thực tế ban đầu
    
    return {
        "cpu": mock_cpu,
        "ram": mock_ram,
        "pods": mock_active_pods
    }

def execute_k3s_scale(target_replicas):
    """
    Hàm thực thi scale deployment trên K3s thật.
    Người 1 sẽ kết nối API K3s để thay đổi ReplicaSet tại đây.
    """
    # CODE MOCK (Chạy tạm):
    print(f"[K3s Client] Đã gửi API lệnh: Scale deployment lên {target_replicas} pods thành công!")
    return True
=======
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
            "max_replicas": self.state.max_replicas
        }

    def scale_up(self, step: int = 1) -> dict:
        before = self.state.current_replicas
        self.state.current_replicas = min(
            self.state.current_replicas + step,
            self.state.max_replicas
        )
        after = self.state.current_replicas

        return {
            "action": "scale_up",
            "replicas_before": before,
            "replicas_after": after,
            "changed": after != before
        }

    def scale_down(self, step: int = 1) -> dict:
        before = self.state.current_replicas
        self.state.current_replicas = max(
            self.state.current_replicas - step,
            self.state.min_replicas
        )
        after = self.state.current_replicas

        return {
            "action": "scale_down",
            "replicas_before": before,
            "replicas_after": after,
            "changed": after != before
        }

    def keep(self) -> dict:
        return {
            "action": "keep",
            "replicas_before": self.state.current_replicas,
            "replicas_after": self.state.current_replicas,
            "changed": False
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
>>>>>>> Stashed changes
