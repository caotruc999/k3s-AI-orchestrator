import logging
import time
from dataclasses import dataclass
from datetime import datetime, timezone

logger = logging.getLogger(__name__)

# Cụm demo chỉ có 1 node duy nhất (quyết định phạm vi đồ án, không phải multi-node).
NODE_NAME = "edge-node-01"


@dataclass
class K3sResourceState:
    deployment_name: str = "edge-ai-app"
    namespace: str = "default"
    current_replicas: int = 1
    min_replicas: int = 1
    max_replicas: int = 5


# Mặc định request/limit của deployment mẫu k8s/edge-ai-app-deployment.yaml
# (cores, MiB) — dùng làm giá trị mock nhất quán với manifest thật.
MOCK_RESOURCE_SPEC = {
    "cpu_request": 0.1,
    "cpu_limit": 0.25,
    "memory_request": 64.0,
    "memory_limit": 128.0,
}


class K3sClientMock:
    def __init__(self, state: K3sResourceState | None = None):
        self.state = state or K3sResourceState()
        self._created_at = time.monotonic()

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

    def get_pods(self, cpu_percent: float = 0.0, memory_percent: float = 0.0) -> list:
        """
        TODO (Người 1): thay bằng dữ liệu pod thật lấy qua Kubernetes API
        (metrics-server, list_namespaced_pod + list pod metrics). Hiện tại đây
        là mock: pod đầu tiên phản ánh CPU/RAM host thật, các pod còn lại lệch
        nhẹ theo index để giao diện có dữ liệu khác nhau giữa các pod trong lúc
        chờ tích hợp K8s API thật.
        """
        pods = []
        age_seconds = time.monotonic() - self._created_at
        for i in range(self.state.current_replicas):
            offset = i * 6
            pod_cpu = max(0.0, min(100.0, cpu_percent - offset if i > 0 else cpu_percent))
            pod_ram = max(0.0, min(100.0, memory_percent - offset if i > 0 else memory_percent))
            status = "High load" if pod_cpu >= 85 else "Running"

            pods.append({
                "name": f"{self.state.deployment_name}-{i}",
                "node": NODE_NAME,
                "cpu_percent": round(pod_cpu, 1),
                "memory_percent": round(pod_ram, 1),
                "status": status,
                "age_seconds": round(age_seconds, 1),
            })
        return pods

    def get_resource_spec(self) -> dict:
        """Mock: trả cố định theo k8s/edge-ai-app-deployment.yaml (xem
        MOCK_RESOURCE_SPEC) vì không có deployment thật để đọc."""
        return dict(MOCK_RESOURCE_SPEC)


def _parse_cpu_quantity(value: str | None) -> float | None:
    """Quy đổi chuỗi CPU quantity của K8s ('500m', '1', '2') ra millicores."""
    if not value:
        return None
    value = value.strip()
    if value.endswith("n"):
        return float(value[:-1]) / 1_000_000
    if value.endswith("m"):
        return float(value[:-1])
    return float(value) * 1000


def _parse_memory_quantity(value: str | None) -> float | None:
    """Quy đổi chuỗi memory quantity của K8s ('64Mi', '1Gi', '500Ki') ra MiB."""
    if not value:
        return None
    value = value.strip()
    units_binary = {"Ki": 1 / 1024, "Mi": 1, "Gi": 1024, "Ti": 1024 * 1024}
    units_decimal = {"K": 1000 / 1024 ** 2, "M": 1000 ** 2 / 1024 ** 2, "G": 1000 ** 3 / 1024 ** 2}

    for suffix, factor in {**units_binary, **units_decimal}.items():
        if value.endswith(suffix):
            return float(value[: -len(suffix)]) * factor

    # Không có suffix -> đơn vị byte thô
    return float(value) / 1024 ** 2


def _resource_percent(usage: float | None, limit: float | None, request: float | None) -> float | None:
    """% sử dụng so với limit (ưu tiên) hoặc request của container. None nếu
    thiếu dữ liệu để tính (ví dụ manifest không khai limit/request)."""
    baseline = limit or request
    if usage is None or not baseline:
        return None
    return max(0.0, min(100.0, usage / baseline * 100))


class K3sClient:
    """
    Client K3s THẬT: đọc CPU/RAM per-pod qua metrics-server (metrics.k8s.io)
    và scale deployment thật bằng patch_namespaced_deployment_scale.

    Không tự bắt exception "cluster không kết nối được" ở đây — để
    create_k3s_client() phía dưới quyết định fallback, tránh nuốt lỗi âm thầm.
    """

    def __init__(self, state: K3sResourceState | None = None):
        from kubernetes import client as k8s_client
        from kubernetes import config as k8s_config

        self.state = state or K3sResourceState()

        try:
            k8s_config.load_incluster_config()
        except Exception:
            k8s_config.load_kube_config()

        self._client = k8s_client
        self.apps_api = k8s_client.AppsV1Api()
        self.core_api = k8s_client.CoreV1Api()
        self.metrics_api = k8s_client.CustomObjectsApi()

        # Gọi thật ngay lúc khởi tạo để create_k3s_client() phát hiện được
        # cluster có kết nối được hay không (fail-fast).
        self._refresh_replicas_from_cluster()

    def _refresh_replicas_from_cluster(self):
        # _request_timeout ngắn để create_k3s_client() fallback về mock nhanh
        # khi cluster không kết nối được, thay vì treo theo timeout OS/TCP mặc định.
        deployment = self.apps_api.read_namespaced_deployment(
            self.state.deployment_name, self.state.namespace, _request_timeout=5
        )
        if deployment.spec.replicas is not None:
            self.state.current_replicas = deployment.spec.replicas

    def get_status(self) -> dict:
        self._refresh_replicas_from_cluster()
        return {
            "deployment_name": self.state.deployment_name,
            "namespace": self.state.namespace,
            "current_replicas": self.state.current_replicas,
            "min_replicas": self.state.min_replicas,
            "max_replicas": self.state.max_replicas,
        }

    def _scale_to(self, target_replicas: int, action: str) -> dict:
        before = self.state.current_replicas
        target_replicas = max(self.state.min_replicas, min(target_replicas, self.state.max_replicas))

        self.apps_api.patch_namespaced_deployment_scale(
            name=self.state.deployment_name,
            namespace=self.state.namespace,
            body={"spec": {"replicas": target_replicas}},
            _request_timeout=10,
        )
        self.state.current_replicas = target_replicas

        return {
            "action": action,
            "replicas_before": before,
            "replicas_after": target_replicas,
            "changed": target_replicas != before,
        }

    def scale_up(self, step: int = 1) -> dict:
        return self._scale_to(self.state.current_replicas + step, "scale_up")

    def scale_down(self, step: int = 1) -> dict:
        return self._scale_to(self.state.current_replicas - step, "scale_down")

    def keep(self) -> dict:
        return {
            "action": "keep",
            "replicas_before": self.state.current_replicas,
            "replicas_after": self.state.current_replicas,
            "changed": False,
        }

    def get_pods(self, cpu_percent: float = 0.0, memory_percent: float = 0.0) -> list:
        """
        cpu_percent/memory_percent giữ trong signature để tương thích với
        K3sClientMock, dùng làm giá trị dự phòng khi metrics-server chưa sẵn
        sàng hoặc pod chưa khai resources.requests/limits.
        """
        label_selector = f"app={self.state.deployment_name}"
        pods = self.core_api.list_namespaced_pod(
            self.state.namespace, label_selector=label_selector, _request_timeout=10
        ).items

        try:
            raw_metrics = self.metrics_api.list_namespaced_custom_object(
                "metrics.k8s.io", "v1beta1", self.state.namespace, "pods",
                _request_timeout=5,
            )
            metrics_by_name = {item["metadata"]["name"]: item for item in raw_metrics.get("items", [])}
        except Exception as exc:
            logger.warning("metrics-server không khả dụng, dùng CPU/RAM host làm dự phòng: %s", exc)
            metrics_by_name = {}

        result = []
        for pod in pods:
            name = pod.metadata.name
            node = pod.spec.node_name or NODE_NAME
            container_spec = (pod.spec.containers or [None])[0]
            resources = getattr(container_spec, "resources", None) if container_spec else None
            limits = (resources.limits or {}) if resources else {}
            requests = (resources.requests or {}) if resources else {}

            pod_metrics = metrics_by_name.get(name)
            cpu_pct = mem_pct = None

            if pod_metrics and pod_metrics.get("containers"):
                usage = pod_metrics["containers"][0]["usage"]
                cpu_usage = _parse_cpu_quantity(usage.get("cpu"))
                mem_usage = _parse_memory_quantity(usage.get("memory"))

                cpu_pct = _resource_percent(
                    cpu_usage,
                    _parse_cpu_quantity(limits.get("cpu")),
                    _parse_cpu_quantity(requests.get("cpu")),
                )
                mem_pct = _resource_percent(
                    mem_usage,
                    _parse_memory_quantity(limits.get("memory")),
                    _parse_memory_quantity(requests.get("memory")),
                )

            if cpu_pct is None:
                cpu_pct = cpu_percent
            if mem_pct is None:
                mem_pct = memory_percent

            phase = pod.status.phase if pod.status else "Unknown"
            status = "Running" if phase == "Running" and cpu_pct < 85 else ("High load" if cpu_pct >= 85 else phase)

            created_at = pod.metadata.creation_timestamp
            age_seconds = (datetime.now(timezone.utc) - created_at).total_seconds() if created_at else None

            result.append({
                "name": name,
                "node": node,
                "cpu_percent": round(cpu_pct, 1),
                "memory_percent": round(mem_pct, 1),
                "status": status,
                "age_seconds": round(age_seconds, 1) if age_seconds is not None else None,
            })
        return result

    def get_resource_spec(self) -> dict:
        """Đọc cpu/memory request+limit THẬT của container đầu tiên trong pod
        template của deployment (dùng làm feature cho model AI thay vì bịa số)."""
        deployment = self.apps_api.read_namespaced_deployment(
            self.state.deployment_name, self.state.namespace, _request_timeout=5
        )
        containers = deployment.spec.template.spec.containers or []
        resources = containers[0].resources if containers else None
        limits = (resources.limits or {}) if resources else {}
        requests = (resources.requests or {}) if resources else {}

        cpu_request = _parse_cpu_quantity(requests.get("cpu"))
        cpu_limit = _parse_cpu_quantity(limits.get("cpu"))
        memory_request = _parse_memory_quantity(requests.get("memory"))
        memory_limit = _parse_memory_quantity(limits.get("memory"))

        return {
            "cpu_request": (cpu_request or 0) / 1000,   # millicores -> cores, khớp đơn vị dataset
            "cpu_limit": (cpu_limit or 0) / 1000,
            "memory_request": memory_request or 0,      # đã ở MiB
            "memory_limit": memory_limit or 0,
        }


def create_k3s_client(state: K3sResourceState | None = None):
    """
    Factory: ưu tiên K3sClient thật (đọc/patch cluster K8s qua kubeconfig hoặc
    in-cluster config). Nếu không kết nối được cluster (không có kubeconfig,
    cluster đang tắt, deployment chưa được apply...) tự động rơi về
    K3sClientMock để dashboard vẫn chạy được thay vì crash toàn bộ server.

    Trả về (client, mode) với mode ∈ {"real", "mock"} — /status expose lại
    giá trị này để dashboard không "nói dối" là đang đọc cluster thật khi
    thực ra đang chạy mock.
    """
    try:
        return K3sClient(state), "real"
    except Exception as exc:
        logger.warning(
            "Không kết nối được K3s cluster thật (%s) — dùng K3sClientMock. "
            "Kiểm tra: cluster đã chạy? kubeconfig đúng context? deployment '%s' "
            "đã được kubectl apply chưa?",
            exc,
            (state or K3sResourceState()).deployment_name,
        )
        return K3sClientMock(state), "mock"


if __name__ == "__main__":
    client, mode = create_k3s_client()

    print(f"=== K3s Client Test (mode={mode}) ===")
    print(client.get_status())
    print(client.scale_up())
    print(client.scale_up())
    print(client.keep())
    print(client.scale_down())
    print(client.get_status())
