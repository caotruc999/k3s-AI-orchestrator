import unittest

import _pathsetup  # noqa: F401
from k3s_client import K3sClientMock, K3sResourceState, NODE_NAME


def make_client(**overrides):
    state = K3sResourceState(
        deployment_name="edge-ai-app",
        namespace="default",
        current_replicas=1,
        min_replicas=1,
        max_replicas=3,
    )
    for key, value in overrides.items():
        setattr(state, key, value)
    return K3sClientMock(state)


class TestK3sClientMockScaling(unittest.TestCase):
    def test_get_status_reflects_state(self):
        client = make_client()
        status = client.get_status()
        self.assertEqual(status["deployment_name"], "edge-ai-app")
        self.assertEqual(status["current_replicas"], 1)

    def test_scale_up_capped_at_max_replicas(self):
        client = make_client()
        client.scale_up()  # 1 -> 2
        result = client.scale_up()  # 2 -> 3
        self.assertEqual(result["replicas_after"], 3)

        result = client.scale_up()  # đã ở max, không tăng nữa
        self.assertFalse(result["changed"])
        self.assertEqual(result["replicas_after"], 3)

    def test_scale_down_floored_at_min_replicas(self):
        client = make_client(current_replicas=1)
        result = client.scale_down()
        self.assertFalse(result["changed"])
        self.assertEqual(result["replicas_after"], 1)

    def test_keep_never_changes_replicas(self):
        client = make_client(current_replicas=2)
        result = client.keep()
        self.assertEqual(result["action"], "keep")
        self.assertFalse(result["changed"])
        self.assertEqual(result["replicas_after"], 2)


class TestK3sClientMockPods(unittest.TestCase):
    def test_pod_count_matches_current_replicas(self):
        client = make_client(current_replicas=3)
        pods = client.get_pods(cpu_percent=50.0, memory_percent=40.0)
        self.assertEqual(len(pods), 3)

    def test_pod_fields_and_node(self):
        client = make_client(current_replicas=1)
        pods = client.get_pods(cpu_percent=90.0, memory_percent=30.0)
        pod = pods[0]
        self.assertEqual(pod["name"], "edge-ai-app-0")
        self.assertEqual(pod["node"], NODE_NAME)
        self.assertEqual(pod["cpu_percent"], 90.0)
        self.assertEqual(pod["status"], "High load")  # >= 85% -> High load

    def test_low_cpu_pod_marked_running(self):
        client = make_client(current_replicas=1)
        pods = client.get_pods(cpu_percent=20.0, memory_percent=30.0)
        self.assertEqual(pods[0]["status"], "Running")

    def test_pods_have_non_negative_age(self):
        client = make_client(current_replicas=2)
        pods = client.get_pods(cpu_percent=10.0, memory_percent=10.0)
        for pod in pods:
            self.assertIn("age_seconds", pod)
            self.assertGreaterEqual(pod["age_seconds"], 0)


class TestK3sClientMockResourceSpec(unittest.TestCase):
    def test_resource_spec_matches_sample_manifest(self):
        client = make_client()
        spec = client.get_resource_spec()
        self.assertEqual(spec["cpu_request"], 0.1)
        self.assertEqual(spec["cpu_limit"], 0.25)
        self.assertEqual(spec["memory_request"], 64.0)
        self.assertEqual(spec["memory_limit"], 128.0)


if __name__ == "__main__":
    unittest.main()
