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