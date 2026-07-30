import smtplib
import os
import time
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart
from dotenv import load_dotenv

CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
ENV_PATH = os.path.join(CURRENT_DIR, ".env")
load_dotenv(ENV_PATH)

EMAIL_SENDER = os.environ.get("EMAIL_SENDER")
EMAIL_PASSWORD = os.environ.get("EMAIL_PASSWORD")
EMAIL_RECEIVER = os.environ.get("EMAIL_RECEIVER")
SMTP_SERVER = os.environ.get("SMTP_SERVER", "smtp.gmail.com")
SMTP_PORT = int(os.environ.get("SMTP_PORT", 587))

# Chống spam: không gửi lại email cảnh báo nếu lần gửi trước chưa đủ 5 phút
COOLDOWN_SECONDS = 300
_last_sent_time = 0


def send_alert_email(subject: str, message: str) -> bool:
    """
    Gửi email cảnh báo. Trả về True nếu gửi thành công, False nếu bỏ qua/lỗi.
    Có cơ chế cooldown để tránh việc AI liên tục gửi hàng chục email liền nhau
    khi tải dao động quanh ngưỡng cảnh báo.
    """
    global _last_sent_time

    if not EMAIL_SENDER or not EMAIL_PASSWORD or not EMAIL_RECEIVER:
        print("[Mail Notifier] Chưa cấu hình đủ thông tin trong .env, bỏ qua gửi email.")
        return False

    now = time.time()
    if now - _last_sent_time < COOLDOWN_SECONDS:
        print(f"[Mail Notifier] Bỏ qua gửi email (đang trong thời gian cooldown "
              f"{COOLDOWN_SECONDS}s để tránh spam).")
        return False

    try:
        msg = MIMEMultipart()
        msg["From"] = EMAIL_SENDER
        msg["To"] = EMAIL_RECEIVER
        msg["Subject"] = subject
        msg.attach(MIMEText(message, "plain", "utf-8"))

        with smtplib.SMTP(SMTP_SERVER, SMTP_PORT) as server:
            server.starttls()
            server.login(EMAIL_SENDER, EMAIL_PASSWORD)
            server.send_message(msg)

        _last_sent_time = now
        print(f"[Mail Warning] Đã gửi email cảnh báo tới {EMAIL_RECEIVER}")
        return True

    except Exception as e:
        print(f"[Mail Warning] Lỗi khi gửi email: {e}")
        return False


def build_alert_message(current_cpu: float, predicted_cpu: float, threshold: float) -> tuple[str, str]:
    """Tạo sẵn tiêu đề + nội dung email theo mẫu chuẩn, dùng chung cho các nơi gọi trong app.py."""
    subject = "[CẢNH BÁO] Hệ thống Edge AI cần can thiệp"
    body = (
        f"Hệ thống phát hiện tải sắp vượt ngưỡng an toàn.\n\n"
        f"- CPU hiện tại: {current_cpu}%\n"
        f"- CPU dự báo (vài phút tới): {predicted_cpu}%\n"
        f"- Ngưỡng cảnh báo: {threshold}%\n\n"
        f"Vui lòng kiểm tra dashboard và can thiệp nếu cần thiết."
    )
    return subject, body