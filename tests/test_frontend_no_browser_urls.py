"""Frontend phải tự lấy bytes ảnh từ backend: URL backend đưa cho trình duyệt sẽ vỡ trong docker
(host 'backend' chỉ phân giải được trong mạng compose)."""
import re
from pathlib import Path

SRC = (Path(__file__).parent.parent / "frontend" / "app.py").read_text(encoding="utf-8")


def test_st_image_khong_nhan_url_backend():
    assert not re.search(r"\.image\(\s*f[\"']\{API\}", SRC)
