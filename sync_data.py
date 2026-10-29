"""Replit Scheduled Deployment에서 매일 실행할 수집 작업."""
import logging
from auto_sync import sync_all

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
result = sync_all()
print("살레봇 자동 수집 결과:")
for name, status in result.items():
    print(f"- {name}: {status}")
