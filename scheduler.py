"""
주식 스크리너 자동 업데이트 스케줄러

스케줄:
- 저평가 우량주 (TTM): 매주 토요일 12:00
- 60주선 우량주: 매주 토요일 12:30 (저평가 우량주 이후)
- 기타 스크리너: 매일 12:00, 14:00, 16:00, 18:00 (장중/장마감 후)
"""

import schedule
import time
import subprocess
import sys
import os
from datetime import datetime

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))


def log(message: str):
    """로그 출력"""
    timestamp = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
    print(f"[{timestamp}] {message}")


def run_value_screener():
    """저평가 우량주 스크리너 실행 (TTM)"""
    log("저평가 우량주 스크리너 시작 (TTM)...")
    try:
        result = subprocess.run(
            [sys.executable, os.path.join(SCRIPT_DIR, 'test_value_screener.py')],
            cwd=SCRIPT_DIR,
            capture_output=True,
            text=True,
            timeout=1800  # 30분 타임아웃
        )
        if result.returncode == 0:
            log("저평가 우량주 스크리너 완료")
        else:
            log(f"저평가 우량주 스크리너 오류: {result.stderr}")
    except subprocess.TimeoutExpired:
        log("저평가 우량주 스크리너 타임아웃 (30분 초과)")
    except Exception as e:
        log(f"저평가 우량주 스크리너 실행 실패: {e}")


def run_ma60w_quality_screener():
    """60주선 우량주 스크리너 실행"""
    log("60주선 우량주 스크리너 시작...")
    try:
        result = subprocess.run(
            [sys.executable, os.path.join(SCRIPT_DIR, 'ma60w_quality.py')],
            cwd=SCRIPT_DIR,
            capture_output=True,
            text=True,
            timeout=1800  # 30분 타임아웃
        )
        if result.returncode == 0:
            log("60주선 우량주 스크리너 완료")
        else:
            log(f"60주선 우량주 스크리너 오류: {result.stderr}")
    except subprocess.TimeoutExpired:
        log("60주선 우량주 스크리너 타임아웃 (30분 초과)")
    except Exception as e:
        log(f"60주선 우량주 스크리너 실행 실패: {e}")


def run_other_screeners():
    """기타 스크리너 실행 (박스권, 돌파, 풀백 등)"""
    log("기타 스크리너 시작...")
    try:
        result = subprocess.run(
            [sys.executable, os.path.join(SCRIPT_DIR, 'run_screeners.py')],
            cwd=SCRIPT_DIR,
            capture_output=True,
            text=True,
            timeout=3600  # 1시간 타임아웃
        )
        if result.returncode == 0:
            log("기타 스크리너 완료")
        else:
            log(f"기타 스크리너 오류: {result.stderr}")
    except subprocess.TimeoutExpired:
        log("기타 스크리너 타임아웃 (1시간 초과)")
    except Exception as e:
        log(f"기타 스크리너 실행 실패: {e}")


def setup_schedule():
    """스케줄 설정"""
    # 저평가 우량주: 매주 토요일 12:00
    schedule.every().saturday.at("12:00").do(run_value_screener)

    # 60주선 우량주: 매주 토요일 12:30 (저평가 우량주 이후)
    schedule.every().saturday.at("12:30").do(run_ma60w_quality_screener)

    # 기타 스크리너: 매일 12:00, 14:00, 16:00, 18:00
    schedule.every().day.at("12:00").do(run_other_screeners)
    schedule.every().day.at("14:00").do(run_other_screeners)
    schedule.every().day.at("16:00").do(run_other_screeners)
    schedule.every().day.at("18:00").do(run_other_screeners)

    log("스케줄 설정 완료:")
    log("  - 저평가 우량주 (TTM): 매주 토요일 12:00")
    log("  - 60주선 우량주: 매주 토요일 12:30")
    log("  - 기타 스크리너: 매일 12:00, 14:00, 16:00, 18:00")


def main():
    print("=" * 60)
    print("주식 스크리너 자동 업데이트 스케줄러")
    print("=" * 60)

    setup_schedule()

    log("스케줄러 시작 (Ctrl+C로 종료)")
    print()

    # 다음 실행 시간 표시
    jobs = schedule.get_jobs()
    for job in jobs:
        log(f"다음 실행: {job.next_run}")

    print()

    # 스케줄 루프
    while True:
        schedule.run_pending()
        time.sleep(60)  # 1분마다 체크


if __name__ == '__main__':
    main()
