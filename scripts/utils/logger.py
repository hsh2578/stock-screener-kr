"""
로깅 유틸리티 모듈

콘솔과 파일에 동시에 로그를 출력하며, 각 작업의 실행 시간을 측정합니다.
"""
import logging
import sys
import time
from contextlib import contextmanager
from datetime import datetime
from functools import wraps
from pathlib import Path
from typing import Callable, Generator

# 로그 디렉토리 설정
LOG_DIR = Path(__file__).parent.parent.parent / "logs"
LOG_DIR.mkdir(exist_ok=True)


def setup_logger(name: str = "stock_screener") -> logging.Logger:
    """
    로거를 초기화하고 설정합니다.

    Args:
        name: 로거 이름

    Returns:
        설정된 Logger 인스턴스

    Example:
        >>> logger = setup_logger()
        >>> logger.info("스크리너 시작")
    """
    logger = logging.getLogger(name)

    # 이미 핸들러가 설정되어 있으면 기존 로거 반환
    if logger.handlers:
        return logger

    logger.setLevel(logging.DEBUG)

    # 로그 포맷 설정
    formatter = logging.Formatter(
        fmt="%(asctime)s [%(levelname)-8s] %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S"
    )

    # 콘솔 핸들러 (INFO 이상)
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setLevel(logging.INFO)
    console_handler.setFormatter(formatter)
    logger.addHandler(console_handler)

    # 파일 핸들러 (DEBUG 이상)
    log_filename = LOG_DIR / f"{datetime.now().strftime('%Y-%m-%d')}.log"
    file_handler = logging.FileHandler(log_filename, encoding="utf-8")
    file_handler.setLevel(logging.DEBUG)
    file_handler.setFormatter(formatter)
    logger.addHandler(file_handler)

    return logger


@contextmanager
def timer(description: str, logger: logging.Logger = None) -> Generator[None, None, None]:
    """
    코드 블록의 실행 시간을 측정하는 컨텍스트 매니저.

    Args:
        description: 측정 대상 설명
        logger: 로거 인스턴스 (None이면 기본 로거 사용)

    Yields:
        None

    Example:
        >>> with timer("데이터 수집"):
        ...     fetch_data()
        [INFO] 데이터 수집 완료 (소요시간: 3.45초)
    """
    if logger is None:
        logger = setup_logger()

    start_time = time.perf_counter()
    logger.info(f"{description} 시작...")

    try:
        yield
    except Exception as e:
        elapsed = time.perf_counter() - start_time
        logger.error(f"{description} 실패 (소요시간: {elapsed:.2f}초) - {e}")
        raise
    else:
        elapsed = time.perf_counter() - start_time
        logger.info(f"{description} 완료 (소요시간: {elapsed:.2f}초)")


def log_execution_time(func: Callable) -> Callable:
    """
    함수의 실행 시간을 측정하는 데코레이터.

    Args:
        func: 측정할 함수

    Returns:
        래핑된 함수

    Example:
        >>> @log_execution_time
        ... def process_data():
        ...     pass
    """
    @wraps(func)
    def wrapper(*args, **kwargs):
        logger = setup_logger()
        start_time = time.perf_counter()

        try:
            result = func(*args, **kwargs)
            elapsed = time.perf_counter() - start_time
            logger.debug(f"{func.__name__}() 실행 완료 (소요시간: {elapsed:.2f}초)")
            return result
        except Exception as e:
            elapsed = time.perf_counter() - start_time
            logger.error(f"{func.__name__}() 실행 실패 (소요시간: {elapsed:.2f}초) - {e}")
            raise

    return wrapper


class ProgressLogger:
    """
    진행 상황을 로깅하는 클래스.

    대량의 데이터 처리 시 일정 간격으로 진행률을 로깅합니다.
    """

    def __init__(
        self,
        total: int,
        description: str = "처리 중",
        log_interval: int = 100,
        logger: logging.Logger = None
    ):
        """
        Args:
            total: 전체 작업 수
            description: 작업 설명
            log_interval: 로그 출력 간격 (기본 100개마다)
            logger: 로거 인스턴스
        """
        self.total = total
        self.description = description
        self.log_interval = log_interval
        self.logger = logger or setup_logger()
        self.current = 0
        self.start_time = time.perf_counter()

    def update(self, n: int = 1) -> None:
        """
        진행 상황을 업데이트합니다.

        Args:
            n: 증가량 (기본 1)
        """
        self.current += n

        if self.current % self.log_interval == 0 or self.current == self.total:
            elapsed = time.perf_counter() - self.start_time
            percent = (self.current / self.total) * 100
            rate = self.current / elapsed if elapsed > 0 else 0

            self.logger.info(
                f"{self.description}: {self.current}/{self.total} "
                f"({percent:.1f}%) - {rate:.1f}개/초"
            )

    def finish(self) -> None:
        """작업 완료를 로깅합니다."""
        elapsed = time.perf_counter() - self.start_time
        self.logger.info(
            f"{self.description} 완료: 총 {self.total}개 처리 "
            f"(소요시간: {elapsed:.2f}초)"
        )
