import logging
from logging.handlers import RotatingFileHandler
from pathlib import Path


def create_file_logger(
    logger_name,
    log_file
):
    """
    创建带文件轮转的日志记录器。

    单个日志文件最大约 1 MB，
    最多保留 3 个历史日志。
    """
    logger = logging.getLogger(
        logger_name
    )

    logger.setLevel(
        logging.INFO
    )

    # Streamlit 会反复执行 app.py。
    # 避免重复添加日志处理器。
    if logger.handlers:
        return logger

    log_path = Path(
        log_file
    )

    log_path.parent.mkdir(
        parents=True,
        exist_ok=True
    )

    handler = RotatingFileHandler(
        filename=log_path,
        maxBytes=1_000_000,
        backupCount=3,
        encoding="utf-8"
    )

    formatter = logging.Formatter(
        (
            "%(asctime)s | "
            "%(levelname)s | "
            "%(name)s | "
            "%(message)s"
        )
    )

    handler.setFormatter(
        formatter
    )

    logger.addHandler(
        handler
    )

    return logger
