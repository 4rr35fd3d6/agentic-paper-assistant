"""
PDF 文件校验、内容哈希、缓存路径和格式化工具。
"""

import hashlib
import re
from pathlib import Path

from app_config import (
    CACHE_ROOT,
    UPLOAD_ROOT
)


def calculate_bytes_sha256(file_bytes):
    """
    为上传文件计算内容哈希。
    """
    return hashlib.sha256(
        file_bytes
    ).hexdigest()


def sanitize_pdf_filename(filename):
    """
    清理上传文件名，仅用于界面和辅助记录。
    """
    original_path = Path(filename)

    safe_stem = re.sub(
        r"[^0-9A-Za-z\u4e00-\u9fff_-]+",
        "_",
        original_path.stem
    ).strip("_")

    if not safe_stem:
        safe_stem = "uploaded_paper"

    return f"{safe_stem[:80]}.pdf"


def validate_pdf_bytes(file_bytes):
    """
    执行基础 PDF 文件检查。
    """
    if not file_bytes:
        raise ValueError(
            "上传的 PDF 文件为空。"
        )

    if b"%PDF" not in file_bytes[:1024]:
        raise ValueError(
            "文件头中没有检测到 PDF 标识，"
            "请确认上传的是有效 PDF。"
        )


def prepare_pdf_paths(
    original_filename,
    file_bytes
):
    """
    为每个 PDF 建立独立的上传文件和缓存目录。

    相同内容始终使用相同哈希路径，
    避免不同论文共用 Chunk 或 FAISS 索引。
    """
    validate_pdf_bytes(
        file_bytes
    )

    pdf_hash = calculate_bytes_sha256(
        file_bytes
    )

    safe_filename = sanitize_pdf_filename(
        original_filename
    )

    upload_directory = UPLOAD_ROOT
    cache_directory = CACHE_ROOT / pdf_hash

    upload_directory.mkdir(
        parents=True,
        exist_ok=True
    )

    cache_directory.mkdir(
        parents=True,
        exist_ok=True
    )

    # 使用内容哈希作为稳定文件名。
    # 同一 PDF 即使换了上传名称，也能复用原缓存。
    pdf_file = (
        upload_directory
        / f"{pdf_hash}.pdf"
    )

    if not pdf_file.exists():
        pdf_file.write_bytes(
            file_bytes
        )

    paths = {
        "pdf_file": pdf_file,
        "safe_filename": safe_filename,
        "pages_file": (
            cache_directory
            / "pdf_pages.json"
        ),
        "chunks_file": (
            cache_directory
            / "pdf_chunks.json"
        ),
        "chunking_report_file": (
            cache_directory
            / "chunking_report.json"
        ),
        "pipeline_metadata_file": (
            cache_directory
            / "pipeline_metadata.json"
        ),
        "faiss_index_file": (
            cache_directory
            / "faiss.index"
        ),
        "faiss_metadata_file": (
            cache_directory
            / "faiss_metadata.json"
        )
    }

    return paths, pdf_hash


def get_pdf_page_count(
    report,
    chunks
):
    """
    兼容不同版本报告字段；
    找不到时从 Chunk 页码推断。
    """
    possible_keys = (
        "pdf_page_count",
        "total_pages",
        "page_count",
        "total_page_count"
    )

    for key in possible_keys:
        value = report.get(key)

        if value is not None:
            return value

    page_numbers = [
        chunk.get("page_number")
        for chunk in chunks
        if isinstance(
            chunk.get("page_number"),
            int
        )
    ]

    if page_numbers:
        return max(page_numbers)

    return "-"


def format_float(
    value,
    digits=4
):
    """
    安全格式化数值。
    """
    try:
        return f"{float(value):.{digits}f}"
    except (
        TypeError,
        ValueError
    ):
        return "-"
