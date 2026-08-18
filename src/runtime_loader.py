"""终端入口使用的真实论文缓存加载器。"""

from __future__ import annotations

from pathlib import Path
from typing import Any

from app_config import CACHE_ROOT, UPLOAD_ROOT


def find_latest_ready_pdf() -> tuple[Path, dict[str, Path], str]:
    """查找最近使用且缓存完整的论文。"""
    pdf_files = sorted(
        UPLOAD_ROOT.glob("*.pdf"),
        key=lambda path: path.stat().st_mtime,
        reverse=True,
    )
    if not pdf_files:
        raise FileNotFoundError(
            "data/day23_uploads 中没有 PDF，请先在 Streamlit 中建立知识库。"
        )

    for pdf_file in pdf_files:
        pdf_hash = pdf_file.stem
        cache_directory = CACHE_ROOT / pdf_hash
        paths = {
            "pdf_file": pdf_file,
            "pages_file": cache_directory / "pdf_pages.json",
            "chunks_file": cache_directory / "pdf_chunks.json",
            "chunking_report_file": cache_directory / "chunking_report.json",
            "pipeline_metadata_file": cache_directory / "pipeline_metadata.json",
            "faiss_index_file": cache_directory / "faiss.index",
            "faiss_metadata_file": cache_directory / "faiss_metadata.json",
        }
        required = (
            paths["pages_file"],
            paths["chunks_file"],
            paths["pipeline_metadata_file"],
            paths["faiss_index_file"],
            paths["faiss_metadata_file"],
        )
        if all(path.exists() for path in required):
            return pdf_file, paths, pdf_hash

    raise FileNotFoundError(
        "找到了上传 PDF，但没有完整知识库缓存，请在 Streamlit 中重新建立。"
    )


def load_real_runtime(rag_backend) -> tuple[dict[str, Any], dict[str, Any]]:
    """加载 Chunk、Embedding、FAISS 与 Reranker。"""
    pdf_file, paths, pdf_hash = find_latest_ready_pdf()

    print("\n===== 当前论文缓存 =====")
    print(f"PDF：{pdf_file.name}")
    print(f"缓存目录：{paths['chunks_file'].parent}")

    print("\n===== 加载 Chunk =====")
    chunks, report, chunk_status = rag_backend.load_or_build_chunks(
        pdf_file=paths["pdf_file"],
        pages_file=paths["pages_file"],
        chunks_file=paths["chunks_file"],
        chunking_report_file=paths["chunking_report_file"],
        pipeline_metadata_file=paths["pipeline_metadata_file"],
        force_rebuild=False,
    )
    print(f"状态：{chunk_status}")
    print(f"Chunk 数量：{len(chunks)}")

    print("\n===== 加载 Embedding =====")
    embedding_model = rag_backend.load_embedding_model(
        rag_backend.EMBEDDING_MODEL_NAME
    )

    print("\n===== 加载 FAISS =====")
    index, index_status = rag_backend.load_or_build_faiss_index(
        chunks=chunks,
        model=embedding_model,
        source_file=paths["chunks_file"],
        index_file=paths["faiss_index_file"],
        metadata_file=paths["faiss_metadata_file"],
    )
    print(f"状态：{index_status}")
    print(f"向量数量：{index.ntotal}")

    print("\n===== 加载 Reranker =====")
    tokenizer, reranker_model, device = rag_backend.load_reranker()

    runtime = {
        "pdf_hash": pdf_hash,
        "original_filename": pdf_file.name,
        "paths": paths,
        "chunks": chunks,
        "report": report,
        "chunk_status": chunk_status,
        "index": index,
        "index_status": index_status,
        "embedding_model": embedding_model,
    }
    reranker = {
        "tokenizer": tokenizer,
        "model": reranker_model,
        "device": device,
    }
    return runtime, reranker
