import json
from pathlib import Path

# 复用 Day 16 的 Embedding 与 FAISS 功能
from day16_faiss_retrieval import (
    MIN_SIMILARITY,
    MODEL_NAME,
    TOP_K,
    calculate_file_sha256,
    load_embedding_model,
    load_or_build_faiss_index,
    retrieve_top_chunks,
)

# 复用 Day 17 的 PDF 提取功能
from day17_pdf_extraction import (
    extract_pdf,
)

# 复用 Day 18 的递归切块功能
from day18_recursive_chunking import (
    CHUNK_OVERLAP,
    CHUNK_SIZE,
    MIN_CHUNK_SIZE,
    SEPARATORS,
    build_chunking_report,
    build_pdf_chunks,
)


PDF_FILENAME = "sample_paper.pdf"

# 修改为 True 时，无论缓存是否有效，
# 都会重新解析 PDF 并生成 Chunk。
FORCE_REBUILD = False


def get_project_root():
    """
    职责：获取项目根目录。

    输出：
        Path 类型的项目根目录
    """
    return Path(__file__).resolve().parents[1]


def load_json(input_file):
    """
    职责：读取 JSON 文件。

    输入：
        input_file：JSON 文件路径

    输出：
        Python 字典或列表
    """
    with input_file.open(
        "r",
        encoding="utf-8"
    ) as file:
        return json.load(file)


def save_json(data, output_file):
    """
    职责：保存 JSON 文件。
    """
    output_file.parent.mkdir(
        parents=True,
        exist_ok=True
    )

    with output_file.open(
        "w",
        encoding="utf-8"
    ) as file:
        json.dump(
            data,
            file,
            ensure_ascii=False,
            indent=2
        )


def build_pipeline_signature(pdf_file):
    """
    职责：
        生成当前 PDF 处理配置的签名信息。

    用途：
        判断 PDF 或切块参数是否发生变化。

    只要以下任意内容变化：
        1. PDF 文件内容
        2. chunk_size
        3. chunk_overlap
        4. minimum_chunk_size
        5. separators

    原来的 Chunk 缓存就不能继续使用。
    """
    signature = {
        "pipeline_version": 1,

        "source_pdf": pdf_file.name,

        "source_pdf_sha256": (
            calculate_file_sha256(
                pdf_file
            )
        ),

        "chunk_size": CHUNK_SIZE,

        "chunk_overlap": CHUNK_OVERLAP,

        "minimum_chunk_size": (
            MIN_CHUNK_SIZE
        ),

        "separators": SEPARATORS
    }

    return signature


def chunk_cache_is_valid(
    pdf_file,
    chunks_file,
    pipeline_metadata_file
):
    """
    职责：
        检查已有的分页和切块结果是否仍然有效。

    输出：
        True：可以直接加载现有 Chunk
        False：需要重新解析 PDF
    """
    if not chunks_file.exists():
        return False

    if not pipeline_metadata_file.exists():
        return False

    try:
        old_signature = load_json(
            pipeline_metadata_file
        )

        current_signature = (
            build_pipeline_signature(
                pdf_file
            )
        )

        if old_signature != current_signature:
            return False

        chunks_data = load_json(
            chunks_file
        )

        chunks = chunks_data.get(
            "chunks"
        )

        if not isinstance(chunks, list):
            return False

        if not chunks:
            return False

        return True

    except (
        OSError,
        ValueError,
        KeyError,
        json.JSONDecodeError
    ):
        return False


def build_chunks_from_pdf(
    pdf_file,
    pages_file,
    chunks_file,
    chunking_report_file,
    pipeline_metadata_file
):
    """
    职责：
        执行完整的 PDF 预处理：

        PDF
        → 分页文本
        → 递归切块
        → 保存页面、Chunk 和质量报告

    输出：
        chunks：全部文本块
        report：切块质量报告
    """
    print("\n正在解析 PDF……")

    extraction_data = extract_pdf(
        pdf_file
    )

    save_json(
        data=extraction_data,
        output_file=pages_file
    )

    print("正在执行递归切块……")

    chunks, skipped_pages = (
        build_pdf_chunks(
            extraction_data
        )
    )

    if not chunks:
        raise ValueError(
            "PDF 没有生成任何有效文本块"
        )

    report = build_chunking_report(
        extraction_data=extraction_data,
        chunks=chunks,
        skipped_pages=skipped_pages
    )

    chunks_data = {
        "source_file": (
            extraction_data.get(
                "source_file"
            )
        ),

        "chunking_method": (
            "custom recursive character "
            "chunking"
        ),

        "configuration": (
            report["configuration"]
        ),

        "summary": {
            "pdf_page_count": (
                report["pdf_page_count"]
            ),

            "pages_with_chunks": (
                report["pages_with_chunks"]
            ),

            "skipped_page_numbers": (
                skipped_pages
            ),

            "total_chunk_count": (
                len(chunks)
            )
        },

        "chunks": chunks
    }

    save_json(
        data=chunks_data,
        output_file=chunks_file
    )

    save_json(
        data=report,
        output_file=chunking_report_file
    )

    current_signature = (
        build_pipeline_signature(
            pdf_file
        )
    )

    save_json(
        data=current_signature,
        output_file=pipeline_metadata_file
    )

    return chunks, report


def load_or_build_chunks(
    pdf_file,
    pages_file,
    chunks_file,
    chunking_report_file,
    pipeline_metadata_file,
    force_rebuild=False
):
    """
    职责：
        缓存有效时直接加载 Chunk；
        缓存无效时重新处理 PDF。

    输出：
        chunks：全部文本块
        report：切块报告
        status：本次处理状态
    """
    cache_valid = chunk_cache_is_valid(
        pdf_file=pdf_file,
        chunks_file=chunks_file,
        pipeline_metadata_file=(
            pipeline_metadata_file
        )
    )

    if cache_valid and not force_rebuild:
        chunks_data = load_json(
            chunks_file
        )

        chunks = chunks_data["chunks"]

        if chunking_report_file.exists():
            report = load_json(
                chunking_report_file
            )
        else:
            report = {
                "total_chunk_count": (
                    len(chunks)
                )
            }

        return (
            chunks,
            report,
            "已加载现有 PDF 切块缓存"
        )

    chunks, report = build_chunks_from_pdf(
        pdf_file=pdf_file,
        pages_file=pages_file,
        chunks_file=chunks_file,
        chunking_report_file=(
            chunking_report_file
        ),
        pipeline_metadata_file=(
            pipeline_metadata_file
        )
    )

    return (
        chunks,
        report,
        "已重新解析 PDF 并生成 Chunk"
    )


def display_results(query, results):
    """
    职责：
        显示检索结果和 PDF 来源页码。
    """
    print(f"\n查询内容：{query}")

    if not results:
        print(
            "知识库中没有找到达到"
            "相关性要求的证据。"
        )
        return

    print(
        f"返回证据数量："
        f"{len(results)}"
    )

    for rank, result in enumerate(
        results,
        start=1
    ):
        print("\n" + "=" * 70)

        print(
            f"排名 {rank}："
            f"Chunk {result['chunk_id']}"
        )

        print(
            f"语义相似度："
            f"{result['similarity']:.4f}"
        )

        print(
            f"来源文件："
            f"{result['source_file']}"
        )

        print(
            f"来源页码："
            f"第 {result['page_number']} 页"
        )

        print(
            f"页内 Chunk："
            f"{result['chunk_index_on_page']}"
            f"/{result['chunk_count_on_page']}"
        )

        print(
            f"FAISS 向量编号："
            f"{result['faiss_id']}"
        )

        print("\n证据原文：")
        print(result["content"])


def save_retrieval_results(
    query,
    results,
    index,
    output_file
):
    """
    职责：
        保存完整流水线的检索结果。
    """
    result_data = {
        "pipeline": [
            "PDF extraction",
            "recursive chunking",
            "BGE embedding",
            "FAISS retrieval"
        ],

        "model_name": MODEL_NAME,

        "index_type": "IndexFlatIP",

        "top_k": TOP_K,

        "minimum_similarity": (
            MIN_SIMILARITY
        ),

        "index_vector_count": int(
            index.ntotal
        ),

        "embedding_dimension": int(
            index.d
        ),

        "query": query,

        "result_count": len(results),

        "results": results
    }

    save_json(
        data=result_data,
        output_file=output_file
    )


def display_pipeline_summary(
    pdf_file,
    chunks,
    report,
    chunk_status,
    index,
    index_status
):
    """
    职责：显示知识库准备结果。
    """
    print(
        "\n===== Day 19 完整检索流水线 ====="
    )

    print(
        f"PDF 文件：{pdf_file.name}"
    )

    print(
        f"PDF 处理状态：{chunk_status}"
    )

    print(
        f"文本块数量：{len(chunks)}"
    )

    print(
        f"切块大小：{CHUNK_SIZE}"
    )

    print(
        f"重叠大小：{CHUNK_OVERLAP}"
    )

    if "pdf_page_count" in report:
        print(
            f"PDF 总页数："
            f"{report['pdf_page_count']}"
        )

    print(
        f"FAISS 索引状态："
        f"{index_status}"
    )

    print(
        f"索引向量数量："
        f"{index.ntotal}"
    )

    print(
        f"向量维度：{index.d}"
    )

    print(
        f"最低相似度："
        f"{MIN_SIMILARITY}"
    )

    print("\n输入 q 退出程序。")


def main():
    """
    Day 19 完整流程：

    知识库准备阶段：
        1. 检查 PDF
        2. 提取分页文本
        3. 递归切块
        4. 保存页码和来源元数据
        5. 加载 Embedding 模型
        6. 建立或加载 FAISS 索引

    在线查询阶段：
        7. 接收用户问题
        8. 生成查询向量
        9. 使用 FAISS 检索
        10. 过滤低相关结果
        11. 显示证据原文和页码
        12. 保存检索结果
    """
    project_root = get_project_root()

    pdf_file = (
        project_root
        / "data"
        / "papers"
        / PDF_FILENAME
    )

    pages_file = (
        project_root
        / "data"
        / "day19_pdf_pages.json"
    )

    chunks_file = (
        project_root
        / "data"
        / "day19_pdf_chunks.json"
    )

    chunking_report_file = (
        project_root
        / "data"
        / "day19_chunking_report.json"
    )

    pipeline_metadata_file = (
        project_root
        / "data"
        / "day19_pipeline_metadata.json"
    )

    faiss_index_file = (
        project_root
        / "data"
        / "day19_faiss.index"
    )

    faiss_metadata_file = (
        project_root
        / "data"
        / "day19_faiss_metadata.json"
    )

    retrieval_results_file = (
        project_root
        / "data"
        / "day19_retrieval_results.json"
    )

    if not pdf_file.exists():
        raise FileNotFoundError(
            f"没有找到 PDF：{pdf_file}"
        )

    # 第一阶段：PDF → Chunk
    chunks, report, chunk_status = (
        load_or_build_chunks(
            pdf_file=pdf_file,
            pages_file=pages_file,
            chunks_file=chunks_file,
            chunking_report_file=(
                chunking_report_file
            ),
            pipeline_metadata_file=(
                pipeline_metadata_file
            ),
            force_rebuild=FORCE_REBUILD
        )
    )

    # 第二阶段：加载 Embedding 模型
    model = load_embedding_model(
        MODEL_NAME
    )

    # 第三阶段：Chunk → FAISS
    index, index_status = (
        load_or_build_faiss_index(
            chunks=chunks,
            model=model,

            # FAISS 用 Chunk 文件哈希判断
            # 文本块是否已经变化
            source_file=chunks_file,

            index_file=faiss_index_file,

            metadata_file=(
                faiss_metadata_file
            )
        )
    )

    display_pipeline_summary(
        pdf_file=pdf_file,
        chunks=chunks,
        report=report,
        chunk_status=chunk_status,
        index=index,
        index_status=index_status
    )

    while True:
        query = input(
            "\n请输入论文相关问题："
        ).strip()

        if query.lower() == "q":
            print("已退出论文检索系统。")
            break

        if not query:
            print("问题不能为空。")
            continue

        results = retrieve_top_chunks(
            query=query,
            chunks=chunks,
            model=model,
            index=index,
            top_k=TOP_K,
            min_similarity=(
                MIN_SIMILARITY
            )
        )

        display_results(
            query=query,
            results=results
        )

        save_retrieval_results(
            query=query,
            results=results,
            index=index,
            output_file=(
                retrieval_results_file
            )
        )

        print("\n检索结果已保存到：")
        print(retrieval_results_file)


if __name__ == "__main__":
    main()