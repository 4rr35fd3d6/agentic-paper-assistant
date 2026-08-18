import hashlib
import json
from pathlib import Path


# 每个基础 Chunk 最多约 450 个字符
CHUNK_SIZE = 450

# 相邻 Chunk 之间保留约 80 个字符
CHUNK_OVERLAP = 80

# 最后一个 Chunk 太短时，尝试与前一个合并
MIN_CHUNK_SIZE = 80

# 从粗粒度到细粒度依次尝试
SEPARATORS = [
    "\n\n",   # 段落
    "\n",     # 单行
    "。",     # 中文句号
    "！",
    "？",
    "；",
    ". ",     # 英文句号
    "! ",
    "? ",
    "; ",
    "，",     # 中文逗号
    ", ",
    " ",      # 空格
    ""        # 最后兜底：按字符强制切分
]


def get_project_root():
    """
    职责：获取项目根目录。

    输出：
        Path 类型的项目根目录
    """
    return Path(__file__).resolve().parents[1]


def load_pdf_pages(input_file):
    """
    职责：
        读取 Day 17 生成的分页 PDF JSON。

    输入：
        input_file：day17_pdf_pages.json 路径

    输出：
        extraction_data：完整 PDF 提取数据
    """
    if not input_file.exists():
        raise FileNotFoundError(
            f"没有找到 Day 17 输出文件：{input_file}"
        )

    with input_file.open(
        "r",
        encoding="utf-8"
    ) as file:
        extraction_data = json.load(file)

    pages = extraction_data.get("pages")

    if not isinstance(pages, list):
        raise ValueError(
            "输入 JSON 中缺少有效的 pages 列表"
        )

    if not pages:
        raise ValueError(
            "输入 JSON 中没有可处理的页面"
        )

    return extraction_data


def save_json(data, output_file):
    """
    职责：将数据保存为 JSON 文件。
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


def split_keep_separator(
    text,
    separator
):
    """
    职责：
        按指定分隔符拆分文本，
        同时尽量把分隔符保留在前一个片段末尾。

    示例：
        输入：
            text = "第一句。第二句。"
            separator = "。"

        输出：
            ["第一句。", "第二句。"]
    """
    raw_parts = text.split(separator)

    # 完全找不到当前分隔符
    if len(raw_parts) == 1:
        return [text]

    parts = []

    for index, part in enumerate(
        raw_parts
    ):
        # 除最后一个片段外，
        # 把分隔符重新加回片段末尾
        if index < len(raw_parts) - 1:
            part = part + separator

        # 忽略完全为空的片段
        if part.strip():
            parts.append(part)

    return parts


def hard_split_text(
    text,
    max_length
):
    """
    职责：
        当前面的语义分隔符都无法使用时，
        按固定字符数强制切分。

    这是递归切块的最后兜底方案。
    """
    chunks = []

    start = 0

    while start < len(text):
        end = start + max_length

        chunk = text[start:end].strip()

        if chunk:
            chunks.append(chunk)

        start = end

    return chunks


def recursive_split_text(
    text,
    separators,
    max_length
):
    """
    职责：
        递归选择分隔符，将文本切成不超过
        max_length 的基础 Chunk。

    核心思想：
        当前分隔符切不开或切后仍然太长，
        就继续尝试下一级、更细的分隔符。

    输入：
        text：需要切分的文本
        separators：分隔符优先级列表
        max_length：基础 Chunk 最大长度

    输出：
        chunks：基础 Chunk 列表
    """
    text = text.strip()

    if not text:
        return []

    # 文本已经足够短，无需继续切分
    if len(text) <= max_length:
        return [text]

    # 分隔符全部用完，强制按字符切分
    if not separators:
        return hard_split_text(
            text=text,
            max_length=max_length
        )

    current_separator = separators[0]

    remaining_separators = (
        separators[1:]
    )

    # 空字符串代表最终强制切分
    if current_separator == "":
        return hard_split_text(
            text=text,
            max_length=max_length
        )

    parts = split_keep_separator(
        text=text,
        separator=current_separator
    )

    # 当前文本中没有这个分隔符，
    # 继续尝试更细粒度的分隔符
    if len(parts) == 1:
        return recursive_split_text(
            text=text,
            separators=remaining_separators,
            max_length=max_length
        )

    chunks = []
    current_chunk = ""

    for part in parts:
        if not part.strip():
            continue

        candidate_length = (
            len(current_chunk)
            + len(part)
        )

        # 当前片段加入后仍未超过上限
        if candidate_length <= max_length:
            current_chunk = (
                current_chunk + part
            )
            continue

        # 当前 Chunk 已经积累了内容，
        # 先保存它
        if current_chunk.strip():
            chunks.append(
                current_chunk.strip()
            )

            current_chunk = ""

        # 当前单个片段本身仍然太长，
        # 使用更细一级分隔符递归处理
        if len(part) > max_length:
            nested_chunks = (
                recursive_split_text(
                    text=part,
                    separators=(
                        remaining_separators
                    ),
                    max_length=max_length
                )
            )

            # 前面的递归结果已经完整，
            # 可以直接保存
            chunks.extend(
                nested_chunks[:-1]
            )

            # 最后一个递归结果暂时保留，
            # 尝试与后续片段继续合并
            if nested_chunks:
                current_chunk = (
                    nested_chunks[-1]
                )

        else:
            current_chunk = part

    if current_chunk.strip():
        chunks.append(
            current_chunk.strip()
        )

    return chunks


def merge_short_last_chunk(
    chunks,
    max_length,
    min_length
):
    """
    职责：
        最后一个 Chunk 太短时，
        尝试将其合并到前一个 Chunk。

    只有合并后不超过 max_length 才执行。
    """
    if len(chunks) < 2:
        return chunks

    last_chunk = chunks[-1]

    if len(last_chunk) >= min_length:
        return chunks

    previous_chunk = chunks[-2]

    combined_chunk = (
        previous_chunk
        + "\n"
        + last_chunk
    ).strip()

    if len(combined_chunk) <= max_length:
        chunks[-2] = combined_chunk
        chunks.pop()

    return chunks


def get_semantic_overlap(
    previous_chunk,
    overlap_length
):
    """
    职责：
        从上一个 Chunk 末尾获取重叠内容。

    与直接截取最后 overlap_length 个字符相比，
    这里尽量从标点或换行之后开始，
    减少从词语中间截断的情况。
    """
    if overlap_length <= 0:
        return ""

    if len(previous_chunk) <= overlap_length:
        return previous_chunk.strip()

    tail = previous_chunk[
        -overlap_length:
    ]

    overlap_separators = [
        "\n\n",
        "\n",
        "。",
        "！",
        "？",
        "；",
        ". ",
        "! ",
        "? ",
        "; ",
        "，",
        ", ",
        " "
    ]

    minimum_useful_length = max(
        20,
        overlap_length // 3
    )

    for separator in overlap_separators:
        separator_position = tail.find(
            separator
        )

        if separator_position == -1:
            continue

        candidate = tail[
            separator_position
            + len(separator):
        ].strip()

        if (
            len(candidate)
            >= minimum_useful_length
        ):
            return candidate

    # 找不到合适边界时，
    # 退回到普通字符截取
    return tail.strip()


def add_chunk_overlap(
    base_chunks,
    overlap_length
):
    """
    职责：
        给相邻基础 Chunk 添加重叠上下文。

    输出：
        列表中的每个元素为：
        {
            "content": Chunk 内容,
            "overlap_char_count": 重叠字符数
        }
    """
    if not base_chunks:
        return []

    overlapped_chunks = [
        {
            "content": base_chunks[0],
            "overlap_char_count": 0
        }
    ]

    for index in range(
        1,
        len(base_chunks)
    ):
        previous_chunk = (
            base_chunks[index - 1]
        )

        current_chunk = base_chunks[index]

        overlap_text = get_semantic_overlap(
            previous_chunk=previous_chunk,
            overlap_length=overlap_length
        )

        if overlap_text:
            chunk_content = (
                overlap_text
                + "\n"
                + current_chunk
            ).strip()
        else:
            chunk_content = (
                current_chunk.strip()
            )

        overlapped_chunks.append(
            {
                "content": chunk_content,
                "overlap_char_count": (
                    len(overlap_text)
                )
            }
        )

    return overlapped_chunks


def calculate_text_sha256(text):
    """
    职责：
        为 Chunk 内容计算 SHA-256 指纹。

    内容发生变化时，指纹通常也会变化。
    后面可以用它判断 Chunk 是否被修改。
    """
    return hashlib.sha256(
        text.encode("utf-8")
    ).hexdigest()


def build_pdf_chunks(
    extraction_data
):
    """
    职责：
        对 PDF 每一页进行递归切块，
        并给每个 Chunk 添加来源元数据。

    输出：
        chunks：全部 Chunk
        skipped_pages：没有文本、被跳过的页码
    """
    source_file = extraction_data.get(
        "source_file",
        "unknown.pdf"
    )

    all_chunks = []
    skipped_pages = []#无文字页码

    global_chunk_id = 1

    for page in extraction_data["pages"]:
        page_number = page["page_number"]
        page_index = page["page_index"]

        page_text = page.get(
            "text",
            ""
        ).strip()

        # 没有文字的页面暂时不切块
        if not page_text:
            skipped_pages.append(
                page_number
            )
            continue

        base_chunks = recursive_split_text(
            text=page_text,
            separators=SEPARATORS,
            max_length=CHUNK_SIZE
        )

        base_chunks = merge_short_last_chunk(
            chunks=base_chunks,
            max_length=CHUNK_SIZE,
            min_length=MIN_CHUNK_SIZE
        )

        page_chunks = add_chunk_overlap(
            base_chunks=base_chunks,
            overlap_length=CHUNK_OVERLAP
        )

        page_chunk_count = len(
            page_chunks
        )

        for local_index, chunk_data in enumerate(
            page_chunks,
            start=1
        ):
            content = chunk_data["content"]

            chunk = {
                "chunk_id": global_chunk_id,

                "source_file": source_file,

                # 当前设计中 Chunk 不跨页所以page_start == page_end == page_number
                "page_number": page_number,
                "page_index": page_index,
                "page_start": page_number,
                "page_end": page_number,

                "chunk_index_on_page": (
                    local_index
                ),#该页第几个chunk

                "chunk_count_on_page": (
                    page_chunk_count
                ),#这页一共多少chunk

                "char_count": len(content),#添加啊重叠后的长度

                "overlap_char_count": (
                    chunk_data[
                        "overlap_char_count"
                    ]
                ),

                "needs_ocr_review": page.get(
                    "needs_ocr_review",
                    False
                ),#ocr标记

                "source_reference": (
                    f"{source_file}"
                    f"#page={page_number}"
                ),

                "content_sha256": (
                    calculate_text_sha256(
                        content
                    )
                ),

                "content": content
            }

            all_chunks.append(chunk)

            global_chunk_id += 1

    return all_chunks, skipped_pages


def build_chunking_report(
    extraction_data,
    chunks,
    skipped_pages
):
    """
    职责：
        统计递归切块质量，
        生成便于检查的报告。
    """
    chunk_lengths = [
        chunk["char_count"]
        for chunk in chunks
    ]

    pages_with_chunks = {
        chunk["page_number"]
        for chunk in chunks
    }

    # 基础 Chunk 最长为 CHUNK_SIZE；
    # 添加重叠后允许再增加 CHUNK_OVERLAP，
    # 外加一个换行符。
    expected_max_length = (
        CHUNK_SIZE
        + CHUNK_OVERLAP
        + 1
    )

    oversized_chunks = [
        chunk["chunk_id"]
        for chunk in chunks
        if (
            chunk["char_count"]
            > expected_max_length
        )
    ]

    if chunk_lengths:
        minimum_length = min(
            chunk_lengths
        )

        maximum_length = max(
            chunk_lengths
        )

        average_length = round(
            sum(chunk_lengths)
            / len(chunk_lengths),
            2
        )
    else:
        minimum_length = 0
        maximum_length = 0
        average_length = 0

    report = {
        "source_file": extraction_data.get(
            "source_file"
        ),

        "chunking_method": (
            "custom recursive character "
            "chunking"
        ),

        "configuration": {
            "chunk_size": CHUNK_SIZE,
            "chunk_overlap": (
                CHUNK_OVERLAP
            ),
            "minimum_chunk_size": (
                MIN_CHUNK_SIZE
            ),
            "separators": SEPARATORS
        },

        "pdf_page_count": extraction_data.get(
            "page_count",
            len(extraction_data["pages"])
        ),

        "pages_with_chunks": len(
            pages_with_chunks
        ),

        "skipped_page_count": len(
            skipped_pages
        ),

        "skipped_page_numbers": (
            skipped_pages
        ),

        "total_chunk_count": len(
            chunks
        ),

        "minimum_chunk_length": (
            minimum_length
        ),

        "maximum_chunk_length": (
            maximum_length
        ),

        "average_chunk_length": (
            average_length
        ),

        "expected_maximum_length": (
            expected_max_length
        ),

        "oversized_chunk_count": len(
            oversized_chunks
        ),

        "oversized_chunk_ids": (
            oversized_chunks
        ),

        "ocr_review_page_numbers": [
            page["page_number"]
            for page in extraction_data["pages"]
            if page.get(
                "needs_ocr_review",
                False
            )
        ]
    }

    return report


def display_summary(report):
    """
    职责：在终端显示切块摘要。
    """
    print(
        "\n===== Day 18 PDF 递归切块 ====="
    )

    print(
        f"来源文件："
        f"{report['source_file']}"
    )

    print(
        f"PDF 总页数："
        f"{report['pdf_page_count']}"
    )

    print(
        f"成功生成 Chunk 的页面："
        f"{report['pages_with_chunks']}"
    )

    print(
        f"跳过页面数量："
        f"{report['skipped_page_count']}"
    )

    print(
        f"Chunk 总数量："
        f"{report['total_chunk_count']}"
    )

    print(
        f"最短 Chunk："
        f"{report['minimum_chunk_length']} 字符"
    )

    print(
        f"最长 Chunk："
        f"{report['maximum_chunk_length']} 字符"
    )

    print(
        f"平均 Chunk："
        f"{report['average_chunk_length']} 字符"
    )

    print(
        f"超出预期长度的 Chunk："
        f"{report['oversized_chunk_count']}"
    )

    if report["skipped_page_numbers"]:
        print(
            f"跳过页码："
            f"{report['skipped_page_numbers']}"
        )

    if report["ocr_review_page_numbers"]:
        print(
            f"需要 OCR 检查的页码："
            f"{report['ocr_review_page_numbers']}"
        )


def preview_chunks(
    chunks,
    preview_count=3,
    preview_length=300
):
    """
    职责：预览前几个 Chunk。
    """
    print(
        "\n===== 前几个 Chunk 预览 ====="
    )

    for chunk in chunks[:preview_count]:
        print("\n" + "=" * 70)

        print(
            f"Chunk {chunk['chunk_id']}"
        )

        print(
            f"来源："
            f"{chunk['source_reference']}"
        )

        print(
            f"页内编号："
            f"{chunk['chunk_index_on_page']}"
            f"/{chunk['chunk_count_on_page']}"
        )

        print(
            f"字符数量："
            f"{chunk['char_count']}"
        )

        print(
            f"重叠字符："
            f"{chunk['overlap_char_count']}"
        )

        print("\n内容预览：")

        content = chunk["content"]

        print(
            content[:preview_length]
        )

        if len(content) > preview_length:
            print("……")


def main():
    """
    Day 18 主流程：

    1. 读取 Day 17 分页 PDF 文本
    2. 对每页执行递归切块
    3. 合并过短尾部 Chunk
    4. 添加相邻 Chunk 重叠
    5. 保存文件名、页码等来源元数据
    6. 生成切块质量报告
    7. 保存并预览结果
    """
    project_root = get_project_root()

    input_file = (
        project_root
        / "data"
        / "day17_pdf_pages.json"
    )

    chunks_output_file = (
        project_root
        / "data"
        / "day18_pdf_chunks.json"
    )

    report_output_file = (
        project_root
        / "data"
        / "day18_chunking_report.json"
    )

    extraction_data = load_pdf_pages(
        input_file
    )

    chunks, skipped_pages = (
        build_pdf_chunks(
            extraction_data
        )
    )

    report = build_chunking_report(
        extraction_data=extraction_data,
        chunks=chunks,
        skipped_pages=skipped_pages
    )

    output_data = {
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
            "total_chunk_count": (
                report[
                    "total_chunk_count"
                ]
            ),

            "pages_with_chunks": (
                report[
                    "pages_with_chunks"
                ]
            ),

            "skipped_page_numbers": (
                skipped_pages
            )
        },

        "chunks": chunks
    }

    save_json(
        data=output_data,
        output_file=chunks_output_file
    )

    save_json(
        data=report,
        output_file=report_output_file
    )

    display_summary(report)

    preview_chunks(
        chunks=chunks,
        preview_count=3,
        preview_length=300
    )

    print("\nChunk 文件已保存到：")
    print(chunks_output_file)

    print("\n质量报告已保存到：")
    print(report_output_file)


if __name__ == "__main__":
    main()