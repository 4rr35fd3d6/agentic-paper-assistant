import json
import re
from pathlib import Path

import pymupdf


PDF_FILENAME = "sample_paper.pdf"

# 一页提取出的有效字符少于这个数时，
# 不直接断定是扫描页，只标记为“需要检查”
MIN_PAGE_TEXT_LENGTH = 20


def get_project_root():
    """
    职责：获取项目根目录。

    输入：
        无

    输出：
        Path 类型的项目根目录
    """
    return Path(__file__).resolve().parents[1]


def clean_page_text(raw_text):
    """
    职责：
        对单页 PDF 文本做基础清洗。

    输入：
        raw_text：PyMuPDF 提取出的原始文本

    输出：
        cleaned_text：清洗后的文本

    注意：
        暂时保留正常换行，避免破坏论文段落结构。
    """
    # 统一 Windows、旧 Mac 和 Linux 换行符
    text = raw_text.replace(
        "\r\n",
        "\n"
    ).replace(
        "\r",
        "\n"
    )

    # 将不换行空格转换为普通空格
    text = text.replace(
        "\u00a0",
        " "
    )

    # 删除每一行行首和行尾多余空白
    cleaned_lines = []

    for line in text.split("\n"):
        cleaned_line = line.strip()

        # 一行内部连续的空格或制表符，
        # 统一压缩为一个普通空格
        cleaned_line = re.sub(
            r"[ \t]+",
            " ",
            cleaned_line
        )

        cleaned_lines.append(
            cleaned_line
        )

    text = "\n".join(
        cleaned_lines
    )

    # 三个及以上连续换行压缩成两个，
    # 保留段落之间的空行
    text = re.sub(
        r"\n{3,}",
        "\n\n",
        text
    )

    return text.strip()


def extract_pdf(pdf_file):
    """
    职责：
        打开 PDF，按页提取文字和页面信息。

    输入：
        pdf_file：PDF 文件路径

    输出：
        extraction_data：包含文档信息和每页文本的字典
    """
    if not pdf_file.exists():
        raise FileNotFoundError(
            f"没有找到 PDF 文件：{pdf_file}"
        )

    if pdf_file.suffix.lower() != ".pdf":
        raise ValueError(
            f"输入文件不是 PDF：{pdf_file}"
        )

    pages = []

    with pymupdf.open(pdf_file) as document:#把打开的pdf对象保存到变量document
        # 加密且需要密码的 PDF 无法直接处理
        if document.needs_pass:
            raise PermissionError(
                "该 PDF 需要密码，当前程序无法直接读取。"
            )

        # PDF 自带元数据，例如标题、作者和创建日期
        document_metadata = dict(
            document.metadata or {}
        )#如果 document.metadata 有有效内容，就使用它；如果它是 None 或空值，就使用空字典 {}。

        page_count = document.page_count#页面数

        for page_index in range(
            page_count
        ):
            # page_index 从 0 开始
            page = document.load_page(
                page_index
            )#页面对象

            # text：提取纯文本
            # sort=True：尽量按照页面坐标恢复阅读顺序
            raw_text = page.get_text(
                "text",
                sort=True#会尽量按照页面坐标重新排序，让输出更接近自然阅读顺序
            )

            cleaned_text = clean_page_text(
                raw_text
            )

            char_count = len(
                cleaned_text
            )

            # 字符太少只能说明“需要检查”，
            # 不能直接断定它一定是扫描页面
            needs_ocr_review = (
                char_count
                < MIN_PAGE_TEXT_LENGTH
            )

            page_data = {
                # 给用户看的页码从 1 开始
                "page_number": (
                    page_index + 1
                ),

                # 程序内部页索引从 0 开始
                "page_index": page_index,

                "char_count": char_count,

                "line_count": (
                    len(cleaned_text.splitlines())
                    if cleaned_text
                    else 0
                ),

                "needs_ocr_review": (
                    needs_ocr_review
                ),

                "page_width": round(
                    float(page.rect.width),
                    2
                ),

                "page_height": round(
                    float(page.rect.height),
                    2
                ),

                "text": cleaned_text
            }

            pages.append(
                page_data
            )

    total_char_count = sum(
        page["char_count"]
        for page in pages
    )

    pages_with_text = sum(
        1
        for page in pages
        if page["char_count"] > 0
    )

    pages_needing_review = sum(
        1
        for page in pages
        if page["needs_ocr_review"]
    )

    extraction_data = {
        "source_file": pdf_file.name,
        "source_path": str(
            pdf_file.resolve()
        ),
        "page_count": len(pages),
        "pages_with_text": (
            pages_with_text
        ),
        "pages_needing_ocr_review": (
            pages_needing_review
        ),
        "total_char_count": (
            total_char_count
        ),
        "metadata": (
            document_metadata
        ),
        "pages": pages
    }

    return extraction_data


def save_json(data, output_file):
    """
    职责：将分页提取结果保存为 JSON。
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


def save_plain_text(
    extraction_data,
    output_file
):
    """
    职责：
        将全部页面保存成便于人工查看的 TXT。

    每页之间加入明显的页码分隔符。
    """
    text_parts = []

    for page in extraction_data["pages"]:
        page_number = page["page_number"]
        page_text = page["text"]

        page_header = (
            "\n"
            + "=" * 70
            + f"\n第 {page_number} 页\n"
            + "=" * 70
            + "\n"
        )

        text_parts.append(
            page_header
        )

        if page_text:
            text_parts.append(
                page_text
            )
        else:
            text_parts.append(
                "[本页未提取到文本]"
            )

        text_parts.append("\n")

    full_text = "".join(
        text_parts
    ).strip()

    output_file.parent.mkdir(
        parents=True,
        exist_ok=True
    )

    with output_file.open(
        "w",
        encoding="utf-8"
    ) as file:
        file.write(full_text)


def display_summary(
    extraction_data
):
    """
    职责：在终端显示 PDF 提取摘要。
    """
    print(
        "\n===== Day 17 PDF 文本提取 ====="
    )

    print(
        f"文件名称："
        f"{extraction_data['source_file']}"
    )

    print(
        f"PDF 总页数："
        f"{extraction_data['page_count']}"
    )

    print(
        f"成功提取到文字的页面："
        f"{extraction_data['pages_with_text']}"
    )

    print(
        f"需要检查或 OCR 的页面："
        f"{extraction_data['pages_needing_ocr_review']}"
    )

    print(
        f"总字符数量："
        f"{extraction_data['total_char_count']}"
    )

    metadata = extraction_data[
        "metadata"
    ]

    print("\nPDF 元数据：")

    print(
        f"标题："
        f"{metadata.get('title') or '未提供'}"
    )

    print(
        f"作者："
        f"{metadata.get('author') or '未提供'}"
    )

    print(
        f"主题："
        f"{metadata.get('subject') or '未提供'}"
    )

    print("\n各页提取情况：")

    for page in extraction_data["pages"]:
        review_mark = (
            "，需要检查"
            if page["needs_ocr_review"]
            else ""
        )

        print(
            f"第 {page['page_number']} 页："
            f"{page['char_count']} 个字符"
            f"{review_mark}"
        )


def preview_pages(
    extraction_data,
    preview_length=300
):
    """
    职责：
        预览前两页的部分文字，
        快速检查提取顺序和乱码问题。
    """
    print("\n===== 前两页文本预览 =====")

    preview_pages_data = (
        extraction_data["pages"][:2]
    )

    for page in preview_pages_data:
        print("\n" + "-" * 60)

        print(
            f"第 {page['page_number']} 页"
        )

        page_text = page["text"]

        if not page_text:
            print("[本页未提取到文本]")
            continue

        preview_text = page_text[
            :preview_length
        ]

        print(preview_text)

        if len(page_text) > preview_length:
            print("……")


def main():
    """
    Day 17 主流程：

    1. 确定 PDF 路径
    2. 打开并逐页提取文字
    3. 保留页码、页面大小和 PDF 元数据
    4. 标记疑似需要 OCR 的页面
    5. 保存分页 JSON
    6. 保存人工可读 TXT
    7. 输出提取摘要与文本预览
    """
    project_root = get_project_root()

    pdf_file = (
        project_root
        / "data"
        / "papers"
        / PDF_FILENAME
    )

    json_output_file = (
        project_root
        / "data"
        / "day17_pdf_pages.json"
    )

    text_output_file = (
        project_root
        / "data"
        / "day17_pdf_text.txt"
    )

    extraction_data = extract_pdf(
        pdf_file
    )

    save_json(
        data=extraction_data,
        output_file=json_output_file
    )

    save_plain_text(
        extraction_data=extraction_data,
        output_file=text_output_file
    )

    display_summary(
        extraction_data
    )

    preview_pages(
        extraction_data=extraction_data,
        preview_length=300
    )

    print("\n分页 JSON 已保存到：")
    print(json_output_file)

    print("\n完整 TXT 已保存到：")
    print(text_output_file)


if __name__ == "__main__":
    main()

