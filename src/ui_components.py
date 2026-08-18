"""
Streamlit 答案、状态和证据展示组件。
"""

import streamlit as st

from file_utils import (
    format_float
)


def render_evidence(evidence):
    """
    展开显示证据原文及排序变化。
    """
    if not evidence:
        st.info(
            "本次没有向大语言模型提供证据。"
        )
        return

    for result in evidence:
        citation_id = result.get(
            "citation_id",
            "S?"
        )

        page_number = result.get(
            "page_number",
            "未知"
        )

        chunk_id = result.get(
            "chunk_id",
            "未知"
        )

        title = (
            f"[{citation_id}] "
            f"第 {page_number} 页 · "
            f"Chunk {chunk_id}"
        )

        with st.expander(
            title,
            expanded=False
        ):
            metric_columns = st.columns(5)

            metric_columns[0].metric(
                "FAISS 分数",
                format_float(
                    result.get(
                        "similarity"
                    )
                )
            )

            metric_columns[1].metric(
                "Reranker 分数",
                format_float(
                    result.get(
                        "reranker_score"
                    )
                )
            )

            metric_columns[2].metric(
                "FAISS 排名",
                result.get(
                    "faiss_rank",
                    "-"
                )
            )

            metric_columns[3].metric(
                "重排后排名",
                result.get(
                    "reranker_rank",
                    "-"
                )
            )

            rank_change = result.get(
                "rank_change"
            )

            rank_change_text = (
                f"{int(rank_change):+d}"
                if rank_change is not None
                else "-"
            )

            metric_columns[4].metric(
                "排名变化",
                rank_change_text
            )

            st.caption(
                f"来源文件："
                f"{result.get('source_file', '未知')} · "
                f"第 {page_number} 页"
            )

            st.markdown(
                "**证据原文**"
            )

            st.write(
                result.get(
                    "content",
                    ""
                )
            )


def render_assistant_result(
    query_result
):
    """
    显示答案、拒答状态、引用检查和证据。
    """
    generation_data = (
        query_result[
            "generation_data"
        ]
    )

    evidence_status = (
        generation_data[
            "evidence_status"
        ]
    )

    citation_validation = (
        generation_data[
            "citation_validation"
        ]
    )

    st.markdown(
        generation_data["answer"]
    )

    st.divider()

    status_columns = st.columns(4)

    status_columns[0].metric(
        "FAISS 候选",
        query_result[
            "candidate_count"
        ]
    )

    status_columns[1].metric(
        "最终证据",
        query_result[
            "final_evidence_count"
        ]
    )

    top_similarity = (
        evidence_status.get(
            "top_similarity"
        )
    )

    status_columns[2].metric(
        "最高相似度",
        format_float(
            top_similarity
        )
    )

    status_columns[3].metric(
        "是否调用 LLM",
        (
            "是"
            if generation_data[
                "llm_called"
            ]
            else "否"
        )
    )

    if evidence_status[
        "should_refuse"
    ]:
        st.info(
            "本次由本地拒答机制直接处理："
            f"{evidence_status['reason']}"
        )

    elif citation_validation[
        "citation_valid"
    ]:
        st.success(
            "引用格式检查通过。"
        )

    else:
        st.warning(
            "答案已生成，但引用格式检查未通过。"
        )

        invalid_ids = (
            citation_validation.get(
                "invalid_citation_ids",
                []
            )
        )

        if invalid_ids:
            st.write(
                f"无效引用：{invalid_ids}"
            )

    if generation_data.get(
        "citation_format_repaired",
        False
    ):
        st.caption(
            "程序已修复引用编号内部的异常空白。"
        )

    st.markdown(
        "### 引用证据"
    )

    render_evidence(
        generation_data.get(
            "evidence",
            []
        )
    )
