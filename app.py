from __future__ import annotations

import os
import sys
import uuid
from pathlib import Path

import streamlit as st
from langgraph.checkpoint.memory import InMemorySaver

from logging_utils import create_file_logger


# ============================================================
# 1. 项目路径与模块导入
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parent
SRC_DIR = PROJECT_ROOT / "src"

if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from agent_graph import create_agent_graph
from agent_state import create_turn_input
from app_config import DAY23_SYSTEM_INSTRUCTIONS, PRACTICE_DIR
from file_utils import (
    calculate_bytes_sha256,
    get_pdf_page_count,
    prepare_pdf_paths,
    validate_pdf_bytes,
)
from rag_service import retrieve_ranked_evidence
from nli_service import (
    DEFAULT_NLI_MODEL_NAME,
    create_local_nli_model,
)
from ui_components import render_assistant_result

if str(PRACTICE_DIR) not in sys.path:
    sys.path.insert(0, str(PRACTICE_DIR))

import day22_reranker_rag as rag


LOGGER = create_file_logger(
    logger_name="paper_rag_app",
    log_file=PROJECT_ROOT / "logs" / "app.log",
)


# ============================================================
# 2. 页面与 RAG 配置
# ============================================================

st.set_page_config(
    page_title="论文 RAG 问答助手",
    page_icon="📚",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Day 22 后端从模块级变量读取系统提示词。
# 当前网页支持任意论文，因此使用通用学术提示词。
rag.SYSTEM_INSTRUCTIONS = DAY23_SYSTEM_INSTRUCTIONS
CLASSIC_MODE = "经典 RAG"
AGENTIC_MODE = "Agentic RAG"


# ============================================================
# 3. Session State
# ============================================================

def initialize_session_state() -> None:
    """初始化当前浏览器会话的知识库、聊天和 Agent 状态。"""
    defaults = {
        "runtime": None,
        "messages": [],
        "active_pdf_hash": None,
        "active_pdf_name": None,
        "answer_mode": CLASSIC_MODE,
        "agent_graph": None,
        "agent_checkpointer": None,
        "agent_thread_id": str(uuid.uuid4()),
        "agent_pdf_hash": None,
    }

    for key, value in defaults.items():
        if key not in st.session_state:
            st.session_state[key] = value


def reset_conversation(*, reset_agent_graph: bool) -> None:
    """清空网页消息，并为 Agentic RAG 创建新的会话线程。"""
    st.session_state.messages = []
    st.session_state.agent_thread_id = str(uuid.uuid4())

    if reset_agent_graph:
        st.session_state.agent_graph = None
        st.session_state.agent_checkpointer = None
        st.session_state.agent_pdf_hash = None


initialize_session_state()


# ============================================================
# 4. 重量级资源缓存
# ============================================================

@st.cache_resource(show_spinner=False)
def load_embedding_resource():
    """Embedding 只在首次建立知识库时加载一次。"""
    return rag.load_embedding_model(rag.EMBEDDING_MODEL_NAME)


@st.cache_resource(show_spinner=False)
def load_reranker_resource():
    """Reranker 只在首次提问时加载一次。"""
    tokenizer, model, device = rag.load_reranker()
    return {
        "tokenizer": tokenizer,
        "model": model,
        "device": device,
    }


@st.cache_resource(show_spinner=False)
def load_llm_client_resource():
    """经典 RAG 使用的火山方舟客户端只创建一次。"""
    return rag.create_openai_client()


@st.cache_resource(show_spinner=False)
def load_nli_resource():
    """Agentic RAG 使用的本地 NLI 模型在进程内只创建一次。"""
    return create_local_nli_model()


# ============================================================
# 5. 建立知识库
# ============================================================

def build_knowledge_base(
    original_filename,
    file_bytes,
    status=None,
):
    """PDF → 分页提取 → 递归切块 → Embedding → FAISS。"""
    paths, pdf_hash = prepare_pdf_paths(
        original_filename=original_filename,
        file_bytes=file_bytes,
    )

    if status is not None:
        status.write("正在加载 Embedding 模型……")

    embedding_model = load_embedding_resource()

    if status is not None:
        status.write("正在解析 PDF 并生成 Chunk……")

    chunks, report, chunk_status = rag.load_or_build_chunks(
        pdf_file=paths["pdf_file"],
        pages_file=paths["pages_file"],
        chunks_file=paths["chunks_file"],
        chunking_report_file=paths["chunking_report_file"],
        pipeline_metadata_file=paths["pipeline_metadata_file"],
        force_rebuild=False,
    )

    if not chunks:
        raise ValueError("该 PDF 没有生成任何有效 Chunk。")

    if status is not None:
        status.write("正在建立或加载 FAISS 索引……")

    index, index_status = rag.load_or_build_faiss_index(
        chunks=chunks,
        model=embedding_model,
        source_file=paths["chunks_file"],
        index_file=paths["faiss_index_file"],
        metadata_file=paths["faiss_metadata_file"],
    )

    if int(index.ntotal) != len(chunks):
        raise ValueError(
            "FAISS 向量数量与 Chunk 数量不一致："
            f"{index.ntotal} != {len(chunks)}"
        )

    return {
        "pdf_hash": pdf_hash,
        "original_filename": original_filename,
        "paths": paths,
        "chunks": chunks,
        "report": report,
        "chunk_status": chunk_status,
        "index": index,
        "index_status": index_status,
        "embedding_model": embedding_model,
    }


# ============================================================
# 6. 经典 RAG 查询
# ============================================================

def run_rag_query(query, runtime, status=None):
    """执行现有经典 RAG 查询流程，保持原有行为不变。"""
    if status is not None:
        status.write("正在加载语义 Reranker……")

    reranker = load_reranker_resource()

    if status is not None:
        status.write("正在执行 FAISS 召回、Reranker 重排和证据筛选……")

    retrieval_data = retrieve_ranked_evidence(
        query=query,
        runtime=runtime,
        reranker=reranker,
        rag_backend=rag,
    )

    if status is not None:
        status.write("正在执行拒答判断并生成带引用答案……")

    llm_client = load_llm_client_resource()

    generation_data = rag.generate_reliable_answer(
        client=llm_client,
        query=query,
        results=retrieval_data["selected_results"],
        # 拒答仍使用原始 FAISS 相似度，保持原项目阈值逻辑不变。
        refusal_results=retrieval_data["candidate_results"],
    )

    for evidence in generation_data.get("evidence", []):
        evidence["source_file"] = runtime["original_filename"]

    if status is not None:
        status.write("正在追加保存问答历史……")

    pdf_history_file = runtime["paths"]["chunks_file"].parent / "qa_history.jsonl"

    rag.append_qa_history(
        query=query,
        generation_data=generation_data,
        index=runtime["index"],
        output_file=pdf_history_file,
    )

    return {
        "generation_data": generation_data,
        "candidate_count": retrieval_data["candidate_count"],
        "reranked_count": retrieval_data["reranked_count"],
        "final_evidence_count": len(generation_data.get("evidence", [])),
        "selected_before_refusal_count": retrieval_data["selected_count"],
        "pdf_name": runtime["original_filename"],
        "history_file": str(pdf_history_file),
    }


# ============================================================
# 7. Agentic RAG 查询与展示
# ============================================================

def get_or_create_agent_graph(runtime, status=None):
    """为当前 PDF 创建一次 LangGraph，并在本会话中复用。"""
    graph_is_ready = (
        st.session_state.agent_graph is not None
        and st.session_state.agent_pdf_hash == runtime["pdf_hash"]
    )

    if graph_is_ready:
        return st.session_state.agent_graph

    if status is not None:
        status.write("正在加载 Agent 使用的语义 Reranker……")

    reranker = load_reranker_resource()

    if status is not None:
        status.write("正在创建 LangGraph Agent 工作流……")

    if status is not None:
        status.write("正在准备本地 NLI 逐条校验器……")

    nli_model = load_nli_resource()
    checkpointer = InMemorySaver()
    graph = create_agent_graph(
        runtime=runtime,
        reranker=reranker,
        rag_backend=rag,
        nli_model=nli_model,
        checkpointer=checkpointer,
    )

    st.session_state.agent_checkpointer = checkpointer
    st.session_state.agent_graph = graph
    st.session_state.agent_pdf_hash = runtime["pdf_hash"]

    return graph


def run_agentic_rag_query(query, runtime, status=None):
    """运行正式 LangGraph，并整理为适合 Streamlit 展示的数据。"""
    graph = get_or_create_agent_graph(runtime, status=status)

    if status is not None:
        status.write("Agent 正在分类问题并选择执行路线……")

    graph_config = {
        "configurable": {
            "thread_id": st.session_state.agent_thread_id,
        },
        "recursion_limit": 20,
    }

    final_state = graph.invoke(
        create_turn_input(query),
        config=graph_config,
    )

    evidence = []
    for item in final_state.get("evidence", []):
        normalized_item = dict(item)
        normalized_item["source_file"] = runtime["original_filename"]
        evidence.append(normalized_item)

    citation_reason = final_state.get("citation_validation_reason")
    if final_state.get("citation_valid"):
        citation_status = "通过"
    elif citation_reason:
        citation_status = "未通过"
    else:
        citation_status = "未执行"

    return {
        "answer": final_state.get("answer", "没有生成回答。"),
        "query_type": final_state.get("query_type", "unknown"),
        "raw_query": final_state.get("raw_query", query),
        "resolved_query": final_state.get("resolved_query", query),
        "target_page": final_state.get("target_page"),
        "sub_queries": final_state.get("sub_queries", []),
        "rewritten_query": final_state.get("rewritten_query"),
        "retry_count": final_state.get("retry_count", 0),
        "tool_call_count": final_state.get("tool_call_count", 0),
        "llm_call_count": final_state.get("llm_call_count", 0),
        "execution_trace": final_state.get("execution_trace", []),
        "evidence": evidence,
        "evidence_sufficient": final_state.get("evidence_sufficient", False),
        "evidence_check_reason": final_state.get("evidence_check_reason"),
        "citation_valid": final_state.get("citation_valid", False),
        "citation_status": citation_status,
        "citation_validation_reason": citation_reason,
        "generated_claim_count": len(final_state.get("generated_claims", [])),
        "accepted_claims": [
            {
                "section": item.get("section"),
                "text": item.get("text"),
                "citation_ids": item.get("citation_ids", []),
                "nli": item.get("nli", {}),
            }
            for item in final_state.get("accepted_claims", [])
        ],
        "rejected_claims": [
            {
                "text": item.get("text"),
                "rejection_reason": item.get("rejection_reason"),
            }
            for item in final_state.get("rejected_claims", [])
        ],
        "claim_validation_reason": final_state.get("claim_validation_reason"),
        "answer_retry_count": final_state.get("answer_retry_count", 0),
        "citation_map": final_state.get("citation_map", {}),
        "refusal_reason": final_state.get("refusal_reason"),
        "current_topic": final_state.get("current_topic"),
        "thread_id": st.session_state.agent_thread_id,
        "pdf_name": runtime["original_filename"],
    }


def get_evidence_content(item: dict) -> str:
    """返回第一个清理后非空的证据正文字段。"""
    for field_name in (
        "text",
        "content",
        "chunk_text",
        "page_content",
    ):
        value = item.get(field_name)
        if value is None:
            continue
        text = str(value).strip()
        if text:
            return text
    return ""


def render_agent_result(query_result: dict) -> None:
    """展示 Agent 最终回答、证据和可观察执行轨迹。"""
    st.caption("回答模式：Agentic RAG")
    st.markdown(query_result.get("answer", "没有生成回答。"))

    trace = query_result.get("execution_trace", [])
    evidence = query_result.get("evidence", [])

    summary_columns = st.columns(5)
    summary_columns[0].metric(
        "问题类型",
        query_result.get("query_type", "unknown"),
    )
    summary_columns[1].metric(
        "工具调用",
        query_result.get("tool_call_count", 0),
    )
    summary_columns[2].metric(
        "证据数量",
        len(evidence),
    )
    summary_columns[3].metric(
        "查询改写",
        query_result.get("retry_count", 0),
    )
    summary_columns[4].metric(
        "Ark 调用",
        query_result.get("llm_call_count", 0),
    )

    with st.expander("Agent 执行轨迹", expanded=False):
        if trace:
            for index, node_name in enumerate(trace, start=1):
                st.write(f"{index}. `{node_name}`")
        else:
            st.write("没有记录执行轨迹。")

        st.divider()
        st.write(
            "**解析后问题：**",
            query_result.get("resolved_query") or "—",
        )

        target_page = query_result.get("target_page")
        if target_page is not None:
            st.write("**指定页码：**", target_page)

        sub_queries = query_result.get("sub_queries", [])
        if sub_queries:
            st.write("**比较子问题：**")
            for index, sub_query in enumerate(sub_queries, start=1):
                st.write(f"{index}. {sub_query}")

        rewritten_query = query_result.get("rewritten_query")
        if rewritten_query:
            st.write("**改写后的查询：**", rewritten_query)

        evidence_reason = query_result.get("evidence_check_reason")
        if evidence_reason:
            st.write("**证据判断：**", evidence_reason)

        st.write(
            "**引用检查：**",
            query_result.get("citation_status", "未执行"),
        )

        citation_reason = query_result.get("citation_validation_reason")
        if citation_reason:
            st.write("**引用检查说明：**", citation_reason)

        claim_reason = query_result.get("claim_validation_reason")
        if claim_reason:
            st.write("**Claim 校验：**", claim_reason)

        accepted_claims = query_result.get("accepted_claims", [])
        rejected_claims = query_result.get("rejected_claims", [])
        st.write(
            "**结论筛选：**",
            f"保留 {len(accepted_claims)} 条，过滤 {len(rejected_claims)} 条，"
            f"答案重试 {query_result.get('answer_retry_count', 0)} 次。",
        )

        current_topic = query_result.get("current_topic")
        if current_topic:
            st.write("**当前会话主题：**", current_topic)

    with st.expander(
        f"参考证据（{len(evidence)} 条）",
        expanded=False,
    ):
        if not evidence:
            st.write("本次没有可展示的论文证据。")

        for item in evidence:
            citation_id = item.get("citation_id", "S?")
            page_number = item.get("page_number", "?")
            content = get_evidence_content(item)

            chunk_id = item.get("chunk_id", "?")
            st.markdown(
                f"**[{citation_id}] 第 {page_number} 页 · Chunk {chunk_id}**"
            )
            used_quotes = item.get("used_quotes", [])
            if used_quotes:
                st.markdown("**本回答实际使用的原文锚点：**")
                for quote in used_quotes:
                    st.write(f"• {quote}")
            st.markdown("**完整 Chunk：**")
            st.write(content or "该证据没有可显示的正文。")
            st.divider()


# ============================================================
# 8. 侧边栏
# ============================================================

with st.sidebar:
    st.header("📄 论文知识库")

    st.radio(
        "回答模式",
        options=(CLASSIC_MODE, AGENTIC_MODE),
        key="answer_mode",
        help=(
            "经典 RAG 使用单次检索与生成；"
            "Agentic RAG 使用 LangGraph 完成分类、拆解、改写、拒答和引用检查。"
        ),
    )

    uploaded_file = st.file_uploader(
        "上传一篇文本型 PDF",
        type=["pdf"],
        accept_multiple_files=False,
        help="扫描型 PDF 可能无法直接提取文字。",
    )

    uploaded_bytes = None
    uploaded_hash = None

    if uploaded_file is not None:
        uploaded_bytes = uploaded_file.getvalue()
        if uploaded_bytes:
            uploaded_hash = calculate_bytes_sha256(uploaded_bytes)

    build_button = st.button(
        "建立 / 加载知识库",
        type="primary",
        use_container_width=True,
        disabled=uploaded_file is None,
    )

    clear_button = st.button(
        "新建会话 / 清空聊天",
        use_container_width=True,
    )

    if clear_button:
        # 保留当前 PDF 对应的图对象，只更换 thread_id，
        # 从而清除多轮记忆且避免重复初始化模型。
        reset_conversation(reset_agent_graph=False)

    st.divider()
    st.subheader("运行状态")

    api_key_ready = bool(os.getenv("ARK_API_KEY", "").strip())
    model_id_ready = bool(os.getenv("ARK_MODEL_ID", "").strip())
    llm_config_ready = api_key_ready and model_id_ready

    st.write(
        "Ark API Key："
        + ("✅ 已检测到" if api_key_ready else "⚠️ 未检测到")
    )
    st.write(
        "Ark Model ID："
        + ("✅ 已检测到" if model_id_ready else "⚠️ 未检测到")
    )

    st.subheader("检索配置")
    st.write(f"Embedding：`{rag.EMBEDDING_MODEL_NAME}`")
    st.write(f"Reranker：`{rag.RERANKER_MODEL_NAME}`")
    st.write(
        "本地 NLI："
        f"`{os.getenv('NLI_MODEL_NAME', DEFAULT_NLI_MODEL_NAME)}`"
    )
    st.write(f"FAISS 候选数：`{rag.RETRIEVAL_CANDIDATE_K}`")
    st.write(f"最终证据数：`{rag.FINAL_EVIDENCE_COUNT}`")
    st.write(f"Chunk 阈值：`{rag.CHUNK_MIN_SIMILARITY}`")
    st.write(f"问题拒答阈值：`{rag.QUERY_REFUSAL_THRESHOLD}`")


# ============================================================
# 9. 建立知识库
# ============================================================

if build_button and uploaded_file is not None:
    try:
        validate_pdf_bytes(uploaded_bytes)

        with st.status(
            "正在建立论文知识库……",
            expanded=True,
        ) as build_status:
            runtime = build_knowledge_base(
                original_filename=uploaded_file.name,
                file_bytes=uploaded_bytes,
                status=build_status,
            )

            st.session_state.runtime = runtime
            st.session_state.active_pdf_hash = runtime["pdf_hash"]
            st.session_state.active_pdf_name = uploaded_file.name

            # Graph 内部闭包绑定当前 runtime。
            # 每次重新建立知识库都必须清除旧图和旧会话记忆。
            reset_conversation(reset_agent_graph=True)

            build_status.update(
                label="知识库建立完成",
                state="complete",
                expanded=False,
            )

            LOGGER.info(
                "Knowledge base ready: pdf=%s chunks=%s vectors=%s",
                uploaded_file.name,
                len(runtime["chunks"]),
                int(runtime["index"].ntotal),
            )

        st.success(f"已加载：{uploaded_file.name}")

    except Exception as error:
        LOGGER.exception(
            "Knowledge base build failed: pdf=%s",
            uploaded_file.name if uploaded_file is not None else "unknown",
        )
        st.error(f"知识库建立失败：{error}")


# ============================================================
# 10. 主页面
# ============================================================

st.title("📚 论文 RAG 问答助手")
st.caption(
    "BGE Embedding · FAISS · BGE Reranker · "
    "经典 RAG / LangGraph Agentic RAG · 可追溯引用"
)
st.info(f"当前回答模式：**{st.session_state.answer_mode}**")

runtime = st.session_state.runtime

uploaded_file_changed = (
    runtime is not None
    and uploaded_hash is not None
    and uploaded_hash != st.session_state.active_pdf_hash
)

if runtime is None:
    st.info("请先在左侧上传论文，然后点击“建立 / 加载知识库”。")
else:
    if uploaded_file_changed:
        st.warning(
            "左侧选择的 PDF 与当前知识库不一致。"
            "请点击“建立 / 加载知识库”后再提问，"
            "避免使用旧索引回答新论文。"
        )

    report = runtime["report"]
    chunks = runtime["chunks"]

    st.success(f"当前知识库：{st.session_state.active_pdf_name}")

    summary_columns = st.columns(4)
    summary_columns[0].metric("PDF 页数", get_pdf_page_count(report, chunks))
    summary_columns[1].metric("Chunk 数量", len(chunks))
    summary_columns[2].metric("FAISS 向量", int(runtime["index"].ntotal))
    summary_columns[3].metric("向量维度", int(runtime["index"].d))


# 显示当前会话聊天记录。
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        if message["role"] == "user":
            st.markdown(message["content"])
            continue

        message_mode = message.get("answer_mode", CLASSIC_MODE)
        if message_mode == AGENTIC_MODE:
            render_agent_result(message["query_result"])
        else:
            st.caption("回答模式：经典 RAG")
            render_assistant_result(message["query_result"])


# ============================================================
# 11. 用户提问
# ============================================================

query_disabled = (
    runtime is None
    or uploaded_file_changed
    or not llm_config_ready
)

if runtime is not None and not llm_config_ready:
    st.warning(
        "当前 Ark 配置不完整。"
        "请在启动 Streamlit 的同一个 PowerShell 中设置 "
        "ARK_API_KEY 和 ARK_MODEL_ID，然后重新启动网页。"
    )

query = st.chat_input(
    "请输入与当前论文有关的问题",
    disabled=query_disabled,
)

if query:
    cleaned_query = query.strip()

    if cleaned_query:
        st.session_state.messages.append(
            {
                "role": "user",
                "content": cleaned_query,
            }
        )

        with st.chat_message("user"):
            st.markdown(cleaned_query)

        with st.chat_message("assistant"):
            try:
                answer_mode = st.session_state.answer_mode
                status_label = (
                    "Agent 正在分析并执行工作流……"
                    if answer_mode == AGENTIC_MODE
                    else "正在检索和生成答案……"
                )

                with st.status(status_label, expanded=True) as query_status:
                    if answer_mode == AGENTIC_MODE:
                        query_result = run_agentic_rag_query(
                            query=cleaned_query,
                            runtime=runtime,
                            status=query_status,
                        )
                    else:
                        query_result = run_rag_query(
                            query=cleaned_query,
                            runtime=runtime,
                            status=query_status,
                        )

                    query_status.update(
                        label="回答生成完成",
                        state="complete",
                        expanded=False,
                    )

                if answer_mode == AGENTIC_MODE:
                    render_agent_result(query_result)
                    LOGGER.info(
                        "Agentic RAG query completed: pdf=%s type=%s evidence=%s trace=%s",
                        runtime.get("original_filename", "unknown"),
                        query_result.get("query_type"),
                        len(query_result.get("evidence", [])),
                        " -> ".join(query_result.get("execution_trace", [])),
                    )
                else:
                    st.caption("回答模式：经典 RAG")
                    render_assistant_result(query_result)

                    generation_data = query_result.get("generation_data", {})
                    LOGGER.info(
                        "Classic RAG query completed: pdf=%s llm_called=%s evidence=%s",
                        runtime.get("original_filename", "unknown"),
                        generation_data.get("llm_called"),
                        query_result.get("final_evidence_count"),
                    )

                st.session_state.messages.append(
                    {
                        "role": "assistant",
                        "answer_mode": answer_mode,
                        "query_result": query_result,
                    }
                )

            except Exception as error:
                LOGGER.exception(
                    "RAG query failed: mode=%s pdf=%s query=%r",
                    st.session_state.answer_mode,
                    runtime.get("original_filename", "unknown"),
                    cleaned_query,
                )
                st.error(f"问答失败：{error}")

                # 查询失败时撤回刚加入的用户消息，
                # 避免页面重执行后留下没有助手回复的孤立问题。
                if (
                    st.session_state.messages
                    and st.session_state.messages[-1].get("role") == "user"
                    and st.session_state.messages[-1].get("content")
                    == cleaned_query
                ):
                    st.session_state.messages.pop()
