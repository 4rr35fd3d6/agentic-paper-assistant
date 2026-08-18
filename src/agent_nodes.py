"""Agentic RAG 的 LangGraph 节点实现。

职责拆分：
- 大模型：生成自然语言原子结论，并选择 chunk_id + 原文锚点；
- Python：验证 Chunk 身份、原文锚点、数值和对象覆盖；
- 本地 NLI：逐条判断自然语言结论是否被证据上下文蕴含；
- Python：最后统一分配 [S1]、[S2] 引用编号并渲染答案。
"""

from __future__ import annotations

import re
from typing import Any

from langchain_core.messages import HumanMessage, SystemMessage

from agent_prompts import (
    CLAIM_GENERATION_SYSTEM_PROMPT,
    EVIDENCE_JUDGE_SYSTEM_PROMPT,
    QUERY_REWRITE_SYSTEM_PROMPT,
)
from agent_service import create_ark_chat_model
from agent_state import AgentState
from agent_tools import create_retrieve_paper_evidence_tool
from claim_grounding import (
    assign_citations_and_render,
    build_evidence_index,
    clean_reasoning_evidence,
    get_evidence_text,
    message_content_to_text,
    normalize_chunk_id,
    parse_claim_output,
    parse_evidence_decision,
    validate_claims as validate_generated_claims,
)
from nli_service import create_local_nli_model
from retrieval_evidence import (
    count_retained_new_evidence,
    deduplicate_evidence,
    merge_retry_evidence,
)


def append_trace(state: AgentState, node_name: str) -> list[str]:
    return state.get("execution_trace", []) + [node_name]

def resolve_query_pronouns(
    raw_query: str,
    current_topic: str | None,
) -> str:
    """只处理明确代词，不破坏“其它”和“它们”。
    """
    resolved = str(
        raw_query or ""
    )

    if not current_topic:
        return resolved

    for pronoun in (
        "该方法",
        "该算法",
        "这个方法",
        "这个算法",
    ):
        resolved = resolved.replace(
            pronoun,
            current_topic,
        )

    # 只替换明确作为单数代词的“它”。
    # (?<!其) 防止修改“其它”；
    # (?!们) 防止修改“它们”。
    resolved = re.sub(
        r"(?<!其)它(?!们)"
        r"(?=(?:的|是|有|能|会|和|与|跟|"
        r"相比|如何|怎么|在))",
        current_topic,
        resolved,
    )

    return resolved


def prepare_turn(state: AgentState) -> dict[str, Any]:
    """开始新一轮，并保留 Checkpointer 中的跨轮主题。"""
    raw_query = str(
        state.get("raw_query") or state.get("query") or ""
    ).replace("\ue000", "").strip()

    current_topic = state.get("current_topic")
    resolved_query = resolve_query_pronouns(
        raw_query,
        current_topic,
    )

    if resolved_query != raw_query:
        print(f"[prepare_turn] 指代消解：{raw_query} → {resolved_query}")
    else:
        print(f"[prepare_turn] 本轮问题：{resolved_query}")

    return {
        "query": resolved_query,
        "raw_query": raw_query,
        "resolved_query": resolved_query,
        "query_type": "unknown",
        "target_page": None,
        "rewritten_query": None,
        "sub_queries": [],
        "comparison_targets": [],
        "evidence": [],
        "evidence_sufficient": False,
        "evidence_check_reason": None,
        "seen_chunk_ids": [],
        "new_evidence_count": 0,
        "tool_call_count": 0,
        "retrieval_round_count": 0,
        "llm_call_count": 0,
        "retry_count": 0,
        "generated_claims": [],
        "accepted_claims": [],
        "rejected_claims": [],
        "claim_validation_reason": None,
        "answer_retry_count": 0,
        "citation_map": {},
        "answer": "",
        "citation_valid": False,
        "citation_validation_reason": None,
        "refusal_reason": None,
        "execution_trace": ["prepare_turn"],
    }


def extract_current_topic(query: str) -> str | None:
    candidates = re.findall(
        r"(?<![A-Za-z0-9-])([A-Za-z][A-Za-z0-9-]{1,})(?![A-Za-z0-9-])",
        query,
    )
    ignored = {"FL", "PFL", "IID", "NON-IID"}
    for candidate in candidates:
        if candidate.upper() not in ignored:
            return candidate
    return None

def resolve_memory_topic(
    *,
    resolved_query: str,
    previous_topic: str | None,
    query_type: str | None,
    citation_valid: bool,
    accepted_claims: object,
) -> tuple[str | None, str | None]:
    """仅在本轮产生可靠论文回答时更新会话主题。

    返回：
    1. 最终保存的 current_topic；
    2. 本轮实际检测到的新主题。
    """

    can_update_topic = (
        query_type != "out_of_scope"
        and citation_valid
        and bool(accepted_claims)
    )

    if not can_update_topic:
        return previous_topic, None

    detected_topic = extract_current_topic(
        resolved_query
    )

    current_topic = (
        detected_topic
        or previous_topic
    )

    return current_topic, detected_topic

def classify_query(state: AgentState) -> dict[str, Any]:
    """使用低成本确定性规则识别真正影响工作流的查询类型。"""
    query = state["query"].replace("\ue000", "").strip().lower()
    page_match = re.search(r"第\s*(\d+)\s*页", query)

    out_of_scope_keywords = (
        "天气",
        "股票",
        "汇率",
        "彩票",
        "菜谱",
        "食谱",
        "旅游攻略",
        "电影",
        "明星",
        "购物",
    )
    comparison_keywords = (
        "比较",
        "区别",
        "不同",
        "相比",
        "相较",
        "差异",
        "优于",
        "不如",
    )
    workflow_keywords = (
        "流程",
        "步骤",
        "过程",
        "怎么运行",
        "怎么跑",
        "如何运行",
        "如何执行",
        "如何训练",
        "训练流程",
        "训练过程",
        "聚类流程",
        "聚合流程",
        "客户端训练",
        "服务端聚合",
        "核心机制",
        "工作机制",
        "具体机制",
        "实现过程",
    )

    target_page = None
    if page_match:
        query_type = "page_read"
        target_page = int(page_match.group(1))
    elif any(keyword in query for keyword in out_of_scope_keywords):
        query_type = "out_of_scope"
    elif any(keyword in query for keyword in comparison_keywords):
        query_type = "comparison"
    elif any(keyword in query for keyword in workflow_keywords):
        query_type = "method_workflow"
    else:
        query_type = "fact"

    print(f"[classify_query] 问题类型：{query_type}")
    if target_page is not None:
        print(f"[classify_query] 指定页码：{target_page}")

    return {
        "query_type": query_type,
        "target_page": target_page,
        "execution_trace": append_trace(state, "classify_query"),
    }


def create_agent_nodes(
    runtime,
    reranker,
    rag_backend,
    *,
    nli_model=None,
) -> dict[str, object]:
    """创建正式工作流节点，共享同一 LLM、检索工具和本地 NLI。"""
    model = create_ark_chat_model()
    local_nli = nli_model or create_local_nli_model()
    retrieve_tool = create_retrieve_paper_evidence_tool(
        runtime=runtime,
        reranker=reranker,
        rag_backend=rag_backend,
    )

    def decompose_query(state: AgentState) -> dict[str, Any]:
        original_query = state["query"].replace("\ue000", "").strip()
        normalized_query = original_query.rstrip("？?。！!").strip()

        for prefix in ("请比较一下", "请比较", "比较一下", "比较"):
            if normalized_query.startswith(prefix):
                normalized_query = normalized_query[len(prefix):].strip()
                break

        for suffix in (
            "有什么区别",
            "有何区别",
            "有哪些区别",
            "有哪些差异",
            "的区别是什么",
            "差异是什么",
            "的区别",
            "的差异",
        ):
            if normalized_query.endswith(suffix):
                normalized_query = normalized_query[:-len(suffix)].strip()
                break

        if normalized_query.endswith("相比"):
            normalized_query = normalized_query[:-2].strip()

        match = re.match(
            r"^(.+?)\s*(?:相比于|相较于|和|与|跟|相比)\s*(.+?)$",
            normalized_query,
        )
        if match is None:
            reason = (
                "无法稳定识别两个比较对象，请使用“A 和 B 有什么区别”的表达。"
            )
            print(f"[decompose_query] 拆解失败：{reason}")
            return {
                "sub_queries": [],
                "refusal_reason": reason,
                "execution_trace": append_trace(state, "decompose_query"),
            }

        left_target = match.group(1).strip()
        right_target = match.group(2).strip()
        if not left_target or not right_target:
            reason = "比较对象不能为空。"
            return {
                "sub_queries": [],
                "refusal_reason": reason,
                "execution_trace": append_trace(state, "decompose_query"),
            }

        sub_queries = [
            (
                f"请从论文中查找 {left_target} 的定义、目标、核心机制、"
                "训练流程、优点和局限。"
            ),
            (
                f"请从论文中查找 {right_target} 的定义、目标、核心机制、"
                "训练流程、优点和局限。"
            ),
        ]
        print(f"[decompose_query] 比较对象：{left_target} vs {right_target}")
        for index, sub_query in enumerate(sub_queries, start=1):
            print(f"[decompose_query] 子问题{index}：{sub_query}")

        return {
            "sub_queries": sub_queries,
            "comparison_targets": [
                left_target,
                right_target,
            ],
            "refusal_reason": None,
            "execution_trace": append_trace(
                state,
                "decompose_query",
            ),
        }

    def multi_retrieve(state: AgentState) -> dict[str, Any]:
        sub_queries = state.get("sub_queries", [])
        print(f"[multi_retrieve] 准备执行 {len(sub_queries)} 个子查询")
        if len(sub_queries) != 2:
            reason = "比较问题没有成功拆分成两个子查询。"
            return {
                "evidence": [],
                "new_evidence_count": 0,
                "refusal_reason": reason,
                "execution_trace": append_trace(state, "multi_retrieve"),
            }

        collected: list[dict[str, Any]] = []
        failure_reasons: list[str] = []
        retrieval_calls = 0

        for side, sub_query in enumerate(sub_queries, start=1):
            print(f"[multi_retrieve] 执行子查询{side}：{sub_query}")
            result = retrieve_tool.invoke({"query": sub_query})
            retrieval_calls += 1
            status = result.get("status", "unknown")
            current = result.get("evidence", []) if status == "success" else []
            for item in current:
                collected.append(
                    {
                        **item,
                        "comparison_side": side,
                        "source_query": sub_query,
                    }
                )
            print(
                f"[multi_retrieve] 子查询{side}状态：{status}，"
                f"返回：{len(current)}"
            )
            if status != "success":
                failure_reasons.append(
                    result.get("reason")
                    or result.get("message")
                    or f"子查询{side}检索失败。"
                )

        merged = deduplicate_evidence(collected, limit=10)
        reason = None if merged else (
            "；".join(failure_reasons) or "两个子查询均未找到可用证据。"
        )
        print(f"[multi_retrieve] 合并后证据数量：{len(merged)}")
        return {
            "evidence": merged,
            "seen_chunk_ids": sorted(
                item["chunk_id"]
                for item in merged
                if normalize_chunk_id(item.get("chunk_id")) is not None
            ),
            "new_evidence_count": len(merged),
            "tool_call_count": state.get("tool_call_count", 0) + retrieval_calls,
            "retrieval_round_count": state.get("retrieval_round_count", 0) + 1,
            "refusal_reason": reason,
            "execution_trace": append_trace(state, "multi_retrieve"),
        }

    def retrieve(state: AgentState) -> dict[str, Any]:
        active_query = state.get("rewritten_query") or state["query"]
        print(f"[retrieve] 检索问题：{active_query}")
        result = retrieve_tool.invoke({"query": active_query})
        status = result.get("status", "unknown")
        retrieved = result.get("evidence", []) if status == "success" else []

        previous = state.get("evidence", [])
        previous_ids = {
            normalize_chunk_id(item.get("chunk_id"))
            for item in previous
            if normalize_chunk_id(item.get("chunk_id")) is not None
        }
        new_items = [
            item
            for item in retrieved
            if normalize_chunk_id(item.get("chunk_id")) not in previous_ids
        ]
        if state.get("retry_count", 0) > 0:
            merged = merge_retry_evidence(
                previous,
                new_items,
                limit=10,
                reserved_new=5,
            )
        else:
            merged = deduplicate_evidence(previous + new_items, limit=10)

        retained_new_count = count_retained_new_evidence(
            merged,
            new_items,
        )
        print(
            f"[retrieve] 状态：{status}，本轮返回：{len(retrieved)}，"
            f"新增证据：{len(new_items)}，实际保留新增：{retained_new_count}，"
            f"合并证据：{len(merged)}"
        )

        refusal_reason = None
        if status != "success":
            refusal_reason = (
                result.get("reason")
                or result.get("message")
                or "本轮没有检索到可用论文证据。"
            )
        elif state.get("retry_count", 0) > 0 and not new_items:
            refusal_reason = "第二轮检索没有发现新的论文证据。"

        return {
            "evidence": merged,
            "seen_chunk_ids": sorted(
                item["chunk_id"]
                for item in merged
                if normalize_chunk_id(item.get("chunk_id")) is not None
            ),
            "new_evidence_count": retained_new_count,
            "tool_call_count": state.get("tool_call_count", 0) + 1,
            "retrieval_round_count": state.get("retrieval_round_count", 0) + 1,
            "refusal_reason": refusal_reason,
            "execution_trace": append_trace(state, "retrieve"),
        }

    def read_page(state: AgentState) -> dict[str, Any]:
        target_page = state.get("target_page")
        print(f"[read_page] 准备读取第{target_page}页")
        if not isinstance(target_page, int) or target_page < 1:
            reason = "没有识别到有效的论文页码。"
            return {
                "evidence": [],
                "new_evidence_count": 0,
                "refusal_reason": reason,
                "execution_trace": append_trace(state, "read_page"),
            }

        page_chunks: list[dict[str, Any]] = []
        for chunk in runtime.get("chunks", []):
            try:
                page_number = int(chunk.get("page_number"))
            except (TypeError, ValueError):
                continue
            if page_number == target_page:
                page_chunks.append(dict(chunk))

        def safe_int(value: Any) -> int:
            try:
                return int(value)
            except (TypeError, ValueError):
                return 0

        page_chunks.sort(
            key=lambda item: (
                safe_int(item.get("chunk_index_on_page")),
                safe_int(item.get("chunk_id")),
            )
        )
        page_chunks = deduplicate_evidence(page_chunks, limit=50)

        if page_chunks:
            reason = None
            print(
                f"[read_page] 第{target_page}页共读取{len(page_chunks)}个 Chunk"
            )
        else:
            reason = (
                f"论文第{target_page}页没有可读取的文本内容，"
                "或者该页超出当前论文范围。"
            )
            print(f"[read_page] 读取失败：{reason}")

        return {
            "evidence": page_chunks,
            "seen_chunk_ids": [
                item["chunk_id"]
                for item in page_chunks
                if normalize_chunk_id(item.get("chunk_id")) is not None
            ],
            "new_evidence_count": len(page_chunks),
            "tool_call_count": state.get("tool_call_count", 0) + 1,
            "retrieval_round_count": state.get("retrieval_round_count", 0) + 1,
            "refusal_reason": reason,
            "execution_trace": append_trace(state, "read_page"),
        }

    def evidence_check(state: AgentState) -> dict[str, Any]:
        print("[evidence_check] 正在检查证据充分性")
        evidence = state.get("evidence", [])
        if not evidence:
            reason = state.get("refusal_reason") or "没有检索到任何可用论文证据。"
            print(f"[evidence_check] 证据不足：{reason}")
            return {
                "evidence_sufficient": False,
                "evidence_check_reason": reason,
                "refusal_reason": reason,
                "execution_trace": append_trace(state, "evidence_check"),
            }

        blocks: list[str] = []
        for item in evidence:
            chunk_id = normalize_chunk_id(item.get("chunk_id"))
            content = clean_reasoning_evidence(get_evidence_text(item))
            if chunk_id is None or not content:
                continue
            blocks.append(
                f"[Chunk {chunk_id}] 第{item.get('page_number', '?')}页\n{content}"
            )
        if not blocks:
            reason = (
                "本轮证据列表存在，但没有包含"
                "有效的 Chunk ID 和论文正文。"
            )

            print(
                "[evidence_check] "
                f"证据不足：{reason}"
            )

            return {
                "evidence_sufficient": False,
                "evidence_check_reason": reason,
                "refusal_reason": reason,
                "execution_trace": append_trace(
                    state,
                    "evidence_check",
                ),
            }

        evidence_payload = "\n\n".join(blocks)
        if not blocks:
            reason = (
                "本轮证据列表存在，但没有包含"
                "有效的 Chunk ID 和论文正文。"
            )

            print(
                "[evidence_check] "
                f"证据不足：{reason}"
            )

            return {
                "evidence_sufficient": False,
                "evidence_check_reason": reason,
                "refusal_reason": reason,
                "execution_trace": append_trace(
                    state,
                    "evidence_check",
                ),
            }

        response = model.invoke(
            [
                SystemMessage(content=EVIDENCE_JUDGE_SYSTEM_PROMPT),
                HumanMessage(
                    content=(
                        f"用户问题：\n{state['query']}\n\n"
                        f"论文证据：\n{evidence_payload}"
                    )
                ),
            ]
        )
        raw = message_content_to_text(response.content)
        sufficient, reason = parse_evidence_decision(raw)
        status = "充分" if sufficient else "不足"
        print(f"[evidence_check] 证据{status}：{reason}")

        return {
            "evidence_sufficient": sufficient,
            "evidence_check_reason": reason,
            "refusal_reason": None if sufficient else reason,
            "llm_call_count": state.get("llm_call_count", 0) + 1,
            "execution_trace": append_trace(state, "evidence_check"),
        }

    def rewrite_query(state: AgentState) -> dict[str, Any]:
        print("[rewrite_query] 正在改写检索问题")
        response = model.invoke(
            [
                SystemMessage(content=QUERY_REWRITE_SYSTEM_PROMPT),
                HumanMessage(
                    content=(
                        f"原始问题：\n{state['query']}\n\n"
                        f"证据不足原因：\n"
                        f"{state.get('evidence_check_reason') or '现有证据不足。'}"
                    )
                ),
            ]
        )
        rewritten = message_content_to_text(response.content).strip('"\' ')
        if not rewritten:
            rewritten = (
                f"{state['query']}；请重点检索算法名称、核心机制、"
                "训练步骤和直接实验结果。"
            )
        print(f"[rewrite_query] 改写结果：{rewritten}")
        return {
            "rewritten_query": rewritten,
            "retry_count": state.get("retry_count", 0) + 1,
            "llm_call_count": state.get("llm_call_count", 0) + 1,
            "refusal_reason": None,
            "execution_trace": append_trace(state, "rewrite_query"),
        }

    def generate_claims(state: AgentState) -> dict[str, Any]:
        """用一次 Ark 调用生成自然语言 Claim 与稳定原文锚点。"""
        attempt = state.get("answer_retry_count", 0) + 1
        print(f"[generate_claims] 生成自然语言结论，第{attempt}次尝试")

        evidence_index = build_evidence_index(state.get("evidence", []))
        blocks: list[str] = []
        print("\n===== 送给模型的证据 =====")
        for chunk_id, item in evidence_index.items():
            content = item["reasoning_text"]
            page = item.get("page_number", "?")
            blocks.append(
                f"<chunk id=\"{chunk_id}\" page=\"{page}\">\n"
                f"{content}\n</chunk>"
            )
            print(f"[Chunk {chunk_id}] 第{page}页：{content[:300]}")

        evidence_payload = "\n\n".join(blocks)

        retry_feedback = ""
        if state.get("answer_retry_count", 0) > 0:
            retry_feedback = (
                "\n\n上一次生成的结论全部未通过本地校验。"
                "请严格缩小结论范围，确保每个关键事实都被原文锚点直接支持。"
                f"\n失败摘要：{state.get('claim_validation_reason') or '未知'}"
            )

        response = model.invoke(
            [
                SystemMessage(content=CLAIM_GENERATION_SYSTEM_PROMPT),
                HumanMessage(
                    content=(
                        f"用户问题：\n{state['query']}\n\n"
                        f"问题类型：{state.get('query_type')}\n\n"
                        f"论文 Chunk：\n{evidence_payload}"
                        f"{retry_feedback}"
                    )
                ),
            ]
        )
        raw_output = message_content_to_text(response.content)
        preview = " ".join(raw_output.split())
        print(f"[generate_claims] 原始结果：{preview[:700]}")
        claims = parse_claim_output(raw_output)
        print(f"[generate_claims] 解析得到 Claim：{len(claims)}条")

        return {
            "generated_claims": claims,
            "accepted_claims": [],
            "rejected_claims": [],
            "claim_validation_reason": None,
            "refusal_reason": None,
            "llm_call_count": state.get("llm_call_count", 0) + 1,
            "execution_trace": append_trace(state, "generate_claims"),
        }

    def validate_claims(state: AgentState) -> dict[str, Any]:
        """纯本地逐条验证，不调用 Ark。"""
        print("[validate_claims] 正在执行 Chunk、原文锚点和本地 NLI 校验")
        claims = state.get("generated_claims", [])
        if not claims:
            reason = "模型没有返回可解析的 Claim JSON。"
            print(f"[validate_claims] 未通过：{reason}")
            return {
                "accepted_claims": [],
                "rejected_claims": [],
                "claim_validation_reason": reason,
                "refusal_reason": reason,
                "citation_valid": False,
                "execution_trace": append_trace(state, "validate_claims"),
            }

        try:
            accepted, rejected, reason = validate_generated_claims(
                query=state["query"],
                query_type=state.get("query_type", "unknown"),
                comparison_targets=state.get("comparison_targets",[], ),
                evidence=state.get("evidence", []),
                claims=claims,
                nli_model=local_nli,
            )
        except Exception as error:
            reason = (
                "本地 NLI 模型加载或推理失败，未继续展示未经语义校验的结论："
                f"{error}"
            )
            print(f"[validate_claims] 未通过：{reason}")
            return {
                "accepted_claims": [],
                "rejected_claims": [
                    {
                        **claim,
                        "rejection_reason": reason,
                    }
                    for claim in claims
                ],
                "claim_validation_reason": reason,
                "refusal_reason": reason,
                # NLI 故障与答案措辞无关，不浪费第二次 Ark 生成调用。
                "answer_retry_count": 1,
                "citation_valid": False,
                "execution_trace": append_trace(state, "validate_claims"),
            }

        print(f"[validate_claims] {reason}")
        for item in rejected:
            print(
                "[validate_claims] 删除 Claim："
                f"{item.get('text', '')[:120]} | "
                f"{item.get('rejection_reason', '未知原因')}"
            )

        refusal_reason = None
        if not accepted:
            refusal_reason = (
                "本轮生成的自然语言结论均未通过原文锚点或本地 NLI 校验。"
            )

        return {
            "accepted_claims": accepted,
            "rejected_claims": rejected,
            "claim_validation_reason": reason,
            "refusal_reason": refusal_reason,
            "citation_valid": bool(accepted),
            "execution_trace": append_trace(state, "validate_claims"),
        }

    def prepare_answer_retry(state: AgentState) -> dict[str, Any]:
        next_count = state.get("answer_retry_count", 0) + 1
        print(
            "[prepare_answer_retry] 使用同一批证据重新生成 Claim，"
            f"重试次数：{next_count}"
        )
        return {
            "answer_retry_count": next_count,
            "generated_claims": [],
            "accepted_claims": [],
            "rejected_claims": [],
            "answer": "",
            "citation_valid": False,
            "citation_map": {},
            "refusal_reason": None,
            "execution_trace": append_trace(state, "prepare_answer_retry"),
        }

    def render_answer(state: AgentState) -> dict[str, Any]:
        """由 Python 统一分配 S 编号并渲染最终自然语言答案。"""
        answer, citation_map, rendered_claims, cited_evidence = (
            assign_citations_and_render(
                accepted_claims=state.get("accepted_claims", []),
                evidence=state.get("evidence", []),
            )
        )
        if not answer:
            reason = "通过校验的 Claim 无法生成最终引用答案。"
            return {
                "answer": reason,
                "citation_valid": False,
                "citation_validation_reason": reason,
                "refusal_reason": reason,
                "execution_trace": append_trace(state, "render_answer"),
            }

        rejected_count = len(state.get("rejected_claims", []))
        reason = (
            f"引用链路通过：展示 {len(rendered_claims)} 条经原文锚点和本地 NLI "
            f"验证的结论，过滤 {rejected_count} 条不合格结论；"
            "引用编号由 Python 按最终使用的 Chunk 统一生成。"
        )
        print(f"[render_answer] {reason}")
        return {
            "answer": answer,
            "accepted_claims": rendered_claims,
            "evidence": cited_evidence,
            "citation_map": citation_map,
            "citation_valid": True,
            "citation_validation_reason": reason,
            "refusal_reason": None,
            "execution_trace": append_trace(state, "render_answer"),
        }

    def save_turn_memory(state: AgentState) -> dict[str, Any]:
        resolved_query = str(
            state.get("resolved_query")
            or state.get("query")
            or ""
        ).strip()

        previous_topic = state.get(
            "current_topic"
        )

        current_topic, detected_topic = (
            resolve_memory_topic(
                resolved_query=resolved_query,
                previous_topic=previous_topic,
                query_type=state.get(
                    "query_type"
                ),
                citation_valid=bool(
                    state.get(
                        "citation_valid",
                        False,
                    )
                ),
                accepted_claims=state.get(
                    "accepted_claims",
                    [],
                ),
            )
        )

        print(
            "[save_turn_memory] "
            f"提取到的主题：{detected_topic}"
        )

        print(
            "[save_turn_memory] "
            f"当前主题：{current_topic}"
        )
        return {
            "current_topic": current_topic,
            "last_query": resolved_query,
            "last_answer": state.get("answer", ""),
            "execution_trace": append_trace(state, "save_turn_memory"),
        }

    def refuse(state: AgentState) -> dict[str, Any]:
        reason = state.get("refusal_reason")
        if not reason:
            if state.get("query_type") == "out_of_scope":
                reason = "该问题与当前论文知识库无关。"
            else:
                reason = "当前论文证据不足，无法可靠回答该问题。"
        print(f"[refuse] {reason}")
        return {
            "answer": reason,
            "citation_valid": False,
            "citation_validation_reason": (
                state.get("claim_validation_reason")
                or state.get("evidence_check_reason")
            ),
            "refusal_reason": reason,
            "execution_trace": append_trace(state, "refuse"),
        }

    return {
        "prepare_turn": prepare_turn,
        "classify_query": classify_query,
        "decompose_query": decompose_query,
        "multi_retrieve": multi_retrieve,
        "retrieve": retrieve,
        "read_page": read_page,
        "evidence_check": evidence_check,
        "rewrite_query": rewrite_query,
        "generate_claims": generate_claims,
        "validate_claims": validate_claims,
        "prepare_answer_retry": prepare_answer_retry,
        "render_answer": render_answer,
        "save_turn_memory": save_turn_memory,
        "refuse": refuse,
    }
