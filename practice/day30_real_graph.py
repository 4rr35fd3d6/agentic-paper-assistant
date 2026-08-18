"""
运行正式 LangGraph 论文问答流程。

当前测试：
分类 → 真实检索 → 真实答案生成 → Checkpointer 多轮会话
"""

from __future__ import annotations

import sys
import traceback
from pathlib import Path


# ============================================================
# 1. 配置项目路径
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = PROJECT_ROOT / "src"
PRACTICE_DIR = PROJECT_ROOT / "practice"

# 必须先设置导入路径，再导入项目内部模块。
# 同时加入三个目录，以兼容项目里现有的两种导入写法：
#   from src.xxx import ...
#   from xxx import ...
for current_path in (
    PROJECT_ROOT,
    SRC_DIR,
    PRACTICE_DIR,
):
    path_text = str(current_path)

    if path_text not in sys.path:
        sys.path.insert(0, path_text)


# 路径准备完成后，再导入第三方和项目模块。
from langgraph.checkpoint.memory import InMemorySaver

import day22_reranker_rag as rag_backend
from agent_graph import create_agent_graph
from agent_state import create_turn_input
from runtime_loader import load_real_runtime
from nli_service import create_local_nli_model


# ============================================================
# 2. 显示运行结果
# ============================================================


def print_result(
    final_state: dict,
) -> None:
    """显示 LangGraph 执行结果。"""
    print("\n===== LangGraph 执行结果 =====")

    sub_queries = final_state.get(
        "sub_queries",
        [],
    )

    if sub_queries:
        print("比较子问题：")

        for index, sub_query in enumerate(
            sub_queries,
            start=1,
        ):
            print(f"  {index}. {sub_query}")

    print(
        "问题类型："
        f"{final_state.get('query_type')}"
    )

    trace = final_state.get(
        "execution_trace",
        [],
    )

    print(
        "执行路线："
        f"{' → '.join(trace)}"
    )

    print(
        "工具调用次数："
        f"{final_state.get('tool_call_count', 0)}"
    )

    print(
        "Ark 调用次数："
        f"{final_state.get('llm_call_count', 0)}"
    )

    evidence = final_state.get(
        "evidence",
        [],
    )

    print(
        "证据数量："
        f"{len(evidence)}"
    )

    print(
        "查询改写次数："
        f"{final_state.get('retry_count', 0)}"
    )

    rewritten_query = final_state.get(
        "rewritten_query"
    )

    if rewritten_query:
        print(
            "改写后的查询："
            f"{rewritten_query}"
        )

    print(
        "去重后 Chunk 数量："
        f"{len(final_state.get('seen_chunk_ids', []))}"
    )

    if evidence:
        first_evidence = evidence[0]

        print(
            "首条证据："
            f"[{first_evidence.get('citation_id', 'S?')}] "
            f"第{first_evidence.get('page_number', '?')}页"
        )

    citation_reason = final_state.get(
        "citation_validation_reason"
    )

    if final_state.get("citation_valid"):
        citation_status = "通过"
    elif citation_reason:
        citation_status = "未通过"
    else:
        citation_status = "未执行"

    print(f"引用检查：{citation_status}")

    if citation_reason:
        print(
            "引用检查说明："
            f"{citation_reason}"
        )

    print("\n===== 最终回答 =====")
    print(
        final_state.get(
            "answer",
            "没有生成回答。",
        )
    )


# ============================================================
# 3. 主程序
# ============================================================


def main() -> None:
    """模型只加载一次，随后可以连续提问。"""
    runtime, reranker = load_real_runtime(rag_backend)

    print("\n===== 创建正式 LangGraph =====")

    # InMemorySaver 必须只创建一次，并放在交互循环外。
    # 同一个程序进程中，相同 thread_id 会共享会话状态。
    checkpointer = InMemorySaver()

    nli_model = create_local_nli_model()
    graph = create_agent_graph(
        runtime=runtime,
        reranker=reranker,
        rag_backend=rag_backend,
        nli_model=nli_model,
        checkpointer=checkpointer,
    )

    thread_id = "paper-agent-demo"

    graph_config = {
        "configurable": {
            "thread_id": thread_id,
        },
        "recursion_limit": 20,
    }

    print(
        "LangGraph 已准备完成。"
        "\n输入 q、quit 或 exit 退出。"
    )

    while True:
        question = input(
            "\n请输入论文问题："
        ).strip()

        if question.lower() in {
            "q",
            "quit",
            "exit",
        }:
            print("已退出 LangGraph 论文助手。")
            break

        if not question:
            print("问题不能为空。")
            continue

        # 这里只提交本轮问题，不覆盖 Checkpointer 中保存的会话记忆。
        turn_input = create_turn_input(
            question
        )

        print("\n===== LangGraph 开始运行 =====")

        try:
            final_state = graph.invoke(
                turn_input,
                config=graph_config,
            )

            snapshot = graph.get_state(
                graph_config
            )

            checkpoint_query = snapshot.values.get(
                "query",
                "",
            )

            print(
                "解析后问题："
                f"{final_state.get('resolved_query')}"
            )

            print(
                "当前会话主题："
                f"{final_state.get('current_topic')}"
            )

            print(
                "最近问题："
                f"{final_state.get('last_query')}"
            )

            print(
                "[checkpoint] "
                f"当前线程：{thread_id}"
            )

            print(
                "[checkpoint] "
                f"已保存问题：{checkpoint_query}"
            )

        except Exception as error:
            print(
                "\nLangGraph 运行失败："
                f"{type(error).__name__}: {error}"
            )

            print("\n===== 完整错误堆栈 =====")
            traceback.print_exc()

            print("\n===== 底层异常 =====")
            current_error = error.__cause__

            if current_error is None:
                print("没有额外的底层异常。")

            while current_error is not None:
                print(
                    f"{type(current_error).__name__}: "
                    f"{current_error}"
                )
                current_error = current_error.__cause__

            continue

        print_result(final_state)


if __name__ == "__main__":
    main()
