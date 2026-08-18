"""Claim + Chunk + 原文锚点 + 本地 NLI 的纯离线回归测试。"""

from __future__ import annotations

import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = PROJECT_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from agent_routes import (  # noqa: E402
    route_after_claim_validation,
    route_after_evidence_check,
)
from claim_grounding import (  # noqa: E402
    assign_citations_and_render,
    clean_reasoning_evidence,
    extract_quote_context,
    get_evidence_text,
    parse_claim_output,
    parse_evidence_decision,
    quote_is_in_chunk,
    validate_claims,
)
from claim_grounding import NLIResult  # noqa: E402


class FakeNLI:
    entailment_threshold = 0.55

    def predict_many(self, pairs):
        results = []
        for premise, hypothesis in pairs:
            if "错误方向" in hypothesis or "降低收敛速度" in hypothesis:
                results.append(
                    NLIResult(
                        label="contradiction",
                        entailment=0.02,
                        neutral=0.03,
                        contradiction=0.95,
                    )
                )
            elif "无关结论" in hypothesis:
                results.append(
                    NLIResult(
                        label="neutral",
                        entailment=0.08,
                        neutral=0.88,
                        contradiction=0.04,
                    )
                )
            else:
                results.append(
                    NLIResult(
                        label="entailment",
                        entailment=0.91,
                        neutral=0.06,
                        contradiction=0.03,
                    )
                )
        return results

    def accepts(self, result):
        return result.label == "entailment" and result.entailment >= 0.55


def evidence_fixture():
    return [
        {
            "chunk_id": 128,
            "page_number": 35,
            "content": (
                "AACFL采用自动调整策略，只需调整少数客户端，"
                "大幅减少了通信压力。"
            ),
        },
        {
            "chunk_id": 56,
            "page_number": 37,
            "content": "FedAvg[1]：一个标准未聚类的联邦学习基准框架。",
        },
        {
            "chunk_id": 91,
            "page_number": 41,
            "content": (
                "表3-4 在k=4,n=100下的时间消耗，单位：s。"
                "FedAvg 5116 9214 5042；AACFL 5831 10168 5799。"
            ),
        },
    ]


def test_evidence_field_skips_blank_higher_priority_value():
    assert get_evidence_text(
        {"text": "   ", "content": "真实证据正文"}
    ) == "真实证据正文"


def test_evidence_decision_only_uses_first_nonempty_line():
    assert parse_evidence_decision(
        "SUFFICIENT\n证据充分，不是 INSUFFICIENT。"
    )[0] is True
    assert parse_evidence_decision(
        "INSUFFICIENT\n补充后才可能 SUFFICIENT。"
    )[0] is False


def test_claim_parser_uses_chunk_id_and_removes_model_citations():
    raw = r'''
    {
      "claims": [{
        "section": "框架定位",
        "text": "FedAvg是标准未聚类的联邦学习基准框架。[S9]",
        "supports": [{
          "chunk_id": "56",
          "evidence_quote": "FedAvg[1]：一个标准未聚类的联邦学习基准框架。"
        }]
      }]
    }
    '''
    claims = parse_claim_output(raw)
    assert claims[0]["text"] == "FedAvg是标准未聚类的联邦学习基准框架。"
    assert claims[0]["supports"][0]["chunk_id"] == 56



def test_claim_parser_tolerates_code_fence_trailing_comma_and_smart_quotes():
    raw = """```json
    {“claims”: [{
      “section”: “框架定位”,
      “text”: “FedAvg是标准未聚类框架。”,
      “supports”: [{
        “chunk_id”: 56.0,
        “evidence_quote”: “FedAvg[1]：一个标准未聚类的联邦学习基准框架。”,
      }],
    }],}
    ```"""
    claims = parse_claim_output(raw)
    assert len(claims) == 1
    assert claims[0]["supports"][0]["chunk_id"] == 56

def test_quote_must_belong_to_the_declared_chunk():
    assert quote_is_in_chunk(
        "只需调整少数客户端，大幅减少了通信压力",
        evidence_fixture()[0]["content"],
    )
    assert not quote_is_in_chunk(
        "FedAvg[1]：一个标准未聚类的联邦学习基准框架。",
        evidence_fixture()[0]["content"],
    )


def test_context_is_generated_by_python_around_quote():
    context = extract_quote_context(
        evidence_fixture()[0]["content"],
        "只需调整少数客户端，大幅减少了通信压力",
    )
    assert "AACFL采用自动调整策略" in context
    assert "大幅减少了通信压力" in context


def test_faithful_paraphrase_is_accepted_by_local_nli():
    claims = [
        {
            "section": "核心机制",
            "text": "AACFL通过只调整少数被错误分组的客户端降低通信压力。",
            "supports": [
                {
                    "chunk_id": 128,
                    "evidence_quote": "只需调整少数客户端，大幅减少了通信压力",
                }
            ],
        }
    ]
    accepted, rejected, _ = validate_claims(
        query="AACFL是什么？",
        query_type="fact",
        evidence=evidence_fixture(),
        claims=claims,
        nli_model=FakeNLI(),
    )
    assert len(accepted) == 1
    assert rejected == []


def test_contradictory_claim_is_rejected_without_affecting_other_claims():
    claims = [
        {
            "section": "实验表现",
            "text": "AACFL降低收敛速度，属于错误方向。",
            "supports": [
                {
                    "chunk_id": 128,
                    "evidence_quote": "只需调整少数客户端，大幅减少了通信压力",
                }
            ],
        },
        {
            "section": "框架定位",
            "text": "FedAvg是标准未聚类的联邦学习基准框架。",
            "supports": [
                {
                    "chunk_id": 56,
                    "evidence_quote": "FedAvg[1]：一个标准未聚类的联邦学习基准框架。",
                }
            ],
        },
    ]
    accepted, rejected, _ = validate_claims(
        query="FedAvg是什么？",
        query_type="fact",
        evidence=evidence_fixture(),
        claims=claims,
        nli_model=FakeNLI(),
    )
    assert len(accepted) == 1
    assert accepted[0]["text"].startswith("FedAvg")
    assert len(rejected) == 1


def test_claim_with_number_missing_from_evidence_is_rejected():
    claims = [
        {
            "section": "实验表现",
            "text": "AACFL的时间消耗是9999秒。",
            "supports": [
                {
                    "chunk_id": 91,
                    "evidence_quote": "AACFL 5831 10168 5799",
                }
            ],
        }
    ]
    accepted, rejected, _ = validate_claims(
        query="AACFL时间消耗是多少？",
        query_type="fact",
        evidence=evidence_fixture(),
        claims=claims,
        nli_model=FakeNLI(),
    )
    assert accepted == []
    assert "不存在的数值" in rejected[0]["rejection_reason"]


def test_numeric_method_binding_rejects_swapped_values():
    claims = [
        {
            "section": "实验表现",
            "text": "FedAvg为5831秒，AACFL为5116秒。",
            "supports": [
                {
                    "chunk_id": 91,
                    "evidence_quote": (
                        "FedAvg 5116 9214 5042；AACFL 5831 10168 5799"
                    ),
                }
            ],
        }
    ]
    accepted, rejected, _ = validate_claims(
        query="AACFL和FedAvg的时间消耗是多少？",
        query_type="comparison",
        comparison_targets=[
            "AACFL",
            "FedAvg",
        ],
        evidence=evidence_fixture(),
        claims=claims,
        nli_model=FakeNLI(),
    )
    assert accepted == []
    assert "数值归属不一致" in rejected[0]["rejection_reason"]


def test_numeric_method_binding_accepts_correct_values():
    claims = [
        {
            "section": "实验表现",
            "text": "FedAvg为5116秒，AACFL为5831秒。",
            "supports": [
                {
                    "chunk_id": 91,
                    "evidence_quote": (
                        "FedAvg 5116 9214 5042；AACFL 5831 10168 5799"
                    ),
                }
            ],
        }
    ]
    accepted, rejected, _ = validate_claims(
        query="AACFL和FedAvg的时间消耗是多少？",
        query_type="comparison",
        comparison_targets=[
            "AACFL",
            "FedAvg",
        ],
        evidence=evidence_fixture(),
        claims=claims,
        nli_model=FakeNLI(),
    )
    assert len(accepted) == 1
    assert rejected == []


def test_atomic_claim_rejects_multiple_supports():
    claims = [
        {
            "section": "实验表现",
            "text": "在该实验设置下，AACFL训练时间高于FedAvg。",
            "supports": [
                {
                    "chunk_id": 91,
                    "evidence_quote": "FedAvg 5116 9214 5042；AACFL 5831 10168 5799",
                },
                {
                    "chunk_id": 128,
                    "evidence_quote": "AACFL采用自动调整策略，只需调整少数客户端",
                },
            ],
        }
    ]
    accepted, rejected, reason = validate_claims(
        query="AACFL和FedAvg有什么区别？",
        query_type="comparison",
        evidence=evidence_fixture(),
        claims=claims,
        nli_model=FakeNLI(),
    )
    assert accepted == []
    assert "必须且只能绑定一个" in rejected[0]["rejection_reason"]
    assert "主要失败原因" in reason


def test_claim_parser_rejects_multiple_supports_instead_of_truncating():
    raw = """
    {
      "claims": [{
        "section": "实验表现",
        "text": "AACFL的训练时间高于FedAvg。",
        "supports": [
          {
            "chunk_id": 91,
            "evidence_quote": "FedAvg 5116 9214 5042；AACFL 5831 10168 5799"
          },
          {
            "chunk_id": 128,
            "evidence_quote": "只需调整少数客户端，大幅减少了通信压力"
          }
        ]
      }]
    }
    """
    assert parse_claim_output(raw) == []


def test_python_assigns_contiguous_citations_after_filtering():
    accepted = [
        {
            "section": "框架定位",
            "text": "FedAvg是标准未聚类的联邦学习基准框架。",
            "supports": [
                {
                    "chunk_id": 56,
                    "evidence_quote": "FedAvg[1]：一个标准未聚类的联邦学习基准框架。",
                    "evidence_context": "FedAvg[1]：一个标准未聚类的联邦学习基准框架。",
                }
            ],
        },
        {
            "section": "核心机制",
            "text": "AACFL通过调整少数客户端降低通信压力。",
            "supports": [
                {
                    "chunk_id": 128,
                    "evidence_quote": "只需调整少数客户端，大幅减少了通信压力",
                    "evidence_context": evidence_fixture()[0]["content"],
                }
            ],
        },
    ]
    answer, citation_map, rendered, cited = assign_citations_and_render(
        accepted_claims=accepted,
        evidence=evidence_fixture(),
    )
    assert citation_map == {56: "S1", 128: "S2"}
    assert "[S1]" in answer and "[S2]" in answer
    assert [item["citation_id"] for item in cited] == ["S1", "S2"]
    assert len(rendered) == 2


def test_comparison_requires_both_entities_after_validation():
    claims = [
        {
            "section": "核心机制",
            "text": "AACFL通过调整少数客户端降低通信压力。",
            "supports": [
                {
                    "chunk_id": 128,
                    "evidence_quote": "只需调整少数客户端，大幅减少了通信压力",
                }
            ],
        }
    ]
    accepted, rejected, _ = validate_claims(
        query="AACFL和FedAvg有什么区别？",
        query_type="comparison",
        evidence=evidence_fixture(),
        claims=claims,
        nli_model=FakeNLI(),
    )
    assert accepted == []
    assert any("未同时覆盖" in item["rejection_reason"] for item in rejected)


def test_comparison_retrieval_retry_uses_rounds_not_tool_calls():
    state = {
        "evidence_sufficient": False,
        "query_type": "comparison",
        "retry_count": 0,
        "tool_call_count": 2,
        "retrieval_round_count": 1,
    }
    assert route_after_evidence_check(state) == "rewrite_query"

    state["retrieval_round_count"] = 2
    assert route_after_evidence_check(state) == "refuse"


def test_all_claims_failed_routes_to_answer_retry_then_refusal():
    assert route_after_claim_validation(
        {"accepted_claims": [], "answer_retry_count": 0}
    ) == "prepare_answer_retry"
    assert route_after_claim_validation(
        {"accepted_claims": [], "answer_retry_count": 1}
    ) == "refuse"
    assert route_after_claim_validation(
        {"accepted_claims": [{"text": "ok"}], "answer_retry_count": 0}
    ) == "render_answer"



class ShortNLI(FakeNLI):
    def predict_many(self, pairs):
        results = super().predict_many(pairs)
        return results[:1]


def test_nli_result_count_mismatch_rejects_unmatched_claims():
    claims = [
        {
            "section": "框架定位",
            "text": "FedAvg是标准未聚类的联邦学习基准框架。",
            "supports": [
                {
                    "chunk_id": 56,
                    "evidence_quote": (
                        "FedAvg[1]：一个标准未聚类的联邦学习基准框架。"
                    ),
                }
            ],
        },
        {
            "section": "核心机制",
            "text": "AACFL通过调整少数客户端降低通信压力。",
            "supports": [
                {
                    "chunk_id": 128,
                    "evidence_quote": (
                        "只需调整少数客户端，大幅减少了通信压力"
                    ),
                }
            ],
        },
    ]
    accepted, rejected, _ = validate_claims(
        query="AACFL和FedAvg有什么区别？",
        query_type="comparison",
        evidence=evidence_fixture(),
        claims=claims,
        nli_model=ShortNLI(),
    )
    assert accepted == []
    assert any(
        "NLI 返回数量" in item["rejection_reason"]
        for item in rejected
    )

def test_table_cleanup_removes_previous_table_tail():
    contaminated = (
        "FeSEM 2567 5897 3125 WeCFL 2583 5812 3217 "
        "IFCA 4054 8240 4545 AACFL 2787 5990 3364 "
        "表3-4 在k=4,n=100下的时间消耗，单位：s "
        "FedAvg 5116 9214 5042 AACFL 5831 10168 5799"
    )
    cleaned = clean_reasoning_evidence(contaminated)
    assert cleaned.startswith("表3-4")
    assert "2787" not in cleaned
    assert "5831" in cleaned


def test_rendered_evidence_uses_clean_reasoning_text():
    contaminated = (
        "FeSEM 2567 5897 3125 WeCFL 2583 5812 3217 "
        "IFCA 4054 8240 4545 AACFL 2787 5990 3364 "
        "表3-4 在k=4,n=100下的时间消耗，单位：s。"
        "FedAvg 5116 9214 5042；AACFL 5831 10168 5799。"
    )
    evidence = [{
        "chunk_id": 91,
        "page_number": 41,
        "content": contaminated,
    }]
    accepted = [{
        "section": "实验表现",
        "text": "FedAvg在该表中的时间消耗包括5116秒。",
        "supports": [{
            "chunk_id": 91,
            "evidence_quote": "FedAvg 5116 9214 5042",
            "evidence_context": clean_reasoning_evidence(contaminated),
        }],
    }]
    _, _, _, cited = assign_citations_and_render(
        accepted_claims=accepted,
        evidence=evidence,
    )
    assert len(cited) == 1
    assert cited[0]["content"].startswith("表3-4")
    assert "2787" not in cited[0]["content"]
    assert "5831" in cited[0]["content"]
def test_direction_without_numbers_rejects_wrong_relation():
    claims = [
        {
            "section": "实验表现",
            "text": (
                "AACFL的时间消耗低于FedAvg。"
            ),
            "supports": [
                {
                    "chunk_id": 91,
                    "evidence_quote": (
                        "FedAvg 5116 9214 5042；"
                        "AACFL 5831 10168 5799"
                    ),
                }
            ],
        }
    ]

    accepted, rejected, _ = validate_claims(
        query="AACFL和FedAvg的时间消耗谁更高？",
        query_type="comparison",
        comparison_targets=[
            "AACFL",
            "FedAvg",
        ],
        evidence=evidence_fixture(),
        claims=claims,
        nli_model=FakeNLI(),
    )

    assert accepted == []
    assert len(rejected) == 1
    assert "大小关系" in rejected[0][
        "rejection_reason"
    ]


def test_dataset_name_is_not_treated_as_method_entity():

    evidence = evidence_fixture()

    evidence[2]["content"] = (
        "表3-4 在k=4,n=100下的时间消耗，单位：s。"
        "方法 Emnist Cifar10 Fashionmnist "
        "FedAvg 5116 9214 5042；"
        "AACFL 5831 10168 5799。"
    )

    claims = [
        {
            "section": "实验表现",
            "text": (
                "FedAvg在Emnist上为5116秒，"
                "AACFL在Emnist上为5831秒。"
            ),
            "supports": [
                {
                    "chunk_id": 91,
                    "evidence_quote": (
                        "方法 Emnist Cifar10 Fashionmnist "
                        "FedAvg 5116 9214 5042；"
                        "AACFL 5831 10168 5799"
                    ),
                }
            ],
        }
    ]

    accepted, rejected, _ = validate_claims(
        query=(
            "AACFL和FedAvg在Emnist上的"
            "时间消耗是多少？"
        ),
        query_type="comparison",
        comparison_targets=[
            "AACFL",
            "FedAvg",
        ],
        evidence=evidence,
        claims=claims,
        nli_model=FakeNLI(),
    )

    assert len(accepted) == 1
    assert rejected == []


class FailIfCalledNLI:
    """用于确认确定性规则生效后不会再调用 NLI。"""

    def predict_many(self, pairs):
        raise AssertionError(
            "该 Claim 应由 Python 确定性验证，"
            "不应该调用 NLI。"
        )

    def accepts(self, result):
        raise AssertionError(
            "该 Claim 不应该进入 NLI 接受判断。"
        )


def table_evidence_fixture():
    return [
        {
            "chunk_id": 91,
            "page_number": 35,
            "content": (
                "表3-4 在k=4,n=100下的时间消耗，单位：s。"
                "方法 Emnist Cifar10 Fashionmnist "
                "FedAvg 5116 9214 5042；"
                "AACFL 5831 10168 5799。"
            ),
        }
    ]


def test_correct_numeric_claim_bypasses_unreliable_nli():
    quote = (
        "方法 Emnist Cifar10 Fashionmnist "
        "FedAvg 5116 9214 5042；"
        "AACFL 5831 10168 5799"
    )

    claims = [
        {
            "section": "实验表现",
            "text": (
                "FedAvg在Emnist上为5116秒，"
                "AACFL在Emnist上为5831秒。"
            ),
            "supports": [
                {
                    "chunk_id": 91,
                    "evidence_quote": quote,
                }
            ],
        }
    ]

    accepted, rejected, _ = validate_claims(
        query="AACFL和FedAvg的时间消耗是多少？",
        query_type="comparison",
        comparison_targets=[
            "AACFL",
            "FedAvg",
        ],
        evidence=table_evidence_fixture(),
        claims=claims,
        nli_model=FailIfCalledNLI(),
    )

    assert len(accepted) == 1
    assert rejected == []
    assert (
        accepted[0]["validation_mode"]
        == "python_numeric"
    )


def test_swapped_numeric_claim_is_rejected_before_nli():
    quote = (
        "方法 Emnist Cifar10 Fashionmnist "
        "FedAvg 5116 9214 5042；"
        "AACFL 5831 10168 5799"
    )

    claims = [
        {
            "section": "实验表现",
            "text": (
                "FedAvg在Emnist上为5831秒，"
                "AACFL在Emnist上为5116秒。"
            ),
            "supports": [
                {
                    "chunk_id": 91,
                    "evidence_quote": quote,
                }
            ],
        }
    ]

    accepted, rejected, _ = validate_claims(
        query="AACFL和FedAvg的时间消耗是多少？",
        query_type="comparison",
        comparison_targets=[
            "AACFL",
            "FedAvg",
        ],
        evidence=table_evidence_fixture(),
        claims=claims,
        nli_model=FailIfCalledNLI(),
    )

    assert accepted == []
    assert len(rejected) == 1
    assert "数值归属不一致" in rejected[0][
        "rejection_reason"
    ]


def test_correct_direction_claim_bypasses_nli():
    quote = (
        "方法 Emnist Cifar10 Fashionmnist "
        "FedAvg 5116 9214 5042；"
        "AACFL 5831 10168 5799"
    )

    claims = [
        {
            "section": "实验表现",
            "text": (
                "AACFL的时间消耗高于FedAvg。"
            ),
            "supports": [
                {
                    "chunk_id": 91,
                    "evidence_quote": quote,
                }
            ],
        }
    ]

    accepted, rejected, _ = validate_claims(
        query="AACFL和FedAvg谁的时间消耗更高？",
        query_type="comparison",
        comparison_targets=[
            "AACFL",
            "FedAvg",
        ],
        evidence=table_evidence_fixture(),
        claims=claims,
        nli_model=FailIfCalledNLI(),
    )

    assert len(accepted) == 1
    assert rejected == []


def test_wrong_direction_claim_is_rejected_before_nli():
    quote = (
        "方法 Emnist Cifar10 Fashionmnist "
        "FedAvg 5116 9214 5042；"
        "AACFL 5831 10168 5799"
    )

    claims = [
        {
            "section": "实验表现",
            "text": (
                "AACFL的时间消耗低于FedAvg。"
            ),
            "supports": [
                {
                    "chunk_id": 91,
                    "evidence_quote": quote,
                }
            ],
        }
    ]

    accepted, rejected, _ = validate_claims(
        query="AACFL和FedAvg谁的时间消耗更高？",
        query_type="comparison",
        comparison_targets=[
            "AACFL",
            "FedAvg",
        ],
        evidence=table_evidence_fixture(),
        claims=claims,
        nli_model=FailIfCalledNLI(),
    )

    assert accepted == []
    assert len(rejected) == 1
    assert "大小关系" in rejected[0][
        "rejection_reason"
    ]


def test_non_numeric_dual_target_claim_is_rejected():
    evidence = [
        {
            "chunk_id": 200,
            "page_number": 20,
            "content": (
                "FedAvg是一个标准未聚类的联邦学习基准框架。"
                "AACFL是一种自动调整聚类联邦学习框架。"
            ),
        }
    ]

    quote = (
        "FedAvg是一个标准未聚类的联邦学习基准框架。"
        "AACFL是一种自动调整聚类联邦学习框架。"
    )

    claims = [
        {
            "section": "框架定位",
            "text": (
                "AACFL采用未聚类基准框架，"
                "而FedAvg采用自动调整聚类框架。"
            ),
            "supports": [
                {
                    "chunk_id": 200,
                    "evidence_quote": quote,
                }
            ],
        }
    ]

    accepted, rejected, _ = validate_claims(
        query="AACFL和FedAvg有什么区别？",
        query_type="comparison",
        comparison_targets=[
            "AACFL",
            "FedAvg",
        ],
        evidence=evidence,
        claims=claims,
        nli_model=FailIfCalledNLI(),
    )

    assert accepted == []
    assert len(rejected) == 1
    assert "单对象原子结论" in rejected[0][
        "rejection_reason"
    ]


def test_single_target_claim_requires_target_in_context():
    evidence = [
        {
            "chunk_id": 201,
            "page_number": 21,
            "content": (
                "FedAvg是一个标准未聚类的"
                "联邦学习基准框架。"
            ),
        }
    ]

    claims = [
        {
            "section": "框架定位",
            "text": (
                "AACFL是一种自动调整"
                "聚类联邦学习框架。"
            ),
            "supports": [
                {
                    "chunk_id": 201,
                    "evidence_quote": (
                        "FedAvg是一个标准未聚类的"
                        "联邦学习基准框架。"
                    ),
                }
            ],
        }
    ]

    accepted, rejected, _ = validate_claims(
        query="AACFL和FedAvg有什么区别？",
        query_type="comparison",
        comparison_targets=[
            "AACFL",
            "FedAvg",
        ],
        evidence=evidence,
        claims=claims,
        nli_model=FailIfCalledNLI(),
    )

    assert accepted == []
    assert len(rejected) == 1
    assert "没有在对应证据上下文中明确出现" in (
        rejected[0]["rejection_reason"]
    )