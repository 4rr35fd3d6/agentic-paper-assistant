r"""使用真实本地 NLI 验收 AACFL/FedAvg 中文论文样本。

不调用 Ark。首次运行可能从 Hugging Face 下载 NLI 模型。
运行：python .\scripts\run_real_nli_validation.py
"""

from __future__ import annotations

import sys
from dataclasses import dataclass
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = PROJECT_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from nli_service import LocalNLIModel  # noqa: E402


@dataclass(frozen=True)
class Sample:
    name: str
    premise: str
    hypothesis: str
    expected_accept: bool


SAMPLES = [
    Sample(
        "AACFL忠实改写",
        "AACFL采用自动调整策略，只需调整少数客户端，大幅减少了通信压力。",
        "AACFL通过仅调整少量客户端来降低通信压力。",
        True,
    ),
    Sample(
        "AACFL方向反转",
        "AACFL显著提高了模型收敛速度。",
        "AACFL降低了模型收敛速度。",
        False,
    ),
    Sample(
        "AACFL否定反转",
        "AACFL可以自动纠正部分错误聚类的客户端。",
        "AACFL不能纠正错误聚类的客户端。",
        False,
    ),
    Sample(
        "FedAvg忠实定位",
        "FedAvg是一个标准未聚类的联邦学习基准框架。",
        "FedAvg属于未聚类联邦学习基准。",
        True,
    ),
    Sample(
        "主体互换",
        "FedAvg是标准未聚类基准；AACFL是自动调整聚类联邦学习框架。",
        "FedAvg是自动调整聚类联邦学习框架。",
        False,
    ),
    Sample(
        "通信机制扩大",
        "AACFL只需调整少数客户端，大幅减少了通信压力。",
        "AACFL完全消除了通信开销。",
        False,
    ),
    Sample(
        "部分支持复合句",
        "AACFL通过双端聚类和自动调整策略纠正聚类错误。",
        "AACFL通过双端聚类纠正错误，并且训练时间低于所有基线。",
        False,
    ),
    Sample(
        "正确表格数值",
        "表3-4时间消耗：FedAvg 5116；AACFL 5831。",
        "FedAvg的时间消耗为5116，AACFL为5831。",
        True,
    ),
    Sample(
        "表格数值互换",
        "表3-4时间消耗：FedAvg 5116；AACFL 5831。",
        "FedAvg的时间消耗为5831，AACFL为5116。",
        False,
    ),
    Sample(
        "正确大小方向",
        "表3-4时间消耗：FedAvg 5116；AACFL 5831。",
        "AACFL的时间消耗高于FedAvg。",
        True,
    ),
    Sample(
        "错误大小方向",
        "表3-4时间消耗：FedAvg 5116；AACFL 5831。",
        "AACFL的时间消耗低于FedAvg。",
        False,
    ),
    Sample(
        "无关事实",
        "AACFL包含服务端聚类、客户端聚类、更新警戒值和比较经验损失。",
        "AACFL使用ImageNet进行预训练。",
        False,
    ),
]


def main() -> None:
    model = LocalNLIModel(device="cpu")
    pairs = [(sample.premise, sample.hypothesis) for sample in SAMPLES]
    results = model.predict_many(pairs)

    passed = 0
    print("\n===== 真实 NLI 专项验收 =====")
    for sample, result in zip(SAMPLES, results):
        accepted = model.accepts(result)
        ok = accepted == sample.expected_accept
        passed += int(ok)
        print(
            f"[{'PASS' if ok else 'FAIL'}] {sample.name}\n"
            f"  expected_accept={sample.expected_accept}\n"
            f"  predicted={result.label}, entailment={result.entailment:.3f}, "
            f"neutral={result.neutral:.3f}, contradiction={result.contradiction:.3f}"
        )

    total = len(SAMPLES)
    print(f"\n结果：{passed}/{total} 通过；当前阈值={model.entailment_threshold:.2f}")
    if passed < total:
        print(
            "存在失败样本。不要直接调高或调低阈值；先区分是表格推理、"
            "主体互换还是普通忠实改写失败。数值归属已由 Python 额外拦截。"
        )
        raise SystemExit(1)


if __name__ == "__main__":
    main()
