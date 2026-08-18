"""本地自然语言蕴含（NLI）服务。

该模块只负责判断：给定论文证据（premise）是否蕴含自然语言结论
（hypothesis）。模型在本地运行，不调用 Ark API。
"""

from __future__ import annotations

import os
from typing import Iterable

import torch
try:
    from transformers import AutoModelForSequenceClassification, AutoTokenizer
except ImportError:  # 允许纯 Python 离线测试在未安装 transformers 时导入模块
    AutoModelForSequenceClassification = None
    AutoTokenizer = None

from claim_grounding import NLIResult


DEFAULT_NLI_MODEL_NAME = (
    "MoritzLaurer/multilingual-MiniLMv2-L6-mnli-xnli"
)
DEFAULT_ENTAILMENT_THRESHOLD = 0.55


class LocalNLIModel:
    """懒加载的本地多语言 NLI 模型。

    默认在 CPU 上运行，避免与 BGE Reranker 争抢显存。可以通过
    ``NLI_DEVICE=cuda`` 显式切换到 GPU。
    """

    def __init__(
        self,
        model_name: str | None = None,
        *,
        device: str | None = None,
        entailment_threshold: float | None = None,
        max_length: int = 512,
    ) -> None:
        self.model_name = (
            model_name
            or os.getenv("NLI_MODEL_NAME", "").strip()
            or DEFAULT_NLI_MODEL_NAME
        )
        requested_device = (
            device
            or os.getenv("NLI_DEVICE", "").strip()
            or "cpu"
        ).lower()
        if requested_device.startswith("cuda") and not torch.cuda.is_available():
            requested_device = "cpu"
        self.device = requested_device

        if entailment_threshold is None:
            raw_threshold = os.getenv(
                "NLI_ENTAILMENT_THRESHOLD",
                str(DEFAULT_ENTAILMENT_THRESHOLD),
            )
            try:
                entailment_threshold = float(raw_threshold)
            except ValueError:
                entailment_threshold = DEFAULT_ENTAILMENT_THRESHOLD

        self.entailment_threshold = min(
            max(float(entailment_threshold), 0.0),
            1.0,
        )
        self.max_length = int(max_length)

        self._tokenizer = None
        self._model = None
        self._label_to_index: dict[str, int] = {}

    def load(self) -> "LocalNLIModel":
        """加载 tokenizer 与模型；重复调用不会重复加载。"""
        if self._model is not None and self._tokenizer is not None:
            return self

        if AutoTokenizer is None or AutoModelForSequenceClassification is None:
            raise RuntimeError(
                "未安装 transformers，无法加载本地 NLI 模型。"
                "请先安装 requirements.txt 中的依赖。"
            )

        print("\n===== 加载本地 NLI =====")
        print(f"模型：{self.model_name}")
        print(f"设备：{self.device}")

        self._tokenizer = AutoTokenizer.from_pretrained(
            self.model_name,
        )
        self._model = AutoModelForSequenceClassification.from_pretrained(
            self.model_name,
        )
        self._model.to(self.device)
        self._model.eval()

        id2label = getattr(self._model.config, "id2label", {}) or {}
        label2id = getattr(self._model.config, "label2id", {}) or {}
        normalized: dict[str, int] = {}

        for raw_index, raw_label in id2label.items():
            try:
                index = int(raw_index)
            except (TypeError, ValueError):
                continue
            normalized[str(raw_label).strip().lower()] = index

        for raw_label, raw_index in label2id.items():
            try:
                index = int(raw_index)
            except (TypeError, ValueError):
                continue
            normalized[str(raw_label).strip().lower()] = index

        def find_label(*aliases: str) -> int | None:
            for alias in aliases:
                if alias in normalized:
                    return normalized[alias]
            return None

        resolved = {
            "entailment": find_label("entailment", "entails"),
            "neutral": find_label("neutral"),
            "contradiction": find_label("contradiction", "contradicts"),
        }

        if all(index is not None for index in resolved.values()):
            self._label_to_index = {
                key: int(value)
                for key, value in resolved.items()
                if value is not None
            }
        elif self.model_name == DEFAULT_NLI_MODEL_NAME:
            # 默认模型的标签顺序固定为 entailment / neutral / contradiction。
            self._label_to_index = {
                "entailment": 0,
                "neutral": 1,
                "contradiction": 2,
            }
        else:
            raise RuntimeError(
                "自定义 NLI 模型没有提供可识别的 entailment/neutral/"
                "contradiction 标签映射，请更换模型或修正其 config。"
            )
        return self

    def predict_many(
        self,
        pairs: Iterable[tuple[str, str]],
    ) -> list[NLIResult]:
        """批量判断若干 premise–hypothesis 对。"""
        prepared = [
            (str(premise).strip(), str(hypothesis).strip())
            for premise, hypothesis in pairs
        ]
        if not prepared:
            return []

        self.load()
        assert self._tokenizer is not None
        assert self._model is not None

        premises = [item[0] for item in prepared]
        hypotheses = [item[1] for item in prepared]

        encoded = self._tokenizer(
            premises,
            hypotheses,
            padding=True,
            truncation="only_first",
            max_length=self.max_length,
            return_tensors="pt",
        )
        encoded = {
            key: value.to(self.device)
            for key, value in encoded.items()
        }

        with torch.inference_mode():
            logits = self._model(**encoded).logits
            probabilities = torch.softmax(logits.float(), dim=-1).cpu()

        entailment_index = self._label_to_index["entailment"]
        neutral_index = self._label_to_index["neutral"]
        contradiction_index = self._label_to_index["contradiction"]

        results: list[NLIResult] = []
        for row in probabilities:
            entailment = float(row[entailment_index])
            neutral = float(row[neutral_index])
            contradiction = float(row[contradiction_index])

            score_map = {
                "entailment": entailment,
                "neutral": neutral,
                "contradiction": contradiction,
            }
            label = max(score_map, key=score_map.get)
            results.append(
                NLIResult(
                    label=label,
                    entailment=entailment,
                    neutral=neutral,
                    contradiction=contradiction,
                )
            )

        return results

    def accepts(self, result: NLIResult) -> bool:
        """按标签和阈值判断该结论是否可以保留。"""
        return (
            result.label == "entailment"
            and result.entailment >= self.entailment_threshold
        )


def create_local_nli_model() -> LocalNLIModel:
    """使用环境变量创建本地 NLI 服务（模型仍为懒加载）。"""
    return LocalNLIModel()
