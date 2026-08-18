"""自然语言结论、原文锚点与引用编号的确定性处理。

大模型只生成自然语言 claim、稳定 chunk_id 和连续原文 quote；
Python 验证证据身份与原文锚点，本地 NLI 验证语义蕴含，最后统一
分配 [S1]、[S2] 展示编号。
"""

from __future__ import annotations

import ast
import json
import re
from collections import Counter, OrderedDict
from dataclasses import dataclass
from decimal import Decimal, InvalidOperation
from typing import Any, Protocol




@dataclass(frozen=True)
class NLIResult:
    """本地 NLI 的统一结果结构。"""

    label: str
    entailment: float
    neutral: float
    contradiction: float


def parse_evidence_decision(output_text: Any) -> tuple[bool, str]:
    """只解析第一条非空行开头的 SUFFICIENT/INSUFFICIENT。"""
    lines = [
        line.strip()
        for line in str(output_text or "").splitlines()
        if line.strip()
    ]
    if not lines:
        return False, "证据判断模型没有返回有效内容。"

    first_line = re.sub(
        r"^[\s>*#`_-]+",
        "",
        lines[0],
    ).strip().strip("*_`")
    match = re.match(
        r"^(SUFFICIENT|INSUFFICIENT)(?![A-Z])"
        r"(?:\s*[:：-]?\s*(.*))?$",
        first_line,
        flags=re.IGNORECASE,
    )
    if match is None:
        return (
            False,
            "证据判断模型第一行没有按照 SUFFICIENT/INSUFFICIENT 格式返回。",
        )

    decision = match.group(1).upper()
    reason_parts: list[str] = []
    inline_reason = (match.group(2) or "").strip()
    if inline_reason:
        reason_parts.append(inline_reason)
    reason_parts.extend(lines[1:])
    reason = " ".join(reason_parts).strip() or "模型未提供具体判断原因。"
    return decision == "SUFFICIENT", reason


MAX_CLAIMS = 5
MAX_SUPPORTS_PER_CLAIM = 1
MIN_QUOTE_LENGTH = 8
MAX_QUOTE_LENGTH = 240
MAX_CLAIM_LENGTH = 160

_ALLOWED_SECTIONS = {
    "框架定位",
    "核心机制",
    "训练流程",
    "数据异构处理",
    "实验表现",
    "优势与局限",
    "主要结论",
    "指定页面内容",
}

_SECTION_ALIASES = {
    "方法定位": "框架定位",
    "定位": "框架定位",
    "工作机制": "核心机制",
    "机制": "核心机制",
    "流程": "训练流程",
    "性能": "实验表现",
    "实验结果": "实验表现",
    "优缺点": "优势与局限",
    "结论": "主要结论",
}


class NLIProtocol(Protocol):
    entailment_threshold: float

    def predict_many(
        self,
        pairs: list[tuple[str, str]],
    ) -> list[NLIResult]: ...

    def accepts(self, result: NLIResult) -> bool: ...


def get_evidence_text(item: dict[str, Any]) -> str:
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


def normalize_display_text(value: Any) -> str:
    """压缩异常空白，用于展示和 JSON 字段清理。"""
    return " ".join(str(value or "").split()).strip()


def normalize_exact_text(value: Any) -> str:
    """移除全部空白，用于连续原文锚点匹配。"""
    return re.sub(r"\s+", "", str(value or "")).strip()


def normalize_chunk_id(value: Any) -> int | None:
    """将模型返回的 Chunk 编号规范化为正整数。"""
    if isinstance(value, bool) or value is None:
        return None
    try:
        number = Decimal(str(value).strip())
    except (InvalidOperation, TypeError, ValueError):
        return None
    if number != number.to_integral_value():
        return None
    result = int(number)
    return result if result > 0 else None


def clean_reasoning_evidence(content: Any) -> str:
    """清理模型推理证据，避免相邻表格尾部串入当前表格。"""
    text = normalize_display_text(content)
    if not text:
        return ""

    for table_match in re.finditer(
        r"表\s*\d+\s*[-－—]\s*\d+",
        text,
        flags=re.IGNORECASE,
    ):
        prefix = text[:table_match.start()]
        numeric_count = len(
            re.findall(r"(?<![A-Za-z])\d+(?:\.\d+)?", prefix)
        )
        method_like_count = len(
            re.findall(r"\b[A-Za-z][A-Za-z0-9+_-]{2,}\b", prefix)
        )
        if (
            table_match.start() <= 500
            and numeric_count >= 6
            and method_like_count >= 3
        ):
            return text[table_match.start():].strip()

    return text


def message_content_to_text(content: Any) -> str:
    """兼容字符串和常见内容块列表，提取模型文本。"""
    if isinstance(content, str):
        return content.strip()
    if isinstance(content, list):
        parts: list[str] = []
        for item in content:
            if isinstance(item, str):
                parts.append(item)
            elif isinstance(item, dict):
                value = item.get("text") or item.get("content") or ""
                if value:
                    parts.append(str(value))
        return "\n".join(parts).strip()
    return str(content or "").strip()


def _extract_json_object(text: str) -> dict[str, Any] | None:
    """本地解析模型 JSON；容忍代码块、前后说明和少量格式噪声。"""
    cleaned = re.sub(
        r"^```(?:json)?\s*",
        "",
        text.strip(),
        flags=re.IGNORECASE,
    )
    cleaned = re.sub(r"\s*```$", "", cleaned).strip()

    candidates: list[str] = [cleaned]
    start = cleaned.find("{")
    end = cleaned.rfind("}")
    if start >= 0 and end > start:
        candidates.append(cleaned[start:end + 1])

    expanded: list[str] = []
    for candidate in candidates:
        expanded.append(candidate)
        repaired = (
            candidate
            .replace("“", '\"')
            .replace("”", '\"')
            .replace("‘", "'")
            .replace("’", "'")
        )
        repaired = re.sub(r",\s*([}\]])", r"\1", repaired)
        if repaired != candidate:
            expanded.append(repaired)

    for candidate in expanded:
        try:
            payload = json.loads(candidate)
            if isinstance(payload, str):
                payload = json.loads(payload)
        except (json.JSONDecodeError, TypeError, ValueError):
            try:
                payload = ast.literal_eval(candidate)
            except (SyntaxError, ValueError, TypeError):
                continue
        if isinstance(payload, dict):
            return payload
    return None


def sanitize_claim_text(value: Any) -> str:
    """清理自然语言结论，并忽略模型自行生成的 [S编号]。"""
    text = normalize_display_text(value)
    text = re.sub(r"\[S\d+\]", "", text, flags=re.IGNORECASE)
    text = re.sub(r"^[-*•]\s*", "", text).strip()
    return text


def normalize_section(value: Any) -> str:
    section = normalize_display_text(value)
    section = _SECTION_ALIASES.get(section, section)
    return section if section in _ALLOWED_SECTIONS else "主要结论"


def parse_claim_output(output: Any) -> list[dict[str, Any]]:
    """解析模型返回的 claim + chunk_id + evidence_quote JSON。"""
    payload = _extract_json_object(message_content_to_text(output))
    if payload is None:
        return []

    raw_claims = payload.get("claims", [])
    if not isinstance(raw_claims, list):
        return []

    parsed: list[dict[str, Any]] = []
    for raw_claim in raw_claims[:MAX_CLAIMS]:
        if not isinstance(raw_claim, dict):
            continue

        text = sanitize_claim_text(raw_claim.get("text"))
        if not text or len(text) > MAX_CLAIM_LENGTH:
            continue

        raw_supports = raw_claim.get("supports", [])
        if (
            not isinstance(raw_supports, list)
            or len(raw_supports) != MAX_SUPPORTS_PER_CLAIM
        ):
            # 不静默截断多个 support。复合 Claim 必须由模型拆成
            # 多条原子结论，否则整条丢弃并在同证据重试时重新生成。
            continue

        supports: list[dict[str, Any]] = []
        seen_supports: set[tuple[int, str]] = set()
        for raw_support in raw_supports:
            if not isinstance(raw_support, dict):
                continue
            chunk_id = normalize_chunk_id(raw_support.get("chunk_id"))
            quote = normalize_display_text(raw_support.get("evidence_quote"))
            if chunk_id is None or not quote:
                continue
            pair = (chunk_id, quote)
            if pair in seen_supports:
                continue
            seen_supports.add(pair)
            supports.append(
                {
                    "chunk_id": chunk_id,
                    "evidence_quote": quote,
                }
            )

        if not supports:
            continue

        parsed.append(
            {
                "section": normalize_section(raw_claim.get("section")),
                "text": text,
                "supports": supports,
            }
        )

    return parsed


def build_evidence_index(
    evidence: list[dict[str, Any]],
) -> dict[int, dict[str, Any]]:
    """按稳定 chunk_id 建立证据索引。"""
    index: dict[int, dict[str, Any]] = {}
    for item in evidence:
        chunk_id = normalize_chunk_id(item.get("chunk_id"))
        if chunk_id is None or chunk_id in index:
            continue
        reasoning_text = clean_reasoning_evidence(get_evidence_text(item))
        if not reasoning_text:
            continue
        index[chunk_id] = {
            **item,
            "chunk_id": chunk_id,
            "reasoning_text": reasoning_text,
        }
    return index


def quote_is_in_chunk(
    quote: Any,
    chunk_text: Any,
) -> bool:
    """检查 quote 是否为 Chunk 中忽略空白后的连续原文。"""
    normalized_quote = normalize_exact_text(quote)
    normalized_chunk = normalize_exact_text(chunk_text)
    if not normalized_quote or not normalized_chunk:
        return False
    if not (MIN_QUOTE_LENGTH <= len(normalized_quote) <= MAX_QUOTE_LENGTH):
        return False
    if "..." in str(quote) or "……" in str(quote):
        return False
    return normalized_quote in normalized_chunk


def _non_whitespace_projection(text: str) -> tuple[str, list[int]]:
    chars: list[str] = []
    positions: list[int] = []
    for index, char in enumerate(text):
        if char.isspace():
            continue
        chars.append(char)
        positions.append(index)
    return "".join(chars), positions


def extract_quote_context(
    chunk_text: str,
    quote: str,
    *,
    side_chars: int = 80,
    max_chars: int = 240,
) -> str:
    """从 Chunk 中自动截取原文锚点前后的上下文。"""
    projected_chunk, positions = _non_whitespace_projection(chunk_text)
    projected_quote = normalize_exact_text(quote)
    start_projected = projected_chunk.find(projected_quote)
    if start_projected < 0 or not positions:
        return normalize_display_text(quote)

    end_projected = start_projected + len(projected_quote) - 1
    start_original = positions[start_projected]
    end_original = positions[end_projected] + 1

    rough_start = max(0, start_original - side_chars)
    rough_end = min(len(chunk_text), end_original + side_chars)

    left_boundaries = [
        chunk_text.rfind(mark, rough_start, start_original)
        for mark in ("。", "！", "？", "；", "\n")
    ]
    valid_left = [position for position in left_boundaries if position >= 0]
    context_start = max(valid_left) + 1 if valid_left else rough_start

    right_positions = []
    for mark in ("。", "！", "？", "；", "\n"):
        position = chunk_text.find(mark, end_original, rough_end)
        if position >= 0:
            right_positions.append(position + 1)
    context_end = min(right_positions) if right_positions else rough_end

    context = normalize_display_text(chunk_text[context_start:context_end])
    if len(context) > max_chars:
        context = normalize_display_text(
            chunk_text[rough_start:rough_end]
        )[:max_chars]
    return context


def _normalize_number_token(token: str) -> tuple[Decimal, str] | None:
    suffix = "%" if token.endswith("%") else ""
    raw = token[:-1] if suffix else token
    try:
        return Decimal(raw).normalize(), suffix
    except InvalidOperation:
        return None


def extract_numbers(text: Any) -> set[tuple[Decimal, str]]:
    """提取并规范化数值，用于防止模型添加证据中不存在的数字。"""
    result: set[tuple[Decimal, str]] = set()
    for token in re.findall(r"(?<!\d)[-+]?\d+(?:\.\d+)?%?", str(text or "")):
        normalized = _normalize_number_token(token)
        if normalized is not None:
            result.add(normalized)
    return result


def extract_query_entities(query: Any) -> list[str]:
    """提取比较/问答中的英文方法名，用于最低限度的对象覆盖检查。"""
    candidates = re.findall(
        r"(?<![A-Za-z0-9-])([A-Za-z][A-Za-z0-9-]{1,})(?![A-Za-z0-9-])",
        str(query or ""),
    )
    ignored = {"FL", "PFL", "IID", "NON-IID"}
    result: list[str] = []
    seen: set[str] = set()
    for candidate in candidates:
        normalized = candidate.upper()
        if normalized in ignored or normalized in seen:
            continue
        seen.add(normalized)
        result.append(candidate)
    return result[:4]


_NUMERIC_DIRECTION_WORDS = {
    "higher": ("高于", "大于", "多于", "超过"),
    "lower": ("低于", "小于", "少于", "不超过"),
}


def _entity_number_bindings(
    text: Any,
    entities: list[str],
) -> dict[str, set[tuple[Decimal, str]]]:
    """按实体附近片段提取方法—数值绑定。

    只服务于表格/数值 Claim 的保守校验。每个实体从自身位置读取到
    下一个目标实体或句级边界，避免只检查“数字是否在整个证据里”。
    """
    source = normalize_display_text(text)
    if not source or not entities:
        return {}

    occurrences: list[tuple[int, int, str]] = []
    for entity in entities:
        pattern = re.compile(re.escape(entity), flags=re.IGNORECASE)
        for match in pattern.finditer(source):
            occurrences.append((match.start(), match.end(), entity.upper()))
    occurrences.sort(key=lambda item: item[0])

    result: dict[str, set[tuple[Decimal, str]]] = {}
    for index, (_, entity_end, normalized_entity) in enumerate(occurrences):
        next_entity_start = (
            occurrences[index + 1][0]
            if index + 1 < len(occurrences)
            else len(source)
        )
        boundary_positions = [
            position
            for mark in ("。", "！", "？", "；", ";", "\n")
            if (position := source.find(mark, entity_end, next_entity_start)) >= 0
        ]
        segment_end = min(boundary_positions) if boundary_positions else next_entity_start
        segment = source[entity_end:segment_end]
        numbers = extract_numbers(segment)
        if numbers:
            result.setdefault(normalized_entity, set()).update(numbers)
    return result


def _direction_between_entities(
    text: str,
    left: str,
    right: str,
) -> str | None:
    """识别“方法A高于/低于方法B”这一类显式方向关系。"""
    compact = normalize_display_text(text)
    for direction, words in _NUMERIC_DIRECTION_WORDS.items():
        for word in words:
            pattern = (
                re.escape(left)
                + r".{0,40}?"
                + re.escape(word)
                + r".{0,20}?"
                + re.escape(right)
            )
            if re.search(pattern, compact, flags=re.IGNORECASE):
                return direction
    return None


def classify_numeric_entity_bindings(
    *,
    claim_text: str,
    evidence_context: str,
    query: str,
    comparison_targets: list[str] | None = None,
) -> tuple[str, str]:
    """确定性判断表格数值或显式大小关系。

    返回：
    - verified：Python 已完整验证，后续不再交给 NLI；
    - invalid：Python 已确认错误或无法安全验证；
    - not_applicable：普通文本 Claim，继续交给 NLI。
    """
    _ = query

    targets: list[str] = []
    seen_targets: set[str] = set()

    for raw_target in comparison_targets or []:
        target = normalize_display_text(
            raw_target
        )
        normalized = target.upper()

        if (
            not target
            or normalized in seen_targets
        ):
            continue

        if not re.search(
            re.escape(target),
            evidence_context,
            flags=re.IGNORECASE,
        ):
            continue

        seen_targets.add(normalized)
        targets.append(target)

    # 只有证据上下文同时包含两个明确比较对象时，
    # 才执行确定性表格比较。
    if len(targets) != 2:
        return "not_applicable", ""

    evidence_map = _entity_number_bindings(
        evidence_context,
        targets,
    )

    claim_numbers = extract_numbers(
        claim_text
    )

    left_target, right_target = targets

    detected_directions: list[
        tuple[str, str, str]
    ] = []

    for left, right in (
        (left_target, right_target),
        (right_target, left_target),
    ):
        direction = _direction_between_entities(
            claim_text,
            left,
            right,
        )

        if direction is not None:
            detected_directions.append(
                (
                    left,
                    right,
                    direction,
                )
            )

    # 没有具体数字，也没有明确高低关系，
    # 属于普通文本 Claim。
    if (
        not claim_numbers
        and not detected_directions
    ):
        return "not_applicable", ""

    # Claim 写出具体数字时，验证数字归属。
    if claim_numbers:
        claim_map = _entity_number_bindings(
            claim_text,
            targets,
        )

        found_bound_number = False

        for target in targets:
            entity_key = target.upper()

            numbers = claim_map.get(
                entity_key,
                set(),
            )

            if not numbers:
                continue

            found_bound_number = True

            evidence_numbers = evidence_map.get(
                entity_key,
                set(),
            )

            if not evidence_numbers:
                return (
                    "invalid",
                    "证据中无法确定"
                    f"{target}对应的数值。",
                )

            if not numbers.issubset(
                evidence_numbers
            ):
                return (
                    "invalid",
                    "结论中的数值与"
                    f"{target}在证据中的数值归属不一致。",
                )

        # Claim 明明包含数字，但数字无法绑定到任何比较对象，
        # 不交给通用 NLI 猜测。
        if not found_bound_number:
            return (
                "invalid",
                "结论包含数值，但无法将数值"
                "确定性绑定到比较对象。",
            )

    # 检查“高于、低于、多于、少于”等明确关系。
    for left, right, direction in detected_directions:
        left_values = sorted(
            value
            for value, suffix in evidence_map.get(
                left.upper(),
                set(),
            )
            if not suffix
        )

        right_values = sorted(
            value
            for value, suffix in evidence_map.get(
                right.upper(),
                set(),
            )
            if not suffix
        )

        if (
            not left_values
            or not right_values
            or len(left_values) != len(
                right_values
            )
        ):
            return (
                "invalid",
                "结论包含明确大小关系，但无法从证据中"
                f"形成{left}/{right}的可比较数值序列。",
            )

        if (
            direction == "higher"
            and not all(
                left_value > right_value
                for left_value, right_value in zip(
                    left_values,
                    right_values,
                )
            )
        ):
            return (
                "invalid",
                "结论中的大小关系与"
                f"{left}/{right}的证据数值不一致。",
            )

        if (
            direction == "lower"
            and not all(
                left_value < right_value
                for left_value, right_value in zip(
                    left_values,
                    right_values,
                )
            )
        ):
            return (
                "invalid",
                "结论中的大小关系与"
                f"{left}/{right}的证据数值不一致。",
            )

    return "verified", ""


def validate_numeric_entity_bindings(
    *,
    claim_text: str,
    evidence_context: str,
    query: str,
    comparison_targets: list[str] | None = None,
) -> str | None:
    """保留旧接口，避免已有测试或调用代码失效。"""
    status, reason = (
        classify_numeric_entity_bindings(
            claim_text=claim_text,
            evidence_context=evidence_context,
            query=query,
            comparison_targets=comparison_targets,
        )
    )

    if status == "invalid":
        return reason

    return None


def validate_claims(
    *,
    query: str,
    query_type: str,
    evidence: list[dict[str, Any]],
    claims: list[dict[str, Any]],
    nli_model: NLIProtocol,
    comparison_targets: list[str] | None = None,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], str]:
    """逐条执行 Chunk、原文锚点、数值和本地 NLI 校验。"""
    evidence_index = build_evidence_index(evidence)
    validation_targets = [
        normalize_display_text(target)
        for target in (
            comparison_targets or []
        )
        if normalize_display_text(target)
    ]

    # 兼容没有显式传入 comparison_targets 的旧测试。
    if (
        query_type == "comparison"
        and not validation_targets
    ):
        validation_targets = (
            extract_query_entities(
                query
            )[:2]
        )
    preliminarily_valid: list[dict[str, Any]] = []
    rejected: list[dict[str, Any]] = []

    for claim_index, claim in enumerate(claims, start=1):
        text = sanitize_claim_text(claim.get("text"))
        supports = claim.get("supports", [])
        prepared_supports: list[dict[str, Any]] = []
        rejection_reason = ""
        numeric_status = "not_applicable"

        if not text:
            rejection_reason = "结论文本为空。"
        elif not isinstance(supports, list) or not supports:
            rejection_reason = "结论没有提供证据锚点。"
        elif len(supports) != 1:
            rejection_reason = (
                "每条原子结论必须且只能绑定一个证据 Chunk；"
                "需要多条证据时请拆成多条结论。"
            )

        if not rejection_reason:
            for support in supports:
                chunk_id = normalize_chunk_id(support.get("chunk_id"))
                quote = normalize_display_text(support.get("evidence_quote"))
                chunk = evidence_index.get(chunk_id or -1)

                if chunk_id is None or chunk is None:
                    rejection_reason = "引用的 chunk_id 不存在于本轮证据。"
                    break
                if not quote_is_in_chunk(quote, chunk["reasoning_text"]):
                    rejection_reason = "evidence_quote 不是指定 Chunk 中的连续原文。"
                    break

                context = extract_quote_context(
                    chunk["reasoning_text"],
                    quote,
                )
                prepared_supports.append(
                    {
                        "chunk_id": chunk_id,
                        "evidence_quote": quote,
                        "evidence_context": context,
                    }
                )

        if not rejection_reason:
            combined_context = "\n".join(
                support["evidence_context"]
                for support in prepared_supports
            )
            claim_numbers = extract_numbers(text)
            evidence_numbers = extract_numbers(combined_context)
            missing_numbers = claim_numbers - evidence_numbers
            if missing_numbers:
                rejection_reason = "结论包含证据上下文中不存在的数值。"
            else:
                (
                    numeric_status,
                    numeric_reason,
                ) = classify_numeric_entity_bindings(
                    claim_text=text,
                    evidence_context=combined_context,
                    query=query,
                    comparison_targets=validation_targets,
                )

                if numeric_status == "invalid":
                    rejection_reason = (
                        numeric_reason
                    )
                if (
                        not rejection_reason
                        and query_type == "comparison"
                        and validation_targets
                ):
                    mentioned_targets = [
                        target
                        for target in validation_targets
                        if re.search(
                            re.escape(target),
                            text,
                            flags=re.IGNORECASE,
                        )
                    ]

                    # 表格数值结论允许同时出现双方；
                    # 普通文本比较必须拆成单对象原子 Claim。
                    if (
                            numeric_status != "verified"
                            and len(mentioned_targets) > 1
                    ):
                        rejection_reason = (
                            "非数值比较结论不能在同一条 Claim 中"
                            "同时陈述多个方法；请拆成单对象原子结论。"
                        )

                    elif len(mentioned_targets) == 1:
                        target = mentioned_targets[0]

                        if not re.search(
                                re.escape(target),
                                combined_context,
                                flags=re.IGNORECASE,
                        ):
                            rejection_reason = (
                                "结论主体"
                                f"{target}"
                                "没有在对应证据上下文中明确出现。"
                            )

        if rejection_reason:
            rejected.append(
                {
                    **claim,
                    "claim_index": claim_index,
                    "rejection_reason": rejection_reason,
                }
            )
            continue

        preliminarily_valid.append(
            {
                "section": normalize_section(claim.get("section")),
                "text": text,
                "supports": prepared_supports,
                "combined_context": combined_context,
                "claim_index": claim_index,
                "numeric_status": numeric_status,
            }
        )

    accepted: list[dict[str, Any]] = []
    nli_pending: list[dict[str, Any]] = []

    for claim in preliminarily_valid:
        if (
                claim.get("numeric_status")
                == "verified"
        ):
            # 数值归属或大小关系已经由 Python
            # 完整验证，不再允许通用 NLI 错误否决。
            accepted.append(
                {
                    **claim,
                    "validation_mode": (
                        "python_numeric"
                    ),
                    "nli": {
                        "label": (
                            "python_numeric_verified"
                        ),
                        "entailment": 1.0,
                        "neutral": 0.0,
                        "contradiction": 0.0,
                    },
                }
            )
        else:
            nli_pending.append(
                claim
            )

    nli_pairs = [
        (
            claim["combined_context"],
            claim["text"],
        )
        for claim in nli_pending
    ]

    nli_results = (
        nli_model.predict_many(
            nli_pairs
        )
        if nli_pairs
        else []
    )

    for index, claim in enumerate(
            nli_pending
    ):
        if index >= len(nli_results):
            rejected.append(
                {
                    **claim,
                    "rejection_reason": (
                        "本地 NLI 返回数量与"
                        "待校验结论数量不一致。"
                    ),
                }
            )
            continue

        nli_result = nli_results[index]

        if not nli_model.accepts(
                nli_result
        ):
            rejected.append(
                {
                    **claim,
                    "nli": {
                        "label": nli_result.label,
                        "entailment": (
                            nli_result.entailment
                        ),
                        "neutral": (
                            nli_result.neutral
                        ),
                        "contradiction": (
                            nli_result.contradiction
                        ),
                    },
                    "rejection_reason": (
                        "本地 NLI 未判定为蕴含"
                        f"（{nli_result.label}, "
                        "entailment="
                        f"{nli_result.entailment:.3f}）。"
                    ),
                }
            )
            continue

        accepted.append(
            {
                **claim,
                "validation_mode": "nli",
                "nli": {
                    "label": nli_result.label,
                    "entailment": (
                        nli_result.entailment
                    ),
                    "neutral": (
                        nli_result.neutral
                    ),
                    "contradiction": (
                        nli_result.contradiction
                    ),
                },
            }
        )

    # Python 验证和 NLI 验证的 Claim
    # 按模型原始输出顺序重新排列。
    accepted.sort(
        key=lambda item: item.get(
            "claim_index",
            0,
        )
    )

    entities = validation_targets

    combined_claims = " ".join(
        item["text"]
        for item in accepted
    ).lower()

    if query_type == "comparison" and entities:
        missing = [
            entity
            for entity in entities
            if entity.lower() not in combined_claims
        ]
        if missing:
            rejected.extend(
                {
                    **item,
                    "rejection_reason": (
                        "比较回答未同时覆盖核心对象："
                        + "、".join(missing)
                        + "。"
                    ),
                }
                for item in accepted
            )
            accepted = []
    elif entities and accepted:
        if not any(entity.lower() in combined_claims for entity in entities):
            rejected.extend(
                {
                    **item,
                    "rejection_reason": "回答没有覆盖问题中的核心对象。",
                }
                for item in accepted
            )
            accepted = []

    reason = (
        f"Claim 校验完成：通过 {len(accepted)} 条，"
        f"拒绝 {len(rejected)} 条。"
    )
    if rejected:
        counts = Counter(
            str(item.get("rejection_reason") or "未知原因")
            for item in rejected
        )
        details = "；".join(
            f"{message}：{count}条"
            for message, count in counts.most_common(4)
        )
        reason += f" 主要失败原因：{details}。"
    return accepted, rejected, reason


def assign_citations_and_render(
    *,
    accepted_claims: list[dict[str, Any]],
    evidence: list[dict[str, Any]],
) -> tuple[str, dict[int, str], list[dict[str, Any]], list[dict[str, Any]]]:
    """按最终使用的 Chunk 首次出现顺序分配 S 编号并渲染答案。"""
    evidence_index = build_evidence_index(evidence)
    citation_map: dict[int, str] = {}
    next_number = 1
    rendered_claims: list[dict[str, Any]] = []

    for claim in accepted_claims:
        citation_ids: list[str] = []
        seen_chunks: set[int] = set()
        for support in claim.get("supports", []):
            chunk_id = normalize_chunk_id(support.get("chunk_id"))
            if chunk_id is None or chunk_id in seen_chunks:
                continue
            seen_chunks.add(chunk_id)
            if chunk_id not in citation_map:
                citation_map[chunk_id] = f"S{next_number}"
                next_number += 1
            citation_ids.append(citation_map[chunk_id])

        if not citation_ids:
            continue
        rendered_claims.append(
            {
                **claim,
                "citation_ids": citation_ids,
            }
        )

    sections: OrderedDict[str, list[dict[str, Any]]] = OrderedDict()
    for claim in rendered_claims:
        sections.setdefault(claim["section"], []).append(claim)

    lines: list[str] = []
    for section, claims in sections.items():
        if lines:
            lines.append("")
        lines.append(f"### {section}")
        for claim in claims:
            citations = "".join(
                f"[{citation_id}]"
                for citation_id in claim["citation_ids"]
            )
            lines.append(f"- {claim['text']} {citations}")

    cited_evidence: list[dict[str, Any]] = []
    for chunk_id, citation_id in citation_map.items():
        item = evidence_index.get(chunk_id)
        if item is None:
            continue
        quotes = []
        for claim in rendered_claims:
            for support in claim.get("supports", []):
                if support.get("chunk_id") == chunk_id:
                    quote = support.get("evidence_quote")
                    if quote and quote not in quotes:
                        quotes.append(quote)
        cited_evidence.append(
            {
                **item,
                "citation_id": citation_id,
                "used_quotes": quotes,
                # 页面展示与实际 NLI/锚点校验使用同一份清理后证据，
                # 避免把已剔除的相邻表格尾部重新展示给用户。
                "content": item["reasoning_text"],
            }
        )

    return (
        "\n".join(lines).strip(),
        citation_map,
        rendered_claims,
        cited_evidence,
    )
