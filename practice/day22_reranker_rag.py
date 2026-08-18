import json
import os
import re
from pathlib import Path
import httpx
from openai import OpenAI
import torch

from transformers import (
    AutoModelForSequenceClassification,
    AutoTokenizer
)
#后面给每次问答记录保存时间
from datetime import datetime
# Day 16：Embedding、FAISS 和检索
from day16_faiss_retrieval import (
    MODEL_NAME as EMBEDDING_MODEL_NAME,
    load_embedding_model,
    load_or_build_faiss_index,
    retrieve_top_chunks,
)

# Day 19：完整 PDF 预处理流水线
from day19_full_retrieval_pipeline import (
    FORCE_REBUILD,
    PDF_FILENAME,
    display_pipeline_summary,
    get_project_root,
    load_or_build_chunks,
)


# 负责阅读检索证据并生成答案的大语言模型
ARK_BASE_URL = os.getenv(
    "ARK_BASE_URL",
    "https://ark.cn-beijing.volces.com/api/v3"
).strip()

LLM_MODEL = os.getenv(
    "ARK_MODEL_ID",
    ""
).strip()

# 给回答和内部推理预留足够的输出空间。
LLM_MAX_OUTPUT_TOKENS = 1200
# 单个 Chunk 成为候选证据的最低相似度
CHUNK_MIN_SIMILARITY = 0.30

# 整个问题是否应该拒答的最低要求
QUERY_REFUSAL_THRESHOLD = 0.36

# 最终最多发送给大模型的证据数量
FINAL_EVIDENCE_COUNT = 5

# 识别 [S1]、[S2] 等引用编号
CITATION_PATTERN = r"\[S(\d+)\]"

# 先在本地 FAISS 中多取一些候选，再过滤、重排。
# 这一步完全在本地完成，不消耗火山方舟 Token。
RETRIEVAL_CANDIDATE_K = 20

RERANKER_MODEL_NAME = (
    "BAAI/bge-reranker-v2-m3"
)

RERANKER_MAX_LENGTH = 512

RERANKER_BATCH_SIZE = 2



SYSTEM_INSTRUCTIONS = """
你是一个严格基于论文证据回答问题的学术助手。

回答时必须遵守以下规则：

1. 只能依据提供的证据回答，不得补充证据中未出现的信息。
2. 如果证据不足，必须明确说明“根据当前证据无法确定”。
3. 必须区分以下概念：
   - 数据集；
   - 评价指标；
   - 实验设置；
   - 统计汇报方式；
   - 表格展示规则。
4. “独立运行次数”“均值±标准差”属于实验或统计汇报方式，
   不应直接归类为评价指标。
5. “最优结果加粗”属于表格展示规则，不属于评价指标。
6. 不同章节使用的数据集不完全相同时，应分别说明。
7. 回答应简洁、准确，并标明对应证据页码。
""".strip()


def create_openai_client():
    """
    创建火山方舟 OpenAI 兼容客户端。
    """
    api_key = os.getenv(
        "ARK_API_KEY",
        ""
    ).strip()

    if not api_key:
        raise RuntimeError(
            "没有检测到 ARK_API_KEY。"
        )

    model_id = os.getenv(
        "ARK_MODEL_ID",
        ""
    ).strip()

    if not model_id:
        raise RuntimeError(
            "没有检测到 ARK_MODEL_ID。"
        )

    direct_http_client = httpx.Client(
        trust_env=False,
        timeout=httpx.Timeout(
            60.0,
            connect=20.0
        )
    )

    client = OpenAI(
        api_key=api_key,
        base_url=ARK_BASE_URL,
        http_client=direct_http_client,
        max_retries=0
    )

    return client


def load_reranker():
    """
    职责：
        加载 BGE Cross-Encoder Reranker。

    输出：
        tokenizer：负责把 query–Chunk 转成模型输入
        model：Reranker 模型
        device：cuda 或 cpu
    """
    device = (
        "cuda"
        if torch.cuda.is_available()
        else "cpu"
    )

    print("\n===== 加载语义 Reranker =====")
    print(f"模型：{RERANKER_MODEL_NAME}")
    print(f"设备：{device}")

    tokenizer = AutoTokenizer.from_pretrained(
        RERANKER_MODEL_NAME
    )

    model = (
        AutoModelForSequenceClassification
        .from_pretrained(
            RERANKER_MODEL_NAME
        )
    )

    model.to(device)

    if device == "cuda":
        model = model.half()

    model.eval()

    return tokenizer, model, device


def rerank_candidates(
    query,
    candidates,
    tokenizer,
    model,
    device
):
    """
    使用 Cross-Encoder 对 FAISS 候选重新评分。

    每条结果保留：
        similarity：
            FAISS 原始相似度

        faiss_rank：
            FAISS 原始排名

        reranker_score：
            Cross-Encoder 相关性分数

        reranker_rank：
            Reranker 新排名

        rank_change：
            faiss_rank - reranker_rank；
            正数代表排名提升，负数代表排名下降
    """
    if not candidates:
        return []

    prepared_candidates = []

    # 在重排前保存 FAISS 原始排名。
    for faiss_rank, candidate in enumerate(
        candidates,
        start=1
    ):
        prepared_candidate = dict(candidate)
        prepared_candidate["faiss_rank"] = (
            faiss_rank
        )
        prepared_candidates.append(
            prepared_candidate
        )

    reranked_results = []

    # 分批评分，降低 4GB 显存的峰值占用。
    for batch_start in range(
        0,
        len(prepared_candidates),
        RERANKER_BATCH_SIZE
    ):
        batch_candidates = prepared_candidates[
            batch_start:
            batch_start + RERANKER_BATCH_SIZE
        ]

        query_passage_pairs = [
            [
                query,
                str(
                    candidate.get(
                        "content",
                        ""
                    )
                )
            ]
            for candidate in batch_candidates
        ]

        inputs = tokenizer(
            query_passage_pairs,
            padding=True,
            truncation=True,
            max_length=RERANKER_MAX_LENGTH,
            return_tensors="pt"
        )

        inputs = {
            name: tensor.to(device)
            for name, tensor in inputs.items()
        }

        with torch.inference_mode():
            logits = model(
                **inputs,
                return_dict=True
            ).logits.view(-1)

        scores = (
            logits
            .float()
            .cpu()
            .tolist()
        )

        for candidate, score in zip(
            batch_candidates,
            scores
        ):
            reranked_candidate = dict(
                candidate
            )

            reranked_candidate[
                "reranker_score"
            ] = round(
                float(score),
                4
            )

            reranked_results.append(
                reranked_candidate
            )

    reranked_results.sort(
        key=lambda result: result[
            "reranker_score"
        ],
        reverse=True
    )

    # 保存重排后的名次及名次变化。
    for reranker_rank, result in enumerate(
        reranked_results,
        start=1
    ):
        result["reranker_rank"] = (
            reranker_rank
        )

        result["rank_change"] = (
            result["faiss_rank"]
            - reranker_rank
        )

    return reranked_results


def prepare_cited_evidence(results):
    """
    职责：
        为最终证据增加稳定引用编号。

    输入：
        results：过滤和重排后的证据列表

    输出：
        cited_results：带有 citation_id 的新列表

    示例：
        第一条证据 → S1
        第二条证据 → S2
    """
    cited_results = []

    for rank, result in enumerate(
        results,
        start=1
    ):
        cited_result = result.copy()

        cited_result["citation_id"] = (
            f"S{rank}"
        )

        cited_results.append(
            cited_result
        )

    return cited_results


def evaluate_evidence_sufficiency(
    results,
    refusal_threshold
):
    """
    职责：
        判断当前问题是否获得了
        足够可靠的论文证据。

    输入：
        results：
            过滤和重排后的证据列表

        refusal_threshold：
            问题级拒答阈值

    输出：
        evidence_status：
            是否拒答、拒答原因、
            最高向量相似度
    """
    if not results:
        return {
            "should_refuse": True,
            "reason": (
                "没有检索到任何候选证据"
            ),
            "top_similarity": None
        }

    top_similarity = max(
        float(result["similarity"])
        for result in results
    )

    if top_similarity < refusal_threshold:
        return {
            "should_refuse": True,
            "reason": (
                "最高相似度低于"
                "问题级拒答阈值"
            ),
            "top_similarity": (
                top_similarity
            )
        }

    return {
        "should_refuse": False,
        "reason": (
            "存在达到问题级要求的候选证据"
        ),
        "top_similarity": (
            top_similarity
        )
    }

def is_noise_chunk(result):
    """
    判断一个 Chunk 是否主要属于参考文献、目录等检索噪声。

    这里只过滤特征非常明显的内容，避免误删普通正文。
    """
    content = result.get("content", "")
    normalized = content.replace(" ", "")

    if not normalized:
        return True

    # 参考文献章节标题。
    if "参考文献" in normalized[:120]:
        return True

    # 目录页通常包含“目录/目 录”和大量点线、页码。
    if (
        ("目录" in normalized[:120] or "目录" in content[:150])
        and (
            content.count("...") >= 2
            or content.count("……") >= 2
            or len(re.findall(r"\.{5,}", content)) >= 2
        )
    ):
        return True

    # 参考文献条目常见特征：多个 [数字] 编号，并伴随期刊/会议标记。
    citation_count = len(
        re.findall(r"\[\d+\]", content)
    )

    reference_signals = [
        "[J]",
        "[C]",
        "[M]",
        "[D]",
        "IEEE",
        "Springer",
        "PMLR",
        "Proceedings",
        "doi:",
        "DOI:"
    ]

    reference_signal_count = sum(
        signal in content
        for signal in reference_signals
    )

    if (
        citation_count >= 2
        and reference_signal_count >= 1
    ):
        return True

    # 多行都以 [数字] 开头，通常也是参考文献列表。
    numbered_reference_lines = len(
        re.findall(
            r"(?m)^\s*\[\d+\]",
            content
        )
    )

    if numbered_reference_lines >= 2:
        return True

    return False



def select_reranked_evidence(
    reranked_results
):
    """
    对 Reranker 排序后的结果执行最终整理：

    1. 删除明显目录、参考文献噪声；
    2. 删除重复 Chunk；
    3. 同一页最多保留两个 Chunk；
    4. 最终返回前五条证据。

    注意：
        此处不再计算关键词 bonus，
        排序完全来自语义 Reranker。
    """
    selected_results = []
    seen_hashes = set()
    page_counts = {}

    for result in reranked_results:
        if is_noise_chunk(result):
            continue

        content_hash = result.get(
            "content_sha256"
        )

        if (
            content_hash
            and content_hash in seen_hashes
        ):
            continue

        page_number = result.get(
            "page_number"
        )

        current_page_count = page_counts.get(
            page_number,
            0
        )

        if current_page_count >= 2:
            continue

        selected_results.append(
            dict(result)
        )

        if content_hash:
            seen_hashes.add(
                content_hash
            )

        page_counts[page_number] = (
            current_page_count + 1
        )

        if (
            len(selected_results)
            >= FINAL_EVIDENCE_COUNT
        ):
            break

    return selected_results

def build_evidence_context(results):
    """
    职责：
        将带引用编号的证据
        组织成 Prompt 文本。
    """
    evidence_parts = []

    for result in results:
        evidence_text = (
            f"[{result['citation_id']}]\n"
            f"来源文件：{result['source_file']}\n"
            f"来源页码：第 "
            f"{result['page_number']} 页\n"
            f"Chunk ID：{result['chunk_id']}\n"
            f"语义相似度："
            f"{result['similarity']:.4f}\n"
            f"证据正文：\n"
            f"{result['content']}"
        )

        evidence_parts.append(
            evidence_text
        )

    return "\n\n".join(
        evidence_parts
    )

def build_rag_prompt(
    query,
    evidence_context
):
    """
    职责：
        构造要求逐项引用证据的
        RAG Prompt。
    """
    prompt = f"""
请严格根据下面的论文证据回答问题。

用户问题：
{query}

论文证据：
{evidence_context}

回答规则：

1. 只能使用上述证据中的信息。
2. 每个主要事实或结论后必须添加引用，例如 [S1] 或 [S1][S2]。
3. 引用编号必须来自已经提供的证据，不得创造不存在的编号。
4. 不要仅因为证据中出现相关关键词，就推断论文没有明确表达的结论。
5. 数据集、评价指标、实验设置、统计汇报方式和表格展示规则必须严格区分。
6. 多条证据内容重复时，应合并表达，不要重复罗列。
7. 如果证据只能回答问题的一部分，应明确指出无法确定的部分。
8. 不要输出参考文献列表，只在相关句子后标注引用编号。
9. 不要提及检索、向量、相似度或 Prompt。
""".strip()

    return prompt


def normalize_citation_whitespace(answer):
    """
    只修复引用编号内部的异常空格或换行。

    示例：
        [S   1]  -> [S1]
        [\nS3]  -> [S3]

    不改变正文内容，也不创造新的引用编号。
    """
    return re.sub(
        r"\[\s*S\s*(\d+)\s*\]",
        r"[S\1]",
        answer
    )


def validate_answer_citations(
    answer,
    evidence
):
    """
    职责：
        检查答案中的引用编号
        是否真实存在于证据中。

    检查：
        1. 答案是否至少包含一个引用
        2. 是否引用了不存在的证据编号

    输出：
        citation_validation：
            引用检查结果字典
    """
    cited_numbers = re.findall(
        CITATION_PATTERN,
        answer
    )

    cited_ids = {
        f"S{number}"
        for number in cited_numbers
    }

    valid_ids = {
        result["citation_id"]
        for result in evidence
    }

    invalid_ids = sorted(
        cited_ids - valid_ids
    )

    valid_used_ids = sorted(
        cited_ids & valid_ids
    )

    has_citations = bool(
        cited_ids
    )

    citation_valid = (
        has_citations
        and not invalid_ids
    )

    return {
        "citation_valid": citation_valid,
        "has_citations": has_citations,
        "used_citation_ids": (
            valid_used_ids
        ),
        "invalid_citation_ids": (
            invalid_ids
        ),
        "available_citation_ids": sorted(
            valid_ids
        )
    }


def generate_reliable_answer(
    client,
    query,
    results,
    refusal_results=None
):
    """
    职责：
        先判断证据是否充足，
        再决定是否调用大模型。

    输出：
        generation_data：
            包含答案、证据状态、
            引用检查等完整信息
    """
    evidence_check_results = (
        refusal_results
        if refusal_results is not None
        else results
    )

    evidence_status = (
        evaluate_evidence_sufficiency(
            results=evidence_check_results,
            refusal_threshold=(
                QUERY_REFUSAL_THRESHOLD
            )
        )
    )

    if evidence_status["should_refuse"]:
        answer = (
            "根据当前论文知识库，没有检索到"
            "足够可靠的证据，因此无法回答"
            "这个问题。"
        )

        return {
            "answer": answer,
            "llm_called": False,
            "response_id": None,
            "evidence": [],
            "evidence_status": (
                evidence_status
            ),
            "citation_validation": {
                "citation_valid": True,
                "has_citations": False,
                "used_citation_ids": [],
                "invalid_citation_ids": [],
                "available_citation_ids": []
            },
            "raw_citation_validation": {
                "citation_valid": True,
                "has_citations": False,
                "used_citation_ids": [],
                "invalid_citation_ids": [],
                "available_citation_ids": []
            },
            "citation_format_repaired": False
        }

    final_results = results[
        :FINAL_EVIDENCE_COUNT
    ]

    cited_evidence = (
        prepare_cited_evidence(
            final_results
        )
    )

    evidence_context = (
        build_evidence_context(
            cited_evidence
        )
    )

    prompt = build_rag_prompt(
        query=query,
        evidence_context=evidence_context
    )

    try:
        response = (
            client.responses.create(
                model=LLM_MODEL,
                instructions=(
                    SYSTEM_INSTRUCTIONS
                ),
                input=prompt,
                max_output_tokens=LLM_MAX_OUTPUT_TOKENS
            )
        )

    except Exception as error:
        raise RuntimeError(
            f"大模型 API 调用失败：{error}"
        ) from error

    answer = (
        getattr(
            response,
            "output_text",
            ""
        )
        or ""
    ).strip()

    if not answer:
        response_status = getattr(
            response,
            "status",
            None
        )

        incomplete_details = getattr(
            response,
            "incomplete_details",
            None
        )

        incomplete_reason = getattr(
            incomplete_details,
            "reason",
            None
        )

        output_items = (
            getattr(
                response,
                "output",
                None
            )
            or []
        )

        output_types = [
            getattr(
                item,
                "type",
                type(item).__name__
            )
            for item in output_items
        ]

        raise RuntimeError(
            "API 调用成功，但没有返回可见答案。"
            f" status={response_status}；"
            f"incomplete_reason="
            f"{incomplete_reason}；"
            f"output_types={output_types}；"
            f"max_output_tokens="
            f"{LLM_MAX_OUTPUT_TOKENS}。"
        )

    raw_answer = answer

    normalized_answer = (
        normalize_citation_whitespace(
            raw_answer
        )
    )

    citation_format_repaired = (
        normalized_answer != raw_answer
    )

    raw_citation_validation = (
        validate_answer_citations(
            answer=raw_answer,
            evidence=cited_evidence
        )
    )

    citation_validation = (
        validate_answer_citations(
            answer=normalized_answer,
            evidence=cited_evidence
        )
    )

    return {
        "answer": normalized_answer,
        "llm_called": True,
        "response_id": response.id,
        "evidence": cited_evidence,
        "evidence_status": (
            evidence_status
        ),
        "citation_validation": (
            citation_validation
        ),
        "raw_citation_validation": (
            raw_citation_validation
        ),
        "citation_format_repaired": (
            citation_format_repaired
        )
    }


def display_reliable_answer(
    query,
    generation_data
):
    """
    职责：
        显示答案、证据状态、
        证据来源和引用检查结果。
    """
    answer = generation_data["answer"]

    evidence = generation_data[
        "evidence"
    ]

    evidence_status = generation_data[
        "evidence_status"
    ]

    citation_validation = generation_data[
        "citation_validation"
    ]

    print("\n" + "=" * 70)
    print("Day 22 Reranker 论文问答")
    print("=" * 70)

    print(f"\n用户问题：\n{query}")

    print(f"\n回答：\n{answer}")

    print("\n证据状态：")

    print(
        f"是否拒答："
        f"{evidence_status['should_refuse']}"
    )

    print(
        f"判断原因："
        f"{evidence_status['reason']}"
    )

    top_similarity = evidence_status[
        "top_similarity"
    ]

    if top_similarity is not None:
        print(
            f"最高相似度："
            f"{top_similarity:.4f}"
        )

    if not evidence:
        print("\n本次未调用大语言模型。")
        return

    print("\n证据来源：")

    for result in evidence:
        print(
            f"[{result['citation_id']}] "
            f"{result['source_file']}，"
            f"第 {result['page_number']} 页，"
            f"Chunk {result['chunk_id']}，"
            f"相似度 "
            f"{result['similarity']:.4f}"
        )

    print("\n引用检查：")

    print(
        f"引用是否合法："
        f"{citation_validation['citation_valid']}"
    )

    citation_format_repaired = (
        generation_data.get(
            "citation_format_repaired",
            False
        )
    )

    print(
        "引用空白是否自动修复："
        f"{citation_format_repaired}"
    )

    print(
        f"实际使用引用："
        f"{citation_validation['used_citation_ids']}"
    )

    invalid_ids = citation_validation[
        "invalid_citation_ids"
    ]

    if invalid_ids:
        print(
            f"无效引用：{invalid_ids}"
        )

    if not citation_validation[
        "has_citations"
    ]:
        print(
            "警告：模型答案没有包含证据引用。"
        )


def append_qa_history(
    query,
    generation_data,
    index,
    output_file
):
    """
    职责：
        将本次问答追加到
        JSONL 历史文件。

    JSONL：
        每一行都是一个
        独立 JSON 对象。
    """
    history_record = {
        "timestamp": (
            datetime.now()
            .astimezone()
            .isoformat(
                timespec="seconds"
            )
        ),

        "query": query,

        "answer": (
            generation_data["answer"]
        ),

        "llm_called": (
            generation_data["llm_called"]
        ),

        "response_id": (
            generation_data["response_id"]
        ),

        "embedding_model": (
            EMBEDDING_MODEL_NAME
        ),
        "reranker_model": (
            RERANKER_MODEL_NAME
        ),
        "llm_model": LLM_MODEL,

        "llm_max_output_tokens": (
            LLM_MAX_OUTPUT_TOKENS
        ),

        "retrieval_strategy": (
            "BGE embedding + FAISS Top 20 + "
            "BGE Cross-Encoder reranking + Top 5"
        ),

        "retrieval_candidate_k": (
            RETRIEVAL_CANDIDATE_K
        ),

        "final_evidence_count": (
            FINAL_EVIDENCE_COUNT
        ),

        "index_type": "IndexFlatIP",

        "index_vector_count": int(
            index.ntotal
        ),

        "embedding_dimension": int(
            index.d
        ),

        "chunk_min_similarity": (
            CHUNK_MIN_SIMILARITY
        ),

        "query_refusal_threshold": (
            QUERY_REFUSAL_THRESHOLD
        ),

        "evidence_status": (
            generation_data[
                "evidence_status"
            ]
        ),

        "citation_validation": (
            generation_data[
                "citation_validation"
            ]
        ),

        "raw_citation_validation": (
            generation_data.get(
                "raw_citation_validation",
                generation_data[
                    "citation_validation"
                ]
            )
        ),

        "citation_format_repaired": (
            generation_data.get(
                "citation_format_repaired",
                False
            )
        ),

        "evidence_count": len(
            generation_data["evidence"]
        ),

        "evidence": (
            generation_data["evidence"]
        )
    }

    output_file.parent.mkdir(
        parents=True,
        exist_ok=True
    )

    with output_file.open(
        "a",#追加模式"w"覆盖写入原有内容会被清空"a"追加写入 新内容写在文件末尾
        encoding="utf-8"
    ) as file:
        json_line = json.dumps(
            history_record,
            ensure_ascii=False
        )

        file.write(
            json_line + "\n"
        )


def main():
    """
    Day 22 语义 Reranker RAG 流程：

    1. 加载 PDF Chunk 缓存
    2. 加载 Embedding 模型和 FAISS 索引
    3. 接收用户问题并召回候选证据
    4. 使用 BGE Cross-Encoder 对候选进行语义重排
    5. 判断问题级证据是否充足
    6. 为有效证据增加引用编号
    7. 调用大模型生成带引用答案
    8. 验证引用编号是否合法
    9. 将问答追加保存到 JSONL
    """
    project_root = get_project_root()

    pdf_file = (
        project_root
        / "data"
        / "papers"
        / PDF_FILENAME
    )

    pages_file = (
        project_root
        / "data"
        / "day19_pdf_pages.json"
    )
    qa_history_file = (
            project_root
            / "data"
            / "day22_reranker_qa_history.jsonl"
    )
    chunks_file = (
        project_root
        / "data"
        / "day19_pdf_chunks.json"
    )

    chunking_report_file = (
        project_root
        / "data"
        / "day19_chunking_report.json"
    )

    pipeline_metadata_file = (
        project_root
        / "data"
        / "day19_pipeline_metadata.json"
    )

    faiss_index_file = (
        project_root
        / "data"
        / "day19_faiss.index"
    )

    faiss_metadata_file = (
        project_root
        / "data"
        / "day19_faiss_metadata.json"
    )



    if not pdf_file.exists():
        raise FileNotFoundError(
            f"没有找到 PDF 文件：{pdf_file}"
        )

    # 第一阶段：加载 PDF Chunk 缓存
    chunks, report, chunk_status = (
        load_or_build_chunks(
            pdf_file=pdf_file,
            pages_file=pages_file,
            chunks_file=chunks_file,
            chunking_report_file=(
                chunking_report_file
            ),
            pipeline_metadata_file=(
                pipeline_metadata_file
            ),
            force_rebuild=FORCE_REBUILD
        )
    )

    # 第二阶段：加载查询 Embedding 模型
    embedding_model = (
        load_embedding_model(
            EMBEDDING_MODEL_NAME
        )
    )

    # 第三阶段：加载 Day 19 FAISS 索引
    index, index_status = (
        load_or_build_faiss_index(
            chunks=chunks,
            model=embedding_model,
            source_file=chunks_file,
            index_file=faiss_index_file,
            metadata_file=(
                faiss_metadata_file
            )
        )
    )
    # 第四阶段：加载语义 Reranker。
    # 模型只加载一次，不能每问一个问题重新加载。
    (
        reranker_tokenizer,
        reranker_model,
        reranker_device
    ) = load_reranker()

    # 第五阶段：创建大模型客户端
    llm_client = create_openai_client()


    display_pipeline_summary(
        pdf_file=pdf_file,
        chunks=chunks,
        report=report,
        chunk_status=chunk_status,
        index=index,
        index_status=index_status
    )

    print(f"生成模型：{LLM_MODEL}")
    print("输入 q 退出程序。")

    while True:
        query = input(
            "\n请输入论文问题"
            "（输入 q 退出）："
        ).strip()

        if query.lower() == "q":
            print("程序结束。")
            break

        if not query:
            print("问题不能为空。")
            continue

        # 第一步：FAISS 在本地检索更多候选。
        # 该步骤不调用大模型，不消耗火山方舟 Token。
        candidate_results = retrieve_top_chunks(
            query=query,
            chunks=chunks,
            model=embedding_model,
            index=index,
            top_k=RETRIEVAL_CANDIDATE_K,
            min_similarity=CHUNK_MIN_SIMILARITY
        )

        # 第二阶段：真正的 Cross-Encoder 精排。
        reranked_results = rerank_candidates(
            query=query,
            candidates=candidate_results,
            tokenizer=reranker_tokenizer,
            model=reranker_model,
            device=reranker_device
        )

        # 第三阶段：去噪、去重和页码控制，
        # 最终选择五条证据。
        results = select_reranked_evidence(
            reranked_results
        )

        print(
            f"FAISS 召回："
            f"{len(candidate_results)} 条；"
            f"Reranker 完成评分："
            f"{len(reranked_results)} 条；"
            f"最终证据："
            f"{len(results)} 条。"
        )

        print("\n===== Reranker 前五名 =====")

        for result in results:
            print(
                f"Reranker 第 "
                f"{result['reranker_rank']} 名 | "
                f"FAISS 第 "
                f"{result['faiss_rank']} 名 | "
                f"变化 "
                f"{result['rank_change']:+d} | "
                f"第 {result['page_number']} 页 | "
                f"Chunk {result['chunk_id']} | "
                f"FAISS={result['similarity']:.4f} | "
                f"Reranker="
                f"{result['reranker_score']:.4f}"
            )

        try:
            generation_data = (
                generate_reliable_answer(
                    client=llm_client,
                    query=query,
                    results=results,#Reranker 排序后的最终五条证据
                    refusal_results=(
                        candidate_results#原始的faiss候选用于判断是否拒答
                    )
                )
            )

        except RuntimeError as error:
            print(f"\n{error}")
            continue
        display_reliable_answer(
            query=query,
            generation_data=generation_data
        )
        append_qa_history(
            query=query,
            generation_data=generation_data,
            index=index,
            output_file=qa_history_file
        )

        print("\n问答结果已保存到：")
        print(qa_history_file)


if __name__ == "__main__":
    main()