import hashlib
import json
from pathlib import Path

import faiss
import numpy as np
from sentence_transformers import SentenceTransformer


MODEL_NAME = "BAAI/bge-small-zh-v1.5"

QUERY_INSTRUCTION = (
    "为这个句子生成表示以用于检索相关文章："
)

TOP_K = 3
MIN_SIMILARITY = 0.30


def get_project_root():
    """
    职责：获取项目根目录。

    输出：
        Path 类型的项目根目录
    """
    return Path(__file__).resolve().parents[1]


def load_chunks(input_file):
    """
    职责：读取文本块 JSON。

    输入：
        input_file：文本块文件路径

    输出：
        chunks：文本块列表
    """
    if not input_file.exists():
        raise FileNotFoundError(
            f"没有找到文本块文件：{input_file}"
        )

    with input_file.open(
        "r",
        encoding="utf-8"
    ) as file:
        chunks = json.load(file)

    if not chunks:
        raise ValueError("文本块文件中没有数据")

    return chunks


def load_json(json_file):
    """
    职责：读取普通 JSON 文件。
    """
    with json_file.open(
        "r",
        encoding="utf-8"
    ) as file:
        return json.load(file)


def save_json(data, output_file):
    """
    职责：将数据保存成 JSON 文件。
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


def calculate_file_sha256(file_path):
    """
    职责：
        计算文件的 SHA-256 指纹。

    输入：
        file_path：需要检查的文件路径

    输出：
        文件内容对应的哈希字符串

    用途：
        判断文本块文件是否发生变化。
        文件内容只要发生变化，哈希值通常也会变化。
    """
    hasher = hashlib.sha256()

    with file_path.open("rb") as file:
        while True:
            data_block = file.read(8192)

            if not data_block:
                break

            hasher.update(data_block)

    return hasher.hexdigest()


def load_embedding_model(model_name):
    """
    职责：加载 Embedding 模型。
    """
    print(f"正在加载 Embedding 模型：{model_name}")

    model = SentenceTransformer(model_name)

    return model


def encode_chunks(chunks, model):
    """
    职责：
        将所有文本块编码成归一化的 float32 向量。

    输入：
        chunks：文本块列表
        model：Embedding 模型

    输出：
        chunk_embeddings：二维 NumPy 向量矩阵
    """
    corpus = []

    for chunk in chunks:
        corpus.append(chunk["content"])

    chunk_embeddings = model.encode(
        corpus,
        batch_size=32,
        show_progress_bar=True,
        convert_to_numpy=True,
        normalize_embeddings=True
    )

    # FAISS 使用 float32 向量。
    # ascontiguousarray 还会保证数据在内存中连续存储。
    chunk_embeddings = np.ascontiguousarray(
        chunk_embeddings,
        dtype=np.float32
    )#把向量矩阵转成连续存储的 float32 数组

    return chunk_embeddings


def build_faiss_index(
    chunks,
    model,
    source_hash,
    index_file,
    metadata_file
):
    """
    职责：
        生成文档 Embedding，
        建立 FAISS 内积索引，
        将索引与元数据保存到磁盘。

    输出：
        index：建立好的 FAISS 索引
    """
    print("正在重新生成文本块向量……")

    chunk_embeddings = encode_chunks(
        chunks=chunks,
        model=model
    )

    embedding_dimension = (
        chunk_embeddings.shape[1]
    )#向量维度

    # IndexFlatIP：
    # Flat 表示精确搜索；
    # IP 表示 Inner Product，即内积。
    index = faiss.IndexFlatIP(
        embedding_dimension
    )

    # 把所有文档向量加入索引。
    index.add(chunk_embeddings)

    index_file.parent.mkdir(
        parents=True,
        exist_ok=True
    )

    # 保存二进制 FAISS 索引。
    faiss.write_index(
        index,
        str(index_file)
    )

    metadata = {
        "index_type": "IndexFlatIP",
        "model_name": MODEL_NAME,
        "embedding_dimension": (
            embedding_dimension
        ),
        "chunk_count": len(chunks),
        "source_sha256": source_hash,
        "normalize_embeddings": True
    }

    save_json(
        data=metadata,
        output_file=metadata_file
    )

    return index


def index_is_valid(
    index,
    metadata,
    chunks,
    current_source_hash
):
    """
    职责：
        检查磁盘索引是否仍然与当前知识库匹配。

    检查内容：
        1. 模型名称
        2. 文本块文件指纹
        3. 文本块数量
        4. 向量数量
        5. 向量维度
    """
    required_keys = {
        "model_name",
        "embedding_dimension",
        "chunk_count",
        "source_sha256"
    }

    if not required_keys.issubset(
        metadata.keys()
    ):
        return False

    if metadata["model_name"] != MODEL_NAME:
        return False

    if (
        metadata["source_sha256"]
        != current_source_hash
    ):
        return False

    if metadata["chunk_count"] != len(chunks):
        return False

    if index.ntotal != len(chunks):
        return False

    if (
        index.d
        != metadata["embedding_dimension"]
    ):
        return False

    return True


def load_or_build_faiss_index(
    chunks,
    model,
    source_file,
    index_file,
    metadata_file
):
    """
    职责：
        已有索引有效时直接加载；
        索引不存在或已经过期时重新建立。

    输出：
        index：FAISS 索引
        index_status：本次是加载还是重建
    """
    current_source_hash = (
        calculate_file_sha256(source_file)
    )#指纹，会与metadata里的旧哈希比较

    files_exist = (
        index_file.exists()
        and metadata_file.exists()
    )

    if files_exist:
        try:
            index = faiss.read_index(
                str(index_file)
            )

            metadata = load_json(
                metadata_file
            )

            if index_is_valid(
                index=index,
                metadata=metadata,
                chunks=chunks,
                current_source_hash=(
                    current_source_hash
                )
            ):
                return index, "已从磁盘加载已有索引"

            print(
                "已有索引与当前文本块不一致，"
                "将重新建立索引。"
            )

        except (
            OSError,
            ValueError,
            KeyError,
            RuntimeError,
            json.JSONDecodeError
        ) as error:
            print(
                "已有索引读取失败，"
                "将重新建立索引。"
            )
            print(f"读取失败原因：{error}")

    index = build_faiss_index(
        chunks=chunks,
        model=model,
        source_hash=current_source_hash,
        index_file=index_file,
        metadata_file=metadata_file
    )

    return index, "已新建并保存索引"


def retrieve_top_chunks(
    query,
    chunks,
    model,
    index,
    top_k=TOP_K,
    min_similarity=MIN_SIMILARITY
):
    """
    职责：
        将查询编码为向量，
        使用 FAISS 检索 Top K 文本块。

    输出：
        results：达到相似度阈值的检索结果
    """
    if top_k <= 0:
        raise ValueError("top_k 必须大于 0")

    cleaned_query = query.strip()

    if not cleaned_query:
        return []

    instructed_query = (
        QUERY_INSTRUCTION
        + cleaned_query
    )

    query_embedding = model.encode(
        [instructed_query],
        convert_to_numpy=True,
        normalize_embeddings=True
    )

    query_embedding = np.ascontiguousarray(
        query_embedding,
        dtype=np.float32
    )

    actual_top_k = min(
        top_k,
        index.ntotal
    )

    # scores：相似度分数
    # indices：向量在索引中的编号
    scores, indices = index.search(
        query_embedding,
        actual_top_k
    )

    results = []

    # 当前只有一个查询，
    # 所以读取 scores[0] 和 indices[0]。
    for score, vector_id in zip(
        scores[0],
        indices[0]
    ):
        vector_id = int(vector_id)
        similarity = float(score)

        # -1 表示 FAISS 没有返回有效编号。
        if vector_id < 0:
            continue

        # IndexFlatIP 已按照内积从高到低返回结果。
        # 当前分数不达标，后面的分数只会更低。
        if similarity < min_similarity:
            break

        # FAISS 编号与 chunks 列表位置一一对应。
        result = chunks[vector_id].copy()

        result["faiss_id"] = vector_id
        result["similarity"] = round(
            similarity,
            4
        )

        results.append(result)

    return results


def display_results(query, results):
    """
    职责：在终端显示检索结果。
    """
    print(f"\n查询内容：{query}")

    if not results:
        print(
            "知识库中没有找到达到"
            "相关性要求的文本块。"
        )
        return

    print(f"返回结果数量：{len(results)}")

    for rank, result in enumerate(
        results,
        start=1
    ):
        print("\n" + "=" * 60)

        print(
            f"排名 {rank}："
            f"Chunk {result['chunk_id']}"
        )

        print(
            f"FAISS 向量编号："
            f"{result['faiss_id']}"
        )

        print(
            f"语义相似度："
            f"{result['similarity']:.4f}"
        )

        print(
            f"原文位置："
            f"{result['start']}～{result['end']}"
        )

        print("\n文本内容：")
        print(result["content"])


def save_results(
    query,
    results,
    index,
    output_file
):
    """
    职责：保存本次 FAISS 检索结果。
    """
    result_data = {
        "retrieval_method": (
            "FAISS dense vector retrieval"
        ),
        "index_type": "IndexFlatIP",
        "model_name": MODEL_NAME,
        "query_instruction": (
            QUERY_INSTRUCTION
        ),
        "top_k": TOP_K,
        "min_similarity": MIN_SIMILARITY,
        "index_vector_count": int(
            index.ntotal
        ),
        "embedding_dimension": int(
            index.d
        ),
        "query": query,
        "result_count": len(results),
        "results": results
    }

    save_json(
        data=result_data,
        output_file=output_file
    )


def main():
    """
    Day 16 主流程：

    索引阶段：
        1. 加载文本块
        2. 加载 Embedding 模型
        3. 检查 FAISS 索引
        4. 索引有效则加载
        5. 索引无效则重新构建

    查询阶段：
        6. 编码用户查询
        7. 使用 FAISS 搜索
        8. 应用相似度阈值
        9. 显示并保存结果
    """
    project_root = get_project_root()

    source_file = (
        project_root
        / "data"
        / "day12_chunks.json"
    )

    index_file = (
        project_root
        / "data"
        / "day16_faiss.index"
    )

    metadata_file = (
        project_root
        / "data"
        / "day16_faiss_metadata.json"
    )

    results_file = (
        project_root
        / "data"
        / "day16_faiss_results.json"
    )

    chunks = load_chunks(source_file)

    model = load_embedding_model(
        MODEL_NAME
    )

    index, index_status = (
        load_or_build_faiss_index(
            chunks=chunks,
            model=model,
            source_file=source_file,
            index_file=index_file,
            metadata_file=metadata_file
        )
    )

    print("\n===== Day 16 FAISS 向量检索 =====")
    print(f"索引状态：{index_status}")
    print(f"索引类型：IndexFlatIP")
    print(f"索引是否已训练：{index.is_trained}")
    print(f"索引向量数量：{index.ntotal}")
    print(f"向量维度：{index.d}")
    print(f"最低相似度：{MIN_SIMILARITY}")
    print("输入 q 退出程序。")

    while True:
        query = input(
            "\n请输入检索问题："
        ).strip()

        if query.lower() == "q":
            print("已退出 FAISS 检索系统。")
            break

        if not query:
            print("查询不能为空。")
            continue

        results = retrieve_top_chunks(
            query=query,
            chunks=chunks,
            model=model,
            index=index,
            top_k=TOP_K,
            min_similarity=MIN_SIMILARITY
        )

        display_results(
            query=query,
            results=results
        )

        save_results(
            query=query,
            results=results,
            index=index,
            output_file=results_file
        )

        print("\n检索结果已保存到：")
        print(results_file)


if __name__ == "__main__":
    main()