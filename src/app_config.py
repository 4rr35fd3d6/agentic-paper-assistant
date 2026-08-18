"""
Streamlit 论文 RAG 应用的路径和通用提示词配置。
"""

from pathlib import Path


PROJECT_ROOT = (
    Path(__file__)
    .resolve()
    .parents[1]
)

PRACTICE_DIR = (
    PROJECT_ROOT
    / "practice"
)

UPLOAD_ROOT = (
    PROJECT_ROOT
    / "data"
    / "day23_uploads"
)

CACHE_ROOT = (
    PROJECT_ROOT
    / "data"
    / "day23_cache"
)


DAY23_SYSTEM_INSTRUCTIONS = """
你是一名严谨的学术论文问答助手。

请遵守以下规则：

1. 只能根据程序提供的论文证据回答。
2. 不得把外部知识当作当前论文中的内容。
3. 如果证据不足，必须明确说明无法确定。
4. 不得捏造方法、公式、数据集、实验结果或结论。
5. 每个主要事实或结论后必须添加 [S1]、[S2] 等证据引用。
6. 引用编号必须来自程序提供的证据。
7. 应区分论文明确陈述的事实与根据证据作出的有限归纳。
8. 回答应准确、简洁。
""".strip()
