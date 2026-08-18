"""火山方舟 Chat Model 的创建与配置。"""

from __future__ import annotations

import os
from pathlib import Path

import httpx
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI


PROJECT_ROOT = Path(__file__).resolve().parents[1]
load_dotenv(PROJECT_ROOT / ".env")


def require_env(name: str) -> str:
    value = os.getenv(name, "").strip()
    if not value:
        raise RuntimeError(
            f"缺少环境变量 {name}，请检查项目根目录中的 .env。"
        )
    return value


def create_ark_chat_model() -> ChatOpenAI:
    """创建温度为 0 的火山方舟 OpenAI 兼容模型。"""
    http_client = httpx.Client(
        trust_env=False,
        timeout=httpx.Timeout(60.0, connect=20.0),
    )
    return ChatOpenAI(
        model=require_env("ARK_MODEL_ID"),
        api_key=require_env("ARK_API_KEY"),
        base_url=require_env("ARK_BASE_URL"),
        temperature=0,
        max_retries=2,
        http_client=http_client,
    )
