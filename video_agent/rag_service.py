"""本地 Markdown 轻量 RAG（可替换为向量检索实现）。"""

from __future__ import annotations

import re
from pathlib import Path

from video_agent.models import RagDocument
from video_agent.settings import KNOWLEDGE_DIR, RAG_CHUNK_MAX_CHARS, RAG_TOP_K


def tokenize_for_rag(text: str) -> set[str]:
    """提取中英文关键词，用于轻量级本地 RAG 检索。"""

    normalized = text.lower()
    tokens = set(re.findall(r"[a-z0-9_]+|[\u4e00-\u9fff]{2,}", normalized))
    chinese_chars = re.findall(r"[\u4e00-\u9fff]", normalized)
    tokens.update(
        "".join(chinese_chars[index : index + 2])
        for index in range(max(len(chinese_chars) - 1, 0))
    )
    synonyms = {
        "puppy": {"小狗", "幼犬", "宠物"},
        "dog": {"小狗", "宠物"},
        "cat": {"猫咪", "宠物"},
        "kitten": {"猫咪", "宠物"},
        "pet": {"宠物", "小狗", "猫咪"},
        "healing": {"治愈", "温暖", "柔和"},
        "cozy": {"治愈", "温暖"},
        "cinematic": {"电影感", "镜头", "构图"},
        "food": {"美食", "食物"},
        "city": {"城市", "街道"},
        "fantasy": {"奇幻", "魔法"},
        "negative": {"负向", "不要", "避免"},
    }
    for token in list(tokens):
        tokens.update(synonyms.get(token, set()))
    return {token for token in tokens if len(token) >= 2}


def split_markdown_sections(path: Path, max_chars: int = RAG_CHUNK_MAX_CHARS) -> list[str]:
    """按标题切分 Markdown；过长片段再按长度粗切。"""

    text = path.read_text(encoding="utf-8")
    sections: list[str] = []
    current: list[str] = []

    for line in text.splitlines():
        if line.startswith("#") and current:
            sections.append("\n".join(current).strip())
            current = [line]
        else:
            current.append(line)

    if current:
        sections.append("\n".join(current).strip())

    chunks: list[str] = []
    for section in sections:
        if len(section) <= max_chars:
            chunks.append(section)
            continue
        for start in range(0, len(section), max_chars):
            chunks.append(section[start : start + max_chars].strip())

    return [chunk for chunk in chunks if chunk]


def search_knowledge_base(query: str, top_k: int = RAG_TOP_K) -> list[RagDocument]:
    """从 knowledge/*.md 中检索和当前输入最相关的片段。"""

    if not KNOWLEDGE_DIR.exists():
        return []

    query_tokens = tokenize_for_rag(query)
    if not query_tokens:
        return []

    candidates: list[RagDocument] = []
    for path in sorted(KNOWLEDGE_DIR.glob("*.md")):
        for chunk in split_markdown_sections(path):
            chunk_lower = chunk.lower()
            chunk_tokens = tokenize_for_rag(chunk)
            overlap_score = len(query_tokens & chunk_tokens) * 3
            substring_score = sum(1 for token in query_tokens if token in chunk_lower)
            score = overlap_score + substring_score
            if score > 0:
                candidates.append(
                    RagDocument(source=path.name, content=chunk, score=score)
                )

    candidates.sort(key=lambda item: item.score, reverse=True)
    return candidates[:top_k]


def format_rag_context(documents: list[RagDocument]) -> str:
    """把检索片段整理成可注入提示词的上下文。"""

    if not documents:
        return "暂无相关知识库资料。"

    return "\n\n".join(
        f"[来源：{doc.source}，相关度：{doc.score}]\n{doc.content}"
        for doc in documents
    )
