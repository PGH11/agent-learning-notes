"""长期记忆 JSON 持久化与格式化。"""

from __future__ import annotations

import json
import time
import uuid
from typing import Any, Literal

from video_agent.models import MemoryRecord, UserMemory
from video_agent.settings import MEMORY_FILE


def unique_append(items: list[str], new_items: list[str], limit: int | None = None) -> list[str]:
    """追加去重并保持原有顺序。"""

    seen = set()
    merged: list[str] = []
    for item in [*items, *new_items]:
        normalized = item.strip()
        if not normalized or normalized in seen:
            continue
        seen.add(normalized)
        merged.append(normalized)
    return merged[-limit:] if limit is not None else merged


def current_time_iso() -> str:
    """返回用于记忆记录的本地时间字符串。"""

    return time.strftime("%Y-%m-%dT%H:%M:%S%z")


def memory_id_for(category: str, value: str) -> str:
    """根据类别和值生成稳定 id，方便去重更新。"""

    raw = f"{category}:{value.strip().lower()}"
    return f"mem_{uuid.uuid5(uuid.NAMESPACE_URL, raw).hex[:12]}"


def build_memory_record(
    category: Literal[
        "video_style",
        "avoid",
        "favorite_subject",
        "workflow_preference",
        "confirmed_prompt",
    ],
    value: str,
    confidence: float = 1.0,
    reason: str = "",
    source_text: str = "",
    created_at: str | None = None,
) -> MemoryRecord:
    """创建一条长期记忆记录。"""

    now = current_time_iso()
    normalized_value = value.strip()
    return MemoryRecord(
        id=memory_id_for(category, normalized_value),
        category=category,
        value=normalized_value,
        confidence=confidence,
        reason=reason,
        source_text=source_text,
        created_at=created_at or now,
        updated_at=now,
    )


def migrate_legacy_memory(data: dict[str, Any]) -> UserMemory:
    """兼容旧版 5 个数组结构，自动迁移到 memories 条目结构。"""

    if "memories" in data:
        return UserMemory.model_validate(data)

    category_map: dict[str, str] = {
        "video_styles": "video_style",
        "avoid": "avoid",
        "favorite_subjects": "favorite_subject",
        "workflow_preferences": "workflow_preference",
        "confirmed_prompts": "confirmed_prompt",
    }
    records: list[MemoryRecord] = []
    for legacy_key, category in category_map.items():
        values = data.get(legacy_key, [])
        if not isinstance(values, list):
            continue
        for value in values:
            if isinstance(value, str) and value.strip():
                records.append(
                    build_memory_record(
                        category=category,  # type: ignore[arg-type]
                        value=value,
                        reason=f"由旧版字段 {legacy_key} 自动迁移",
                        source_text=value,
                    )
                )
    return UserMemory(memories=records)


def load_user_memory() -> UserMemory:
    """读取长期记忆文件，不存在时返回空记忆。"""

    if not MEMORY_FILE.exists():
        return UserMemory()

    try:
        data = json.loads(MEMORY_FILE.read_text(encoding="utf-8"))
    except (json.JSONDecodeError, OSError):
        return UserMemory()
    if not isinstance(data, dict):
        return UserMemory()
    return migrate_legacy_memory(data)


def save_user_memory(memory: UserMemory) -> None:
    """保存长期记忆。"""

    MEMORY_FILE.parent.mkdir(parents=True, exist_ok=True)
    MEMORY_FILE.write_text(
        memory.model_dump_json(ensure_ascii=False, indent=2),
        encoding="utf-8",
    )


def format_user_memory(memory: UserMemory) -> str:
    """把长期记忆整理成提示词上下文。"""

    if not memory.memories:
        return "暂无长期记忆。"

    labels = {
        "video_style": "用户偏好的视频风格",
        "avoid": "用户要求避免的元素",
        "favorite_subject": "用户常用主题",
        "workflow_preference": "用户协作偏好",
        "confirmed_prompt": "最近确认过的 prompt",
    }
    grouped: dict[str, list[MemoryRecord]] = {}
    for record in memory.memories:
        grouped.setdefault(record.category, []).append(record)

    lines: list[str] = []
    for category, label in labels.items():
        records = grouped.get(category, [])
        if not records:
            continue
        if category == "confirmed_prompt":
            lines.append(f"{label}：")
            lines.extend(f"- {record.value}" for record in records[-3:])
        else:
            lines.append(f"{label}：{', '.join(record.value for record in records)}")
    return "\n".join(lines) if lines else "暂无长期记忆。"
