"""图内通用小函数。"""

from __future__ import annotations

from video_agent.models import ChatTurn, CreativeParams


def history_to_text(history: list[ChatTurn], limit: int = 10) -> str:
    """把最近对话转成提示词上下文。"""

    return "\n".join(f"{turn.role}: {turn.content}" for turn in history[-limit:])


def format_final_params(params: CreativeParams) -> str:
    """输出最终内容参数。"""

    return (
        "内容已确认，准备合并前端 UI 参数后提交：\n"
        f"{params.model_dump_json(ensure_ascii=False, exclude_none=True, indent=2)}"
    )
