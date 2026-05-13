"""兼容旧入口：转发到 `video_agent.cli`。"""

from video_agent.cli import run_cli

if __name__ == "__main__":
    run_cli()
