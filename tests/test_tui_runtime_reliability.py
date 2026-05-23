"""Regression tests for TUI runtime reliability fixes."""

import asyncio
from pathlib import Path

from click.testing import CliRunner

from omniagent.agents.sentinel import Milestone, SentinelAgent, TaskPlan
from omniagent.cli.chat import ChatSession
from omniagent.cli.main import main
from omniagent.config.models import AgentConfig, OmniAgentConfig, SentinelConfig
from omniagent.infra.fs import PathTraversalError, safe_path
from omniagent.tools.bash_tool import BashTool


def test_safe_path_allows_private_tmp_workspace(tmp_path: Path) -> None:
    target = tmp_path / "sample.txt"
    target.write_text("ok", encoding="utf-8")

    assert safe_path(tmp_path, target) == target.resolve()

    private_tmp_base = Path("/private/tmp/omniagent-workspace")
    private_tmp_target = private_tmp_base / "sample.txt"
    assert safe_path(private_tmp_base, private_tmp_target) == private_tmp_target.resolve()


def test_safe_path_still_blocks_system_path_outside_workspace(tmp_path: Path) -> None:
    try:
        safe_path(tmp_path, Path("/private/etc/passwd"))
    except PathTraversalError as exc:
        assert "system directory" in str(exc)
    else:
        raise AssertionError("expected system path to be blocked")

    try:
        safe_path(Path("/private/etc"), Path("/private/etc/passwd"))
    except PathTraversalError as exc:
        assert "system directory" in str(exc)
    else:
        raise AssertionError("expected non-temp system workspace to be blocked")


def test_bash_nonzero_exit_includes_stderr(tmp_path: Path) -> None:
    tool = BashTool(work_dir=tmp_path)

    result = asyncio.run(
        tool.execute({"command": "sh -c 'echo command-not-found >&2; exit 7'"})
    )

    assert result.success is False
    assert result.error is not None
    assert "command-not-found" in result.error
    assert result.metadata["exit_code"] == 7


def test_sentinel_mark_milestone_completed_is_sync(tmp_path: Path) -> None:
    sentinel = SentinelAgent(
        config=SentinelConfig(plan_persist_path=".omniagent/sentinel"),
        main_agent_config=AgentConfig(),
        work_dir=str(tmp_path),
    )
    milestone = Milestone(index=0, description="Do work")
    plan = TaskPlan(
        task_hash="task",
        task_description="demo",
        milestones=[milestone],
        created_at="2026-05-23T00:00:00+00:00",
        updated_at="2026-05-23T00:00:00+00:00",
        current_milestone_idx=0,
    )
    sentinel._active_plan = plan
    sentinel.save_plan(plan)

    result = sentinel.mark_milestone_completed(milestone, result_summary="done")

    assert result is None
    assert milestone.status == "completed"
    assert plan.status == "completed"


def test_chat_session_filters_resume_by_work_dir(tmp_path: Path, monkeypatch) -> None:
    home = tmp_path / "home"
    work_dir = tmp_path / "work"
    other_work_dir = tmp_path / "other"
    work_dir.mkdir()
    other_work_dir.mkdir()
    home.mkdir()

    monkeypatch.setenv("HOME", str(home))
    config = OmniAgentConfig()
    monkeypatch.setattr("omniagent.cli.chat.load_config", lambda: config)

    session = ChatSession(work_dir=work_dir)
    current = session.session_manager.create_session("cli_user", "chat", "current")
    current.context["work_dir"] = str(work_dir.resolve())
    current.add_message("user", "hello")
    session.session_manager._save_session(current)

    other = session.session_manager.create_session("cli_user", "chat", "other")
    other.context["work_dir"] = str(other_work_dir.resolve())
    other.add_message("user", "from other workspace")
    other.last_active_at = current.last_active_at.replace(microsecond=999999)
    session.session_manager._save_session(other)

    assert session._find_recent_session().id == "current"


def test_doctor_reports_active_custom_provider(tmp_path: Path) -> None:
    config_path = tmp_path / "config.yaml"
    config_path.write_text(
        """
providers:
  custom:
    api_key: dummy-key
    api_url: https://example.test/v1
    model_id: example-model
agent:
  model_provider: custom
  model_id: example-model
work_dir: .
""",
        encoding="utf-8",
    )

    result = CliRunner().invoke(main, ["--config", str(config_path), "doctor"])

    assert result.exit_code == 0
    assert "Active provider: custom" in result.output
    assert "Active model: example-model" in result.output
    assert "Active API URL: https://example.test/v1" in result.output
    assert "Active provider API key: ✓" in result.output
