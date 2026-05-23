"""Basic smoke tests for project imports and core defaults."""

from omniagent.agents.llm import LLMMessage
from omniagent.config import OmniAgentConfig
from omniagent.tools import ToolRegistry


def test_config_defaults_are_loadable():
    config = OmniAgentConfig(api_key="", enable_self_improving=False)

    assert config.version == "0.1.0"
    assert config.agent.model_provider == "deepseek"
    assert config.agent.model_id == "deepseek-chat"


def test_llm_message_dataclass_is_usable():
    message = LLMMessage(role="user", content="hello")

    assert message.role == "user"
    assert message.content == "hello"
    assert message.tool_calls is None


def test_tool_registry_starts_empty():
    registry = ToolRegistry()

    assert registry.list_tools() == []
