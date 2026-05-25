"""Tests for conversation-history priority in memory recall guidance."""

import asyncio
from typing import AsyncIterator, List

from omniagent.agents.llm import LLMMessage, LLMProvider, LLMResponse
from omniagent.agents.memory_manager import MemorySearchTool
from omniagent.agents.reflexion import ReflexionAgent
from omniagent.config import OmniAgentConfig


class FakeLLMProvider(LLMProvider):
    async def chat(
        self,
        messages: List[LLMMessage],
        temperature: float = 0.7,
        max_tokens: int = 4096,
        stream: bool = False,
        tools=None,
    ) -> LLMResponse:
        return LLMResponse(content="OK")

    async def chat_stream(
        self,
        messages: List[LLMMessage],
        temperature: float = 0.7,
        max_tokens: int = 4096,
    ) -> AsyncIterator[str]:
        if False:
            yield ""


class EmptyMemoryManager:
    def search(self, query: str, max_results: int = 6, min_score: float = 0.35):
        return []


def test_system_prompt_prioritizes_current_conversation_for_short_term_references(tmp_path):
    config = OmniAgentConfig(api_key="", enable_self_improving=False)
    config.sentinel.enabled = False
    config.guardian.enabled = False
    config.agent.compaction_enabled = False
    config.memory.enabled = True

    agent = ReflexionAgent(
        config=config,
        llm_provider=FakeLLMProvider(),
        work_dir=tmp_path,
        enable_security=False,
    )

    prompt = agent._build_system_prompt()

    assert "Current conversation history is the highest-priority source" in prompt
    assert "刚才" in prompt
    assert "do not conclude the user never mentioned it" in prompt


def test_memory_search_tool_description_excludes_current_conversation_recall():
    tool = MemorySearchTool(EmptyMemoryManager())

    assert "Search long-term memory" in tool.description
    assert "use the conversation history first" in tool.description
    assert "刚才" in tool.description


def test_empty_memory_search_reminds_model_to_check_recent_conversation():
    tool = MemorySearchTool(EmptyMemoryManager())

    result = asyncio.run(tool.execute({"query": "刚才的临时测试暗号"}))

    assert result.success
    assert "No relevant long-term memories found" in result.output
    assert "check recent conversation history" in result.output
