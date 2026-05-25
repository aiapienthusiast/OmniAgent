import asyncio
import json
from types import SimpleNamespace

from omniagent.agents.state import AgentState
from omniagent.gateway.api import APIContext, get_health


class FakeRequest:
    def __init__(self, ctx):
        self.app = {"api_ctx": ctx}


def test_health_serializes_pending_tool_calls():
    async def run():
        state = AgentState()
        state.pending_tool_calls.add("tool-call-1")
        agent = SimpleNamespace(
            agent_state=state,
            config=SimpleNamespace(
                agent=SimpleNamespace(model_provider="deepseek", model_id="deepseek-chat")
            ),
            memory_manager=None,
            policy=None,
            approval_manager=None,
        )
        ctx = APIContext(agent=agent, session_manager=None, config=None, channel_manager=None)
        response = await get_health(FakeRequest(ctx))

        assert response.status == 200
        data = json.loads(response.text)
        assert data["agent"]["pending_tool_calls"] == ["tool-call-1"]

    asyncio.run(run())
