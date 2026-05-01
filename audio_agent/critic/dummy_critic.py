"""Deterministic critic implementation for tests and dummy runs."""

from __future__ import annotations

from audio_agent.critic.base import BaseCritic
from audio_agent.core.schemas import CriticCheckResult, CriticResult
from audio_agent.core.state import AgentState
from audio_agent.tools.executor import ToolExecutor
from audio_agent.tools.registry import ToolRegistry


class DummyCritic(BaseCritic):
    """A lightweight critic that passes non-empty answers."""

    @property
    def name(self) -> str:
        return "dummy_critic"

    async def critique(
        self,
        state: AgentState,
        executor: ToolExecutor,
        registry: ToolRegistry,
    ) -> CriticResult:
        del executor, registry
        decision = state.get("current_decision")
        answer = decision.draft_answer if decision else None
        passed = bool(answer and answer.strip())
        critique = None if passed else "Final answer is empty."
        return CriticResult(
            passed=passed,
            critique=critique,
            confidence=1.0 if passed else 0.0,
            checks=[
                CriticCheckResult(
                    name="format",
                    passed=passed,
                    critique=critique,
                    confidence=1.0 if passed else 0.0,
                )
            ],
            transcript_edits=[],
        )
