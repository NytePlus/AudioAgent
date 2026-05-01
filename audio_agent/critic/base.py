"""Abstract base class for final-answer critics."""

from __future__ import annotations

from abc import ABC, abstractmethod

from audio_agent.core.schemas import CriticResult
from audio_agent.core.state import AgentState
from audio_agent.tools.executor import ToolExecutor
from audio_agent.tools.registry import ToolRegistry


class BaseCritic(ABC):
    """Checks a generated final answer before it is accepted."""

    @property
    @abstractmethod
    def name(self) -> str:
        """Return the critic name for logging."""
        raise NotImplementedError

    @abstractmethod
    async def critique(
        self,
        state: AgentState,
        executor: ToolExecutor,
        registry: ToolRegistry,
    ) -> CriticResult:
        """Critique the current final answer using state and optional tools."""
        raise NotImplementedError
