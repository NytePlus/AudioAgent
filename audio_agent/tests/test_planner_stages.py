"""Tests for two-stage planner node flow (initial plan + action decision)."""

import asyncio

import pytest

from audio_agent.core.errors import StateValidationError, ToolExecutionError
from audio_agent.core.schemas import AudioItem, FrontendOutput, InitialPlan, PlannedToolCall, PlannerActionType
from audio_agent.core.state import create_initial_state
from audio_agent.fusion.default_fuser import DefaultEvidenceFuser
from audio_agent.graph.nodes import (
    create_initial_plan_node,
    create_parallel_initial_tools_node,
    create_planner_decision_node,
)
from audio_agent.planner.dummy_planner import DummyPlanner
from audio_agent.tools.executor import ToolExecutor
from audio_agent.tools.dummy_tools import DummyASRTool
from audio_agent.tools.registry import ToolRegistry


class TestPlannerStages:
    """Tests for initial planning and decision-stage fail-fast behavior."""

    def test_initial_prompt_node_generates_prompt_from_question(self):
        from audio_agent.graph.nodes import create_initial_prompt_node
        planner = DummyPlanner()
        node = create_initial_prompt_node(planner)

        result = node({"question": "What is discussed in this audio?"})

        assert result["question_oriented_prompt"] is not None
        assert "What is discussed in this audio?" in result["question_oriented_prompt"]

    def test_initial_plan_node_generates_plan_from_question_and_frontend_output(self):
        from audio_agent.core.schemas import FrontendOutput
        planner = DummyPlanner()
        registry = ToolRegistry()
        registry.register(DummyASRTool())
        node = create_initial_plan_node(planner, registry)

        result = node({
            "question": "What is discussed in this audio?",
            "initial_frontend_output": FrontendOutput(
                question_guided_caption="A discussion about technology."
            ),
        })

        assert result["initial_plan"] is not None
        assert result["initial_plan"].approach
        assert len(result["initial_plan_trace"]) == 1

    def test_parallel_initial_tools_node_noops_for_empty_queue(self):
        registry = ToolRegistry()
        registry.register(DummyASRTool())
        node = create_parallel_initial_tools_node(
            ToolExecutor(registry),
            DefaultEvidenceFuser(),
            registry,
        )
        state = create_initial_state(
            "Question",
            ["/fake/audio.wav"],
            audio_list=[
                AudioItem(
                    audio_id="audio_0",
                    path="/fake/audio.wav",
                    source="original",
                    description="Input audio",
                )
            ],
        )
        state["initial_plan"] = InitialPlan(
            approach="No initial tools needed.",
            focus_points=[],
            possible_tool_types=[],
        )

        result = asyncio.run(node(state))

        assert result == {}

    def test_parallel_initial_tools_node_runs_multiple_tools(self):
        registry = ToolRegistry()
        registry.register(DummyASRTool())
        node = create_parallel_initial_tools_node(
            ToolExecutor(registry),
            DefaultEvidenceFuser(),
            registry,
        )
        state = create_initial_state(
            "Question",
            ["/fake/audio.wav"],
            audio_list=[
                AudioItem(
                    audio_id="audio_0",
                    path="/fake/audio.wav",
                    source="original",
                    description="Input audio",
                )
            ],
        )
        state["initial_plan"] = InitialPlan(
            approach="Run independent ASR checks.",
            focus_points=["speech"],
            possible_tool_types=["asr"],
            planned_tool_calls=[
                PlannedToolCall(
                    step_number=1,
                    tool_name="dummy_asr",
                    tool_args={"audio_path": "audio_0"},
                    audio_id="audio_0",
                    rationale="Initial transcript.",
                ),
                PlannedToolCall(
                    step_number=2,
                    tool_name="dummy_asr",
                    tool_args={},
                    audio_id="audio_0",
                    rationale="Second independent transcript.",
                ),
            ],
        )

        result = asyncio.run(node(state))

        assert len(result["tool_call_history"]) == 2
        assert len(result["evidence_log"]) == 2
        assert result["step_count"] == 2
        assert all(record.request.args["audio_path"] == "/fake/audio.wav" for record in result["tool_call_history"])

    def test_parallel_initial_tools_node_raises_for_invalid_tool(self):
        registry = ToolRegistry()
        registry.register(DummyASRTool())
        node = create_parallel_initial_tools_node(
            ToolExecutor(registry),
            DefaultEvidenceFuser(),
            registry,
        )
        state = create_initial_state(
            "Question",
            ["/fake/audio.wav"],
            audio_list=[
                AudioItem(
                    audio_id="audio_0",
                    path="/fake/audio.wav",
                    source="original",
                    description="Input audio",
                )
            ],
        )
        state["initial_plan"] = InitialPlan(
            approach="Run an unavailable tool.",
            focus_points=["speech"],
            possible_tool_types=["asr"],
            planned_tool_calls=[
                PlannedToolCall(
                    step_number=1,
                    tool_name="missing_tool",
                    tool_args={},
                    audio_id="audio_0",
                    rationale="Should fail.",
                )
            ],
        )

        with pytest.raises(ToolExecutionError, match="Failed to find tool"):
            asyncio.run(node(state))

    def test_planner_decision_node_raises_if_initial_plan_missing(self):
        planner = DummyPlanner()
        registry = ToolRegistry()
        registry.register(DummyASRTool())
        node = create_planner_decision_node(planner, registry)

        state = create_initial_state("Question", "/tmp/audio.wav")
        state["initial_frontend_output"] = FrontendOutput(
            question_guided_caption="A short caption."
        )

        with pytest.raises(StateValidationError, match="Missing required state fields"):
            node(state)

    def test_planner_decision_node_raises_if_frontend_output_missing(self):
        planner = DummyPlanner()
        registry = ToolRegistry()
        registry.register(DummyASRTool())
        node = create_planner_decision_node(planner, registry)

        state = create_initial_state("Question", "/tmp/audio.wav")
        state["initial_plan"] = planner.plan("Question")

        with pytest.raises(StateValidationError, match="Missing required state fields"):
            node(state)

    def test_planner_decision_node_runs_after_stage1_outputs_exist(self):
        planner = DummyPlanner()
        registry = ToolRegistry()
        registry.register(DummyASRTool())
        node = create_planner_decision_node(planner, registry)

        state = create_initial_state("Question", "/tmp/audio.wav")
        state["initial_frontend_output"] = FrontendOutput(
            question_guided_caption="A short caption."
        )
        state["initial_plan"] = planner.plan("Question")

        result = node(state)

        assert result["current_decision"] is not None
        assert result["current_decision"].action == PlannerActionType.CALL_TOOL
        assert len(result["planner_trace"]) == 1
