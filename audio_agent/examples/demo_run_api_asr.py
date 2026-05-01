#!/usr/bin/env python3
"""
API-based ASR demo that starts the agent graph from frontend evidence.

This script uses a fixed JSON ASR prompt and skips only the initial_prompt_node.
The rest of the graph runs normally:
frontend evidence -> initial plan -> planner/tools/evidence fusion -> final answer.
"""

from __future__ import annotations

import argparse
import asyncio
import os
import sys
import tempfile
from pathlib import Path

# Add project root to path for direct execution
project_root = Path(__file__).parent.parent.parent
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

from langgraph.graph import END, START, StateGraph

from audio_agent.config.settings import AgentConfig
from audio_agent.core.constants import AgentStatus
from audio_agent.core.logging import set_debug_mode, setup_logger
from audio_agent.core.state import AgentState
from audio_agent.examples.demo_run_api_full import (
    LOCAL_MODEL_TOOL_NAMES,
    print_evidence_log,
    print_initial_plan,
    print_planner_trace,
    print_separator,
    print_tool_history,
)
from audio_agent.frontend.base import BaseFrontend
from audio_agent.frontend.openai_compatible_frontend import OpenAICompatibleFrontend
from audio_agent.fusion.base import BaseEvidenceFuser
from audio_agent.fusion.default_fuser import DefaultEvidenceFuser
from audio_agent.graph.nodes import (
    answer_node,
    create_evidence_fusion_node,
    create_evidence_summarization_node,
    create_final_answer_node,
    create_format_check_node,
    create_frontend_evidence_node,
    create_initial_plan_node,
    create_intent_clarification_node,
    create_parallel_initial_tools_node,
    create_planner_decision_node,
    create_tool_executor_node,
    failure_node,
)
from audio_agent.graph.routing import (
    NODE_ANSWER,
    NODE_EVIDENCE_FUSION,
    NODE_EVIDENCE_SUMMARIZATION,
    NODE_FAILURE,
    NODE_FINAL_ANSWER,
    NODE_FORMAT_CHECK,
    NODE_INITIAL_PLAN,
    NODE_INTENT_CLARIFICATION,
    NODE_PARALLEL_INITIAL_TOOLS,
    NODE_PLANNER_DECISION,
    NODE_TOOL_EXECUTOR,
    route_after_format_check,
    route_after_planner_decision,
)
from audio_agent.log import RunLogger
from audio_agent.main import AudioAgent
from audio_agent.planner.base import BasePlanner
from audio_agent.planner.openai_compatible_planner import OpenAICompatiblePlanner
from audio_agent.tools.catalog import list_available_tools, register_all_mcp_tools
from audio_agent.tools.executor import ToolExecutor
from audio_agent.tools.mcp import MCPServerManager
from audio_agent.tools.registry import ToolRegistry
from audio_agent.utils.audio_path import resolve_audio_input_paths

DEFAULT_ASR_PROMPT = (
    "Transcribe the audio to text. Return only one valid JSON object with exactly this "
    'schema: {"pred": "<transcript>"}. Do not include markdown, explanations, or extra keys.'
)
# Default OpenAI-compatible endpoint (DashScope). Shared by frontend and planner unless overridden.
DEFAULT_COMPAT_API_BASE_URL = "https://dashscope.aliyuncs.com/compatible-mode/v1"
NODE_FRONTEND_EVIDENCE = "frontend_evidence_node"


def build_frontend_first_graph(
    frontend: BaseFrontend,
    planner: BasePlanner,
    registry: ToolRegistry,
    fuser: BaseEvidenceFuser,
):
    """Build the standard graph without the initial prompt node."""
    if frontend is None:
        raise ValueError("frontend cannot be None")
    if planner is None:
        raise ValueError("planner cannot be None")
    if registry is None:
        raise ValueError("registry cannot be None")
    if fuser is None:
        raise ValueError("fuser cannot be None")

    executor = ToolExecutor(registry)

    frontend_node = create_frontend_evidence_node(frontend)
    initial_plan_node_fn = create_initial_plan_node(planner, registry)
    parallel_initial_tools_node_fn = create_parallel_initial_tools_node(executor, fuser, registry)
    planner_decision_node_fn = create_planner_decision_node(planner, registry)
    tool_executor_node_fn = create_tool_executor_node(executor)
    evidence_fusion_node_fn = create_evidence_fusion_node(fuser)
    intent_clarification_node_fn = create_intent_clarification_node(planner)
    evidence_summarization_node_fn = create_evidence_summarization_node(planner)
    final_answer_node_fn = create_final_answer_node(frontend)
    format_check_node_fn = create_format_check_node(planner)

    graph = StateGraph(AgentState)

    graph.add_node(NODE_FRONTEND_EVIDENCE, frontend_node)
    graph.add_node(NODE_INITIAL_PLAN, initial_plan_node_fn)
    graph.add_node(NODE_PARALLEL_INITIAL_TOOLS, parallel_initial_tools_node_fn)
    graph.add_node(NODE_PLANNER_DECISION, planner_decision_node_fn)
    graph.add_node(NODE_TOOL_EXECUTOR, tool_executor_node_fn)
    graph.add_node(NODE_EVIDENCE_FUSION, evidence_fusion_node_fn)
    graph.add_node(NODE_INTENT_CLARIFICATION, intent_clarification_node_fn)
    graph.add_node(NODE_EVIDENCE_SUMMARIZATION, evidence_summarization_node_fn)
    graph.add_node(NODE_FINAL_ANSWER, final_answer_node_fn)
    graph.add_node(NODE_FORMAT_CHECK, format_check_node_fn)
    graph.add_node(NODE_ANSWER, answer_node)
    graph.add_node(NODE_FAILURE, failure_node)

    graph.add_edge(START, NODE_FRONTEND_EVIDENCE)
    graph.add_edge(NODE_FRONTEND_EVIDENCE, NODE_INITIAL_PLAN)
    graph.add_edge(NODE_INITIAL_PLAN, NODE_PARALLEL_INITIAL_TOOLS)
    graph.add_edge(NODE_PARALLEL_INITIAL_TOOLS, NODE_PLANNER_DECISION)
    graph.add_conditional_edges(
        NODE_PLANNER_DECISION,
        route_after_planner_decision,
        {
            NODE_EVIDENCE_SUMMARIZATION: NODE_EVIDENCE_SUMMARIZATION,
            NODE_TOOL_EXECUTOR: NODE_TOOL_EXECUTOR,
            NODE_INTENT_CLARIFICATION: NODE_INTENT_CLARIFICATION,
            NODE_FAILURE: NODE_FAILURE,
        },
    )
    graph.add_conditional_edges(
        NODE_FORMAT_CHECK,
        route_after_format_check,
        {
            NODE_ANSWER: NODE_ANSWER,
            NODE_PLANNER_DECISION: NODE_PLANNER_DECISION,
        },
    )
    graph.add_edge(NODE_TOOL_EXECUTOR, NODE_EVIDENCE_FUSION)
    graph.add_edge(NODE_EVIDENCE_FUSION, NODE_PLANNER_DECISION)
    graph.add_edge(NODE_EVIDENCE_SUMMARIZATION, NODE_FINAL_ANSWER)
    graph.add_edge(NODE_FINAL_ANSWER, NODE_FORMAT_CHECK)
    graph.add_edge(NODE_INTENT_CLARIFICATION, NODE_PLANNER_DECISION)
    graph.add_edge(NODE_ANSWER, END)
    graph.add_edge(NODE_FAILURE, END)

    return graph.compile()


class FrontendFirstAudioAgent(AudioAgent):
    """AudioAgent variant whose graph starts from frontend_evidence_node."""

    def __init__(
        self,
        frontend: BaseFrontend,
        planner: BasePlanner,
        registry: ToolRegistry,
        fuser: BaseEvidenceFuser,
        config: AgentConfig | None = None,
    ) -> None:
        if frontend is None:
            raise ValueError("frontend cannot be None")
        if planner is None:
            raise ValueError("planner cannot be None")
        if registry is None:
            raise ValueError("registry cannot be None")
        if fuser is None:
            raise ValueError("fuser cannot be None")

        self.frontend = frontend
        self.planner = planner
        self.registry = registry
        self.fuser = fuser
        self.config = config or AgentConfig()
        self._temp_dir: str | None = None

        setup_logger()
        if self.config.debug:
            set_debug_mode(True)

        self._graph = build_frontend_first_graph(frontend, planner, registry, fuser)
        self._run_logger: RunLogger | None = None
        if self.config.enable_run_logging:
            self._run_logger = RunLogger(log_dir=self.config.log_dir)


def build_parser() -> argparse.ArgumentParser:
    """Build command-line parser for the ASR demo."""
    parser = argparse.ArgumentParser(
        description=(
            "Run an API ASR agent demo that skips initial_prompt_node and uses "
            f"the fixed prompt {DEFAULT_ASR_PROMPT!r}."
        )
    )
    parser.add_argument(
        "--audio",
        required=True,
        nargs="+",
        help="Path(s) to input audio file(s). Kaldi-style ark offsets are supported.",
    )
    parser.add_argument(
        "--frontend-model",
        default="qwen3.5-omni-plus",
        help="API model name for frontend (default: qwen3.5-omni-plus).",
    )
    parser.add_argument(
        "--planner-model",
        default="qwen3.5-plus",
        help="API model name for planner (default: qwen3.5-plus).",
    )
    parser.add_argument(
        "--api-key",
        default=None,
        help="API key. If not provided, reads from DASHSCOPE_API_KEY or OPENAI_API_KEY.",
    )
    parser.add_argument(
        "--base-url",
        default=DEFAULT_COMPAT_API_BASE_URL,
        help=(
            "OpenAI-compatible base URL for the frontend API; also used for the planner "
            "unless --planner-base-url is set "
            f"(default: {DEFAULT_COMPAT_API_BASE_URL})."
        ),
    )
    parser.add_argument(
        "--planner-api-key",
        default=None,
        help="API key for the planner only. Defaults to --api-key/env key when omitted.",
    )
    parser.add_argument(
        "--planner-base-url",
        default=None,
        help="OpenAI-compatible base URL for the planner only; defaults to --base-url when omitted.",
    )
    parser.add_argument(
        "--enable-thinking",
        action="store_true",
        help="Enable thinking mode for planner models that support it.",
    )
    parser.add_argument(
        "--temperature",
        type=float,
        default=0.05,
        help="Sampling temperature (default: 0.05).",
    )
    parser.add_argument(
        "--max-tokens",
        type=int,
        default=4096,
        help="Maximum tokens to generate (default: 4096).",
    )
    parser.add_argument(
        "--max-steps",
        type=int,
        default=10,
        help="Maximum number of agent steps.",
    )
    parser.add_argument(
        "--tools",
        nargs="+",
        default=None,
        help="Specific catalog tool servers to register. Defaults to non-local-model tools.",
    )
    parser.add_argument(
        "--include-local-model-tools",
        action="store_true",
        help="Also register local model inference tools if their runtimes are installed.",
    )
    parser.add_argument(
        "--list-tools",
        action="store_true",
        help="List available catalog tool servers and exit.",
    )
    parser.add_argument(
        "--external-memory-text",
        default=None,
        help="History transcript text exposed by the dummy external_memory_retrieve tool.",
    )
    parser.add_argument(
        "--external-memory-path",
        default=None,
        help="Path to history transcript text exposed by the dummy external_memory_retrieve tool.",
    )
    return parser


def select_tool_servers(args: argparse.Namespace) -> tuple[list[str], list[str]]:
    """Select catalog tool servers for the ASR graph."""
    if args.tools is not None:
        return args.tools, []

    available_tools = list_available_tools()
    if args.include_local_model_tools:
        return available_tools, []

    selected = [tool for tool in available_tools if tool not in LOCAL_MODEL_TOOL_NAMES]
    excluded = [tool for tool in available_tools if tool in LOCAL_MODEL_TOOL_NAMES]
    return selected, excluded


async def amain() -> int:
    """Run the ASR demo."""
    parser = build_parser()

    if "--list-tools" in sys.argv:
        print("Available MCP tool servers:")
        for tool in list_available_tools():
            print(f"  - {tool}")
        return 0

    args = parser.parse_args()
    selected_tools, excluded_tools = select_tool_servers(args)
    memory_temp_path: str | None = None
    previous_memory_path = os.environ.get("EXTERNAL_MEMORY_PATH")
    previous_memory_text = os.environ.get("EXTERNAL_MEMORY_TEXT")

    api_key = args.api_key or os.environ.get("DASHSCOPE_API_KEY") or os.environ.get("OPENAI_API_KEY")
    planner_api_key = args.planner_api_key or api_key
    if not api_key:
        print("Error: API key required. Provide --api-key or set DASHSCOPE_API_KEY / OPENAI_API_KEY.")
        return 1

    try:
        audio_paths = resolve_audio_input_paths(args.audio)
    except Exception as exc:
        print(f"Failed to resolve audio input: {type(exc).__name__}: {exc}")
        return 1

    if args.external_memory_path:
        os.environ["EXTERNAL_MEMORY_PATH"] = args.external_memory_path
        os.environ.pop("EXTERNAL_MEMORY_TEXT", None)
    elif args.external_memory_text is not None:
        with tempfile.NamedTemporaryFile("w", encoding="utf-8", suffix=".txt", delete=False) as handle:
            handle.write(args.external_memory_text)
            memory_temp_path = handle.name
        os.environ["EXTERNAL_MEMORY_PATH"] = memory_temp_path
        os.environ["EXTERNAL_MEMORY_TEXT"] = args.external_memory_text

    print_separator("Audio Agent Demo (ASR, Frontend-First Graph)")
    print(f"\nPrompt: {DEFAULT_ASR_PROMPT}")
    planner_base_url = args.planner_base_url or args.base_url
    print(f"- {args.frontend_model} frontend (API-based, base URL: {args.base_url})")
    print(f"- {args.planner_model} planner (API-based, base URL: {planner_base_url})")
    print("- Graph starts at frontend_evidence_node; initial_prompt_node is skipped")
    print(f"- Auto-registered MCP tool servers: {', '.join(selected_tools)}")
    if excluded_tools:
        print(f"- Skipped local model tools: {', '.join(excluded_tools)}")
    if len(audio_paths) == 1:
        print(f"Audio path: {audio_paths[0]}")
    else:
        print(f"Audio paths ({len(audio_paths)}):")
        for index, audio_path in enumerate(audio_paths, 1):
            print(f"  [{index}] {audio_path}")

    setup_logger()
    set_debug_mode(True)

    config = AgentConfig(max_steps=args.max_steps, debug=True)
    server_manager = MCPServerManager()

    try:
        frontend = OpenAICompatibleFrontend(
            model=args.frontend_model,
            api_key=api_key,
            base_url=args.base_url,
            temperature=args.temperature,
            max_tokens=args.max_tokens,
        )
        planner = OpenAICompatiblePlanner(
            model=args.planner_model,
            api_key=planner_api_key,
            base_url=planner_base_url,
            enable_thinking=args.enable_thinking,
            temperature=args.temperature,
            max_tokens=args.max_tokens,
        )
        registry = ToolRegistry()
        fuser = DefaultEvidenceFuser()

        print("\nAuto-registering MCP tools from catalog...")
        registered_tools = await register_all_mcp_tools(
            registry=registry,
            server_manager=server_manager,
            tool_names=selected_tools,
            verbose=True,
        )

        if not registered_tools:
            print("\nWarning: No MCP tools were registered.")
            print("Make sure tools are set up:")
            print("  cd audio_agent/tools/catalog/<tool_name> && ./setup.sh")

        agent = FrontendFirstAudioAgent(
            frontend=frontend,
            planner=planner,
            registry=registry,
            fuser=fuser,
            config=config,
        )

        print_separator("Running Agent")
        final_state = await agent.arun(
            question=DEFAULT_ASR_PROMPT,
            audio_paths=audio_paths,
            max_steps=args.max_steps,
            run_log_name="api_asr_frontend_first",
        )
    except Exception as exc:
        print(f"\nDemo failed with error: {type(exc).__name__}: {exc}")
        import traceback

        traceback.print_exc()
        return 1
    finally:
        print("\nShutting down MCP servers...")
        await server_manager.shutdown_all()
        if previous_memory_path is None:
            os.environ.pop("EXTERNAL_MEMORY_PATH", None)
        else:
            os.environ["EXTERNAL_MEMORY_PATH"] = previous_memory_path
        if previous_memory_text is None:
            os.environ.pop("EXTERNAL_MEMORY_TEXT", None)
        else:
            os.environ["EXTERNAL_MEMORY_TEXT"] = previous_memory_text
        if memory_temp_path:
            try:
                Path(memory_temp_path).unlink()
            except OSError:
                pass

    print_separator("Results")
    status = final_state.get("status", AgentStatus.RUNNING)
    print(f"\nFinal Status: {status.value}")
    print(f"Step Count: {final_state.get('step_count', 0)}")

    frontend_output = final_state.get("initial_frontend_output")
    if frontend_output:
        print_separator("Frontend Caption")
        print(f"\n{frontend_output.question_guided_caption}")

    initial_plan = final_state.get("initial_plan")
    if initial_plan:
        print_initial_plan(initial_plan)

    evidence_log = final_state.get("evidence_log", [])
    if evidence_log:
        print_evidence_log(evidence_log)

    tool_history = final_state.get("tool_call_history", [])
    if tool_history:
        print_tool_history(tool_history)

    planner_trace = final_state.get("planner_trace", [])
    if planner_trace:
        print_planner_trace(planner_trace)

    final_answer = final_state.get("final_answer")
    if final_answer:
        print_separator("Final Transcript")
        print(f"\n{final_answer.answer}")
        print(f"\nConfidence: {final_answer.confidence:.2f}")

    error_message = final_state.get("error_message")
    if error_message:
        print_separator("Error")
        print(f"\n{error_message}")

    print_separator()
    if status == AgentStatus.ANSWERED:
        print("\nDemo completed successfully.")
        return 0

    print(f"\nDemo completed with status: {status.value}")
    return 1


def main() -> int:
    """Run the ASR demo entry point."""
    return asyncio.run(amain())


if __name__ == "__main__":
    sys.exit(main())
