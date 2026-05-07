#!/usr/bin/env python3
"""
Demo script for image-guided audio transcription correction.

This demo extends the full API mode flow:
- Registers Qwen VL OCR and Qwen Omni image captioning tools
- Passes images to the agent as first-class `image_0`, `image_1`, ... inputs
- Run the audio agent with API frontend + API planner

Use this when screenshots, slides, menus, labels, lyric sheets, or document images
can help correct ASR/transcription mistakes in the audio.
"""

from __future__ import annotations

import argparse
import asyncio
import json
import os
import sys
import tempfile
import time
from pathlib import Path

# Add project root to path for direct execution
project_root = Path(__file__).parent.parent.parent
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

from audio_agent.config.settings import AgentConfig
from audio_agent.core.constants import AgentStatus
from audio_agent.core.logging import setup_logger, set_debug_mode
from audio_agent.core.schemas import FinalAnswer
from audio_agent.critic.dummy_critic import DummyCritic
from audio_agent.critic.openai_compatible_critic import OpenAICompatibleCritic
from audio_agent.examples.demo_run_api_full import (
    LOCAL_MODEL_TOOL_NAMES,
    print_evidence_log,
    print_initial_plan,
    print_planner_trace,
    print_separator,
    print_tool_history,
)
from audio_agent.frontend.openai_compatible_frontend import OpenAICompatibleFrontend
from audio_agent.fusion.default_fuser import DefaultEvidenceFuser
from audio_agent.main import AudioAgent
from audio_agent.planner.openai_compatible_planner import OpenAICompatiblePlanner
from audio_agent.tools.catalog import list_available_tools, register_all_mcp_tools
from audio_agent.tools.mcp import MCPServerManager
from audio_agent.tools.registry import ToolRegistry
from audio_agent.utils.audio_path import resolve_audio_input_paths

IMAGE_TOOL_SERVERS = ["qwen_vl_ocr", "image_captioner", "image_qa"]
ABLATION_MODES = ("full", "frontend-only", "no-critic-no-image-tools", "no-critic")


def _agent_debug_log(hypothesis_id: str, message: str, data: dict) -> None:
    """Write one debug NDJSON entry for the active Cursor debug session."""
    # region agent log
    payload = {
        "sessionId": "b99362",
        "runId": "pre-fix",
        "hypothesisId": hypothesis_id,
        "location": "audio_agent/examples/demo_run_api_full_image_correction.py",
        "message": message,
        "data": data,
        "timestamp": int(time.time() * 1000),
    }
    try:
        with open("/workspace/.cursor/debug-b99362.log", "a", encoding="utf-8") as handle:
            handle.write(json.dumps(payload, ensure_ascii=False, default=str) + "\n")
    except Exception:
        pass
    # endregion agent log


def build_parser() -> argparse.ArgumentParser:
    """Build command-line parser for the demo."""
    parser = argparse.ArgumentParser(
        description=(
            "Run the full API audio agent with first-class reference images for "
            "OCR/caption-guided speech recognition correction."
        )
    )
    parser.add_argument(
        "--audio",
        required=True,
        nargs="+",
        help="Path(s) to the input audio file(s).",
    )
    parser.add_argument(
        "--image",
        default=[],
        nargs="+",
        help="Path(s) to image(s) whose text/visual content should guide ASR correction.",
    )
    parser.add_argument(
        "--question",
        default='Transcribe what is being said. Use the image context to correct likely ASR mistakes. Return only valid JSON with exactly this schema: {"transcription": "<corrected transcript text>"}',
        help="Question to ask about the audio.",
    )
    parser.add_argument(
        "--frontend-model",
        default="qwen2.5-omni-7b",
        help="API model name for frontend (default: qwen2.5-omni-7b).",
    )
    parser.add_argument(
        "--planner-model",
        default="qwen3.5-plus",
        help="API model name for planner (default: qwen3.5-plus).",
    )
    parser.add_argument(
        "--api-key",
        default=None,
        help=(
            "Frontend API key. If not provided, reads API_KEY, "
            "DASHSCOPE_API_KEY, or OPENAI_API_KEY."
        ),
    )
    parser.add_argument(
        "--base-url",
        default=None,
        help="Frontend API base URL. If not provided, reads BASE_URL or uses DashScope.",
    )
    parser.add_argument(
        "--planner-api-key",
        default=None,
        help="Planner/critic API key. Defaults to PLANNER_API_KEY or the frontend API key.",
    )
    parser.add_argument(
        "--planner-base-url",
        default=None,
        help="Planner/critic API base URL. Defaults to PLANNER_BASE_URL or the frontend base URL.",
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
        help="Maximum tokens for planner/critic generation (default: 4096).",
    )
    parser.add_argument(
        "--frontend-max-tokens",
        "--frontentend-max-tokens",
        dest="frontend_max_tokens",
        type=int,
        default=None,
        help=(
            "Maximum tokens for frontend generation. Defaults to --max-tokens. "
            "The misspelled --frontentend-max-tokens alias is kept for compatibility."
        ),
    )
    parser.add_argument(
        "--max-steps",
        type=int,
        default=10,
        help="Maximum number of agent steps.",
    )
    parser.add_argument(
        "--ablation",
        choices=ABLATION_MODES,
        default="full",
        help=(
            "Ablation mode: full keeps the complete image-guided agent; "
            "frontend-only skips planner/tools/critic and returns the frontend caption; "
            "no-critic-no-image-tools disables the critic and removes qwen_vl_ocr, "
            "image_captioner, and image_qa; no-critic disables only the real critic."
        ),
    )
    parser.add_argument(
        "--tools",
        nargs="+",
        default=None,
        help=(
            "Specific catalog tool servers to register. qwen_vl_ocr and image_captioner "
            "are added for image analysis unless --ablation=no-critic-no-image-tools."
        ),
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
        help="History transcript text exposed by the external_memory_retrieve tool.",
    )
    parser.add_argument(
        "--external-memory-path",
        default=None,
        help="Path to history transcript text exposed by the external_memory_retrieve tool.",
    )
    return parser


def select_tool_servers(args: argparse.Namespace) -> tuple[list[str], list[str]]:
    """Select catalog tool servers and ensure image-analysis servers are included."""
    if args.ablation == "frontend-only":
        return [], []

    if args.tools is not None:
        selected = list(dict.fromkeys(args.tools))
        if args.ablation != "no-critic-no-image-tools":
            selected = list(dict.fromkeys([*selected, *IMAGE_TOOL_SERVERS]))
        else:
            selected = [tool for tool in selected if tool not in IMAGE_TOOL_SERVERS]
        return selected, []

    available_tools = list_available_tools()
    if args.include_local_model_tools:
        selected = available_tools
        excluded: list[str] = []
    else:
        selected = [tool for tool in available_tools if tool not in LOCAL_MODEL_TOOL_NAMES]
        excluded = [tool for tool in available_tools if tool in LOCAL_MODEL_TOOL_NAMES]

    if args.ablation == "no-critic-no-image-tools":
        selected = [tool for tool in selected if tool not in IMAGE_TOOL_SERVERS]
        excluded = list(dict.fromkeys([*excluded, *IMAGE_TOOL_SERVERS]))
    else:
        for tool_name in IMAGE_TOOL_SERVERS:
            if tool_name not in selected:
                selected.append(tool_name)

    return selected, excluded


def extract_balanced_json(text: str) -> str | None:
    """Return the first balanced JSON object in text."""
    start = text.find("{")
    if start < 0:
        return None

    depth = 0
    in_string = False
    escape = False
    for index in range(start, len(text)):
        char = text[index]
        if in_string:
            if escape:
                escape = False
            elif char == "\\":
                escape = True
            elif char == '"':
                in_string = False
            continue
        if char == '"':
            in_string = True
        elif char == "{":
            depth += 1
        elif char == "}":
            depth -= 1
            if depth == 0:
                return text[start : index + 1]
    return None


def normalize_transcription_json(answer_text: str) -> str:
    """Normalize frontend-only JSON output to a transcription object."""
    json_text = extract_balanced_json(answer_text)
    if not answer_text.strip():
        raise ValueError("Frontend-only final answer was empty.")
    if not json_text:
        raise ValueError(
            "Frontend-only final answer was not JSON. "
            f"Raw output starts with: {answer_text.strip()[:200]!r}"
        )

    parsed = json.loads(json_text)
    if not isinstance(parsed, dict):
        raise ValueError("Frontend-only JSON answer must be an object.")

    for key in ("transcription", "transcript", "pred", "answer"):
        value = parsed.get(key)
        if isinstance(value, str) and value.strip():
            return json.dumps({"transcription": value.strip()}, ensure_ascii=False)

    raise ValueError(
        "Frontend-only JSON answer did not contain a non-empty transcription field. "
        f"Available keys: {sorted(parsed.keys())}"
    )


def run_frontend_only(
    frontend: OpenAICompatibleFrontend,
    question: str,
    audio_paths: list[str],
) -> dict:
    """Run only frontend model calls and return a JSON transcription answer."""
    frontend_output = frontend.run(question, audio_paths, question_oriented_prompt=None)
    answer_text = frontend.generate_final_answer(
        question=question,
        audio_paths=audio_paths,
        context={
            "initial_frontend_output": frontend_output,
            "expected_output_format": '{"transcription": "<transcript text>"}',
            "evidence_log": [],
            "planner_trace": [],
            "tool_call_history": [],
            "audio_list": [],
        },
    )
    normalized_answer = normalize_transcription_json(answer_text)
    return {
        "status": AgentStatus.ANSWERED,
        "step_count": 0,
        "initial_frontend_output": frontend_output,
        "final_answer": FinalAnswer(answer=normalized_answer, confidence=1.0),
    }


async def amain() -> int:
    """Run the demo."""
    parser = build_parser()

    if "--list-tools" in sys.argv:
        print("Available MCP tool servers:")
        for tool in list_available_tools():
            print(f"  - {tool}")
        return 0

    args = parser.parse_args()
    memory_temp_path: str | None = None
    previous_memory_path = os.environ.get("EXTERNAL_MEMORY_PATH")
    previous_memory_text = os.environ.get("EXTERNAL_MEMORY_TEXT")

    selected_tools, excluded_tools = select_tool_servers(args)
    _agent_debug_log(
        "H2",
        "selected_tool_servers",
        {
            "selected_tools": selected_tools,
            "excluded_tools": excluded_tools,
            "image_tool_servers": IMAGE_TOOL_SERVERS,
        },
    )

    frontend_api_key = (
        args.api_key or os.environ.get("DASHSCOPE_API_KEY")
    )
    frontend_base_url = args.base_url or os.environ.get("DASHSCOPE_BASE_URL") or "https://dashscope.aliyuncs.com/compatible-mode/v1"
    planner_api_key = args.planner_api_key or os.environ.get("LANGUAGE_MODEL_API_KEY") or frontend_api_key
    planner_base_url = (
        args.planner_base_url or os.environ.get("LANGUAGE_MODEL_BASE_URL") or frontend_base_url
    )
    if not frontend_api_key:
        print(
            "Error: frontend API key required. Provide --api-key or set API_KEY, "
            "DASHSCOPE_API_KEY, or OPENAI_API_KEY.",
            file=sys.stderr,
        )
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

    audio_paths = resolve_audio_input_paths(args.audio)
    image_paths = [str(Path(path).resolve()) for path in args.image]

    print_separator("Audio Agent Demo (Image-Guided ASR Correction)")
    print("\nThis demo runs with:")
    print(f"- Ablation mode: {args.ablation}")
    print(f"- {args.frontend_model} frontend (API-based)")
    frontend_max_tokens = args.frontend_max_tokens or args.max_tokens
    if args.frontend_max_tokens is not None:
        print(f"- Frontend max tokens: {frontend_max_tokens}")
    if args.ablation != "frontend-only":
        print(f"- {args.planner_model} planner (API-based)")
        if planner_base_url != frontend_base_url:
            print(f"- Planner base URL: {planner_base_url}")
    critic_enabled = args.ablation == "full"
    print(f"- Critic: {'enabled' if critic_enabled else 'disabled'}")
    print(f"- Image tools: {'enabled' if args.ablation != 'no-critic-no-image-tools' else 'disabled'}")
    print(f"- Auto-registered MCP tool servers: {', '.join(selected_tools) if selected_tools else '(none)'}")
    if excluded_tools:
        print(f"- Skipped local model tools: {', '.join(excluded_tools)}")

    setup_logger()
    set_debug_mode(True)

    config = AgentConfig(max_steps=args.max_steps, debug=True)
    server_manager = MCPServerManager()

    try:
        frontend = OpenAICompatibleFrontend(
            model=args.frontend_model,
            api_key=frontend_api_key,
            base_url=frontend_base_url,
            temperature=args.temperature,
            max_tokens=frontend_max_tokens,
        )
        if args.ablation == "frontend-only":
            print_separator("Running Frontend Only")
            print(f"\nOriginal question: {args.question}")
            print(f"Audio paths: {audio_paths}")
            final_state = run_frontend_only(frontend, args.question, audio_paths)
        else:
            planner = OpenAICompatiblePlanner(
                model=args.planner_model,
                api_key=planner_api_key,
                base_url=planner_base_url,
                enable_thinking=args.enable_thinking,
                temperature=args.temperature,
                max_tokens=args.max_tokens,
            )
            critic = (
                OpenAICompatibleCritic(
                    model=args.planner_model,
                    api_key=planner_api_key,
                    base_url=planner_base_url,
                    temperature=0.0,
                    max_tokens=args.max_tokens,
                )
                if critic_enabled
                else DummyCritic()
            )
        registry = ToolRegistry()
        fuser = DefaultEvidenceFuser()

        if args.ablation != "frontend-only":
            print("\nAuto-registering MCP tools from catalog...")
            registered_tools = await register_all_mcp_tools(
                registry=registry,
                server_manager=server_manager,
                tool_names=selected_tools,
                verbose=True,
            )
            _agent_debug_log(
                "H2,H3",
                "registered_tools_after_catalog_registration",
                {
                    "registered_tools": registered_tools,
                    "registry_names": registry.list_names(),
                    "has_image_qa": "image_qa" in registry,
                    "has_kw_verify": "kw_verify" in registry,
                    "has_external_memory": "external_memory_retrieve" in registry,
                },
            )

            if args.ablation != "no-critic-no-image-tools" and image_paths:
                missing_required = {"qwen_vl_ocr", "image_caption", "image_qa"} - set(registered_tools)
                if missing_required:
                    _agent_debug_log(
                        "H2,H4",
                        "exiting_before_agent_due_missing_required_tools",
                        {
                            "missing_required": sorted(missing_required),
                            "registered_tools": registered_tools,
                            "will_call_agent_arun": False,
                        },
                    )
                    print(f"\nMissing required image tools: {', '.join(sorted(missing_required))}")
                    return 1

            if image_paths:
                print_separator("Image Inputs Registered")
                for index, image_path in enumerate(image_paths):
                    print(f"- image_{index}: {image_path}")

            agent = AudioAgent(
                frontend=frontend,
                planner=planner,
                registry=registry,
                fuser=fuser,
                config=config,
                critic=critic,
            )

            print_separator("Running Agent")
            _agent_debug_log(
                "H4",
                "about_to_call_agent_arun",
                {
                    "will_call_agent_arun": True,
                    "audio_count": len(audio_paths),
                    "image_count": len(image_paths),
                    "ablation": args.ablation,
                },
            )
            print(f"\nOriginal question: {args.question}")
            print(f"Audio paths: {audio_paths}")
            if image_paths:
                print(f"Image paths: {image_paths}")

            final_state = await agent.arun(
                question=args.question,
                audio_paths=audio_paths,
                max_steps=args.max_steps,
                image_paths=image_paths,
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
        print_separator("Final Answer")
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
    """Run the demo entry point."""
    return asyncio.run(amain())


if __name__ == "__main__":
    sys.exit(main())
