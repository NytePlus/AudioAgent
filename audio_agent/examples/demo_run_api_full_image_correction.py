#!/usr/bin/env python3
"""
Demo script for image-guided audio transcription correction.

This demo extends the full API mode flow:
- Analyze one or more images with Qwen VL OCR and Qwen Omni image captioning
- Inject the extracted image context into the audio question
- Run the audio agent with API frontend + API planner

Use this when screenshots, slides, menus, labels, lyric sheets, or document images
can help correct ASR/transcription mistakes in the audio.
"""

from __future__ import annotations

import argparse
import asyncio
import os
import sys
from pathlib import Path

# Add project root to path for direct execution
project_root = Path(__file__).parent.parent.parent
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

from audio_agent.config.settings import AgentConfig
from audio_agent.core.constants import AgentStatus
from audio_agent.core.logging import setup_logger, set_debug_mode
from audio_agent.core.schemas import ToolCallRequest
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
from audio_agent.tools.executor import ToolExecutor
from audio_agent.tools.mcp import MCPServerManager
from audio_agent.tools.registry import ToolRegistry
from audio_agent.utils.audio_path import resolve_audio_input_paths

IMAGE_TOOL_SERVERS = ["qwen_vl_ocr", "image_captioner"]


def build_parser() -> argparse.ArgumentParser:
    """Build command-line parser for the demo."""
    parser = argparse.ArgumentParser(
        description=(
            "Run the full API audio agent with image OCR/caption context to correct "
            "speech recognition mistakes."
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
        required=True,
        nargs="+",
        help="Path(s) to image(s) whose text/visual content should guide ASR correction.",
    )
    parser.add_argument(
        "--question",
        default="Transcribe what is being said. Use the image context to correct likely ASR mistakes.",
        help="Question to ask about the audio.",
    )
    parser.add_argument(
        "--ocr-prompt",
        default=(
            "Extract all readable text from this image. Preserve proper nouns, numbers, "
            "line breaks, labels, and any terms that may be spoken in the audio."
        ),
        help="Prompt for the OCR tool.",
    )
    parser.add_argument(
        "--caption-prompt",
        default=(
            "Describe this image with emphasis on objects, scene context, visible text, "
            "people, slides, diagrams, labels, and terms that could help correct audio transcription."
        ),
        help="Prompt for the image captioning tool.",
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
        default="https://dashscope.aliyuncs.com/compatible-mode/v1",
        help="API base URL.",
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
        help=(
            "Specific catalog tool servers to register. qwen_vl_ocr and image_captioner "
            "are always added for image analysis."
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
    return parser


def select_tool_servers(args: argparse.Namespace) -> tuple[list[str], list[str]]:
    """Select catalog tool servers and ensure image-analysis servers are included."""
    if args.tools is not None:
        selected = list(dict.fromkeys([*args.tools, *IMAGE_TOOL_SERVERS]))
        return selected, []

    available_tools = list_available_tools()
    if args.include_local_model_tools:
        selected = available_tools
        excluded: list[str] = []
    else:
        selected = [tool for tool in available_tools if tool not in LOCAL_MODEL_TOOL_NAMES]
        excluded = [tool for tool in available_tools if tool in LOCAL_MODEL_TOOL_NAMES]

    for tool_name in IMAGE_TOOL_SERVERS:
        if tool_name not in selected:
            selected.append(tool_name)

    return selected, excluded


async def analyze_images(
    registry: ToolRegistry,
    image_paths: list[str],
    ocr_prompt: str,
    caption_prompt: str,
) -> str:
    """Run OCR and caption tools for each image and return formatted context."""
    executor = ToolExecutor(registry)
    sections: list[str] = []

    for index, image_path in enumerate(image_paths, 1):
        print(f"\nAnalyzing image {index}: {image_path}")

        ocr_result = await executor.execute(
            ToolCallRequest(
                tool_name="qwen_vl_ocr",
                args={"image_path": image_path, "prompt": ocr_prompt},
            )
        )
        caption_result = await executor.execute(
            ToolCallRequest(
                tool_name="image_caption",
                args={"image_path": image_path, "prompt": caption_prompt},
            )
        )

        ocr_text = (
            ocr_result.output.get("text", "").strip()
            if ocr_result.success
            else f"OCR failed: {ocr_result.error_message}"
        )
        caption_text = (
            caption_result.output.get("text", "").strip()
            if caption_result.success
            else f"Image caption failed: {caption_result.error_message}"
        )

        sections.append(
            "\n".join(
                [
                    f"## Image {index}: {image_path}",
                    "",
                    "### OCR text",
                    ocr_text or "(empty)",
                    "",
                    "### Visual caption",
                    caption_text or "(empty)",
                ]
            )
        )

    return "\n\n".join(sections)


def build_image_guided_question(original_question: str, image_context: str) -> str:
    """Inject image context into the user question for audio transcription correction."""
    return f"""You are answering an audio question with additional image context.

Original user question:
{original_question}

Image-derived context:
{image_context}

Instructions:
- Use the audio as the primary evidence for what was spoken.
- Use OCR text and visual context from the image to correct likely ASR mistakes, especially
  proper nouns, technical terms, numbers, labels, slide text, menu items, and named entities.
- Do not invent speech from the image alone. If the audio is unclear, say what is uncertain.
- In the final answer, provide the corrected transcription or answer requested by the user,
  and briefly mention the image cues used for correction when relevant.
"""


async def amain() -> int:
    """Run the demo."""
    parser = build_parser()

    if "--list-tools" in sys.argv:
        print("Available MCP tool servers:")
        for tool in list_available_tools():
            print(f"  - {tool}")
        return 0

    args = parser.parse_args()
    selected_tools, excluded_tools = select_tool_servers(args)

    api_key = args.api_key or os.environ.get("DASHSCOPE_API_KEY") or os.environ.get("OPENAI_API_KEY")
    if not api_key:
        print("Error: API key required. Provide --api-key or set DASHSCOPE_API_KEY / OPENAI_API_KEY.")
        return 1

    audio_paths = resolve_audio_input_paths(args.audio)
    image_paths = [str(Path(path).resolve()) for path in args.image]

    print_separator("Audio Agent Demo (Image-Guided ASR Correction)")
    print("\nThis demo runs with:")
    print(f"- {args.frontend_model} frontend (API-based)")
    print(f"- {args.planner_model} planner (API-based)")
    print(f"- Image tools: {', '.join(IMAGE_TOOL_SERVERS)}")
    print(f"- Auto-registered MCP tool servers: {', '.join(selected_tools)}")
    if excluded_tools:
        print(f"- Skipped local model tools: {', '.join(excluded_tools)}")

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
            api_key=api_key,
            base_url=args.base_url,
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

        missing_required = {"qwen_vl_ocr", "image_caption"} - set(registered_tools)
        if missing_required:
            print(f"\nMissing required image tools: {', '.join(sorted(missing_required))}")
            return 1

        print_separator("Image Context Extraction")
        image_context = await analyze_images(
            registry=registry,
            image_paths=image_paths,
            ocr_prompt=args.ocr_prompt,
            caption_prompt=args.caption_prompt,
        )
        print("\n" + image_context)

        image_guided_question = build_image_guided_question(args.question, image_context)

        agent = AudioAgent(
            frontend=frontend,
            planner=planner,
            registry=registry,
            fuser=fuser,
            config=config,
        )

        print_separator("Running Agent")
        print(f"\nOriginal question: {args.question}")
        print(f"Audio paths: {audio_paths}")
        print(f"Image paths: {image_paths}")

        final_state = await agent.arun(
            question=image_guided_question,
            audio_paths=audio_paths,
            max_steps=args.max_steps,
        )

    except Exception as exc:
        print(f"\nDemo failed with error: {type(exc).__name__}: {exc}")
        import traceback

        traceback.print_exc()
        return 1
    finally:
        print("\nShutting down MCP servers...")
        await server_manager.shutdown_all()

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
