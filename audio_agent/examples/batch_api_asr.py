#!/usr/bin/env python3
"""Run the API ASR demo over a JSONL manifest and write id/pred lines."""

from __future__ import annotations

import argparse
import asyncio
import json
import os
import re
import sys
from pathlib import Path
from typing import Any

from tqdm import tqdm


SOURCE_PREFIX = "/aistor/sjtu/hpc_stor01/home/wangchencheng/data/slidespeech"
TARGET_PREFIX = "/data"


def print_subprocess_failure(
    key: str,
    returncode: int | None,
    timed_out: bool,
    log_path: Path,
    output: str,
    *,
    max_snippet_chars: int = 4000,
) -> None:
    """Echo subprocess failure to stderr (full output is still in log_path)."""
    status = "timeout" if timed_out else f"exit_code={returncode}"
    print(f"[batch_api_asr] key={key} {status} log={log_path}", file=sys.stderr)
    text = output.rstrip()
    if not text:
        return
    if len(text) > max_snippet_chars:
        text = f"...(truncated, see log)\n{text[-max_snippet_chars:]}"
    print(text, file=sys.stderr)
    print("", file=sys.stderr)


def build_parser() -> argparse.ArgumentParser:
    """Build command-line arguments."""
    parser = argparse.ArgumentParser(description="Batch-run demo_run_api_asr over a JSONL file.")
    parser.add_argument("--input", default="/data/dev_oracle_v1/multitask.jsonl")
    parser.add_argument("--output", default="pred.txt")
    parser.add_argument("--log-dir", default="logs/batch_api_asr")
    parser.add_argument("--frontend-model", default="qwen3-omni-flash")
    parser.add_argument("--planner-model", default="qwen3.5-plus")
    parser.add_argument("--planner-api-key", default=None, help="Passed through to demo_run_api_asr --planner-api-key (planner endpoint only).")
    parser.add_argument("--planner-base-url", default=None, help="Passed through to demo_run_api_asr --planner-base-url (planner endpoint only).",)
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--retries", type=int, default=1)
    parser.add_argument("--timeout", type=float, default=0.0, help="Per-item timeout in seconds; 0 disables.")
    parser.add_argument("--max-steps", type=int, default=10)
    parser.add_argument("--limit", type=int, default=None, help="Optional limit for smoke tests.")
    parser.add_argument("--no-progress", action="store_true", help="Disable the tqdm progress bar.")
    parser.add_argument(
        "--history-field",
        default="history",
        help=(
            "JSONL field containing historical transcript text for the dummy external "
            "memory tool. Set empty to disable field lookup."
        ),
    )
    parser.add_argument(
        "--history-file",
        default=None,
        help=(
            "Optional JSONL/text file with historical transcript memory. JSONL can contain "
            "key/history records; plain text is passed to every demo run."
        ),
    )
    return parser


def load_entries(input_path: Path, limit: int | None) -> list[dict[str, Any]]:
    """Load and normalize manifest entries."""
    entries: list[dict[str, Any]] = []
    with input_path.open("r", encoding="utf-8") as handle:
        for line in handle:
            if not line.strip():
                continue
            item = json.loads(line)
            item["path"] = item["path"].replace(SOURCE_PREFIX, TARGET_PREFIX, 1)
            entries.append(item)
            if limit is not None and len(entries) >= limit:
                break
    return entries


def load_history_map(history_file: str | None, history_field: str) -> tuple[dict[str, str], str | None]:
    """Load optional external memory from a JSONL or plain-text file."""
    if not history_file:
        return {}, None

    path = Path(history_file)
    text = path.read_text(encoding="utf-8")
    history_by_key: dict[str, str] = {}
    parsed_jsonl = False
    for line in text.splitlines():
        if not line.strip():
            continue
        try:
            item = json.loads(line)
        except json.JSONDecodeError:
            parsed_jsonl = False
            break
        parsed_jsonl = True
        key = item.get("key") or item.get("id") or item.get("utt_id")
        history = item.get(history_field) if history_field else None
        if key is not None and history is not None:
            history_by_key[str(key)] = str(history)

    if parsed_jsonl and history_by_key:
        return history_by_key, None
    return {}, text


def get_entry_history(
    entry: dict[str, Any],
    args: argparse.Namespace,
    history_by_key: dict[str, str],
    global_history: str | None,
) -> str | None:
    """Return history text for one manifest entry."""
    key = str(entry["key"])
    if key in history_by_key:
        return history_by_key[key]
    if global_history is not None:
        return global_history
    if args.history_field:
        history = entry.get(args.history_field)
        if history is not None:
            return str(history)
    return None


def compact_prediction(prediction: str) -> str:
    """Keep each pred.txt record on one line."""
    return " ".join(prediction.strip().split())


def load_existing_predictions(output_path: Path) -> dict[str, str]:
    """Read an existing pred.txt-style file into key -> prediction."""
    if not output_path.exists():
        return {}

    predictions: dict[str, str] = {}
    with output_path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            stripped = line.strip()
            if not stripped:
                continue
            parts = stripped.split(maxsplit=1)
            if len(parts) != 2:
                print(
                    f"[batch_api_asr] Skipping malformed prediction line "
                    f"{output_path}:{line_number}: {stripped!r}",
                    file=sys.stderr,
                )
                continue
            key, prediction = parts
            predictions[key] = prediction
    return predictions


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


def parse_prediction(output: str) -> str:
    """Extract the transcript from demo output."""
    final_section = output
    if "Final Transcript" in output:
        final_section = output.split("Final Transcript", 1)[1]
    if "Confidence:" in final_section:
        final_section = final_section.split("Confidence:", 1)[0]

    fenced_match = re.search(r"```(?:json)?\s*(.*?)\s*```", final_section, re.DOTALL)
    candidate = fenced_match.group(1) if fenced_match else final_section
    json_text = extract_balanced_json(candidate)
    if json_text:
        parsed = json.loads(json_text)
        for key in ("pred", "transcript", "answer"):
            value = parsed.get(key)
            if isinstance(value, str) and value.strip():
                return compact_prediction(value)
        raise ValueError(f"JSON answer did not contain a usable pred field: {json_text}")

    stripped = final_section.strip()
    if stripped:
        return compact_prediction(stripped)
    raise ValueError("Could not extract prediction from demo output")


def write_predictions(
    output_path: Path,
    entries: list[dict[str, Any]],
    predictions: dict[str, str],
) -> None:
    """Atomically write predictions completed so far in manifest order."""
    tmp_path = output_path.with_suffix(output_path.suffix + ".tmp")
    with tmp_path.open("w", encoding="utf-8") as handle:
        for entry in entries:
            key = str(entry["key"])
            if key in predictions:
                handle.write(f"{key} {predictions[key]}\n")
    tmp_path.replace(output_path)


async def run_entry(
    entry: dict[str, Any],
    args: argparse.Namespace,
    log_dir: Path,
    history_text: str | None,
) -> tuple[str, str, bool]:
    """Run one manifest entry and return key, prediction, success."""
    key = str(entry["key"])
    command = [
        sys.executable,
        "-m",
        "audio_agent.examples.demo_run_api_asr",
        "--audio",
        str(entry["path"]),
        "--frontend-model",
        args.frontend_model,
        "--planner-model",
        args.planner_model,
        "--max-steps",
        str(args.max_steps),
    ]
    if args.planner_api_key:
        command.extend(["--planner-api-key", args.planner_api_key])
    if args.planner_base_url:
        command.extend(["--planner-base-url", args.planner_base_url])
    if history_text is not None:
        history_path = log_dir / f"{key}.history.txt"
        history_path.write_text(history_text, encoding="utf-8")
        command.extend(["--external-memory-path", str(history_path)])

    log_path = log_dir / f"{key}.log"
    last_output = ""
    for attempt in range(args.retries + 1):
        process = await asyncio.create_subprocess_exec(
            *command,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.STDOUT,
        )
        timed_out = False
        try:
            if args.timeout > 0:
                stdout_bytes, _ = await asyncio.wait_for(process.communicate(), args.timeout)
            else:
                stdout_bytes, _ = await process.communicate()
        except asyncio.TimeoutError:
            timed_out = True
            process.kill()
            stdout_bytes, _ = await process.communicate()
            last_output = stdout_bytes.decode("utf-8", errors="replace")
            last_output += f"\nTimed out after {args.timeout} seconds.\n"
        else:
            last_output = stdout_bytes.decode("utf-8", errors="replace")

        attempt_header = f"===== attempt {attempt + 1}/{args.retries + 1} =====\n"
        if attempt == 0:
            log_path.write_text(attempt_header + last_output, encoding="utf-8")
        else:
            with log_path.open("a", encoding="utf-8") as handle:
                handle.write("\n" + attempt_header + last_output)

        rc = process.returncode
        if timed_out or rc != 0:
            print_subprocess_failure(key, rc, timed_out, log_path, last_output)

        if process.returncode == 0:
            try:
                return key, parse_prediction(last_output), True
            except Exception as exc:  # noqa: BLE001
                parse_error = f"\nPrediction parse error: {type(exc).__name__}: {exc}\n"
                with log_path.open("a", encoding="utf-8") as handle:
                    handle.write(parse_error)

    return key, "", False


async def run_batch(args: argparse.Namespace) -> int:
    """Run all entries with bounded concurrency."""
    input_path = Path(args.input)
    output_path = Path(args.output)
    log_dir = Path(args.log_dir)
    log_dir.mkdir(parents=True, exist_ok=True)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    entries = load_entries(input_path, args.limit)
    history_by_key, global_history = load_history_map(args.history_file, args.history_field)
    predictions = load_existing_predictions(output_path)
    pending_entries = [entry for entry in entries if str(entry["key"]) not in predictions]
    failures: list[str] = []
    semaphore = asyncio.Semaphore(max(1, args.workers))
    progress_lock = asyncio.Lock()
    progress = tqdm(
        total=len(pending_entries),
        desc="ASR",
        unit="utt",
        dynamic_ncols=True,
        disable=args.no_progress,
    )

    async def worker(entry: dict[str, Any]) -> None:
        async with semaphore:
            history_text = get_entry_history(entry, args, history_by_key, global_history)
            key, prediction, ok = await run_entry(entry, args, log_dir, history_text)
            if ok:
                predictions[key] = prediction
            else:
                failures.append(key)
            write_predictions(output_path, entries, predictions)
            async with progress_lock:
                progress.update(1)
                progress.set_postfix(
                    ok=len(predictions),
                    failed=len(failures),
                    last=key,
                    refresh=True,
                )

    try:
        await asyncio.gather(*(worker(entry) for entry in pending_entries))
    finally:
        progress.close()
    write_predictions(output_path, entries, predictions)

    if failures:
        failure_path = output_path.with_suffix(output_path.suffix + ".failures")
        failure_path.write_text("\n".join(failures) + "\n", encoding="utf-8")
        print(f"Finished with {len(failures)} failures. See {failure_path}.", file=sys.stderr)
        return 1

    skipped = len(entries) - len(pending_entries)
    print(f"Wrote {len(predictions)} predictions to {output_path}. Skipped {skipped} existing ids.")
    return 0


def main() -> int:
    """Entry point."""
    args = build_parser().parse_args()
    if not os.environ.get("DASHSCOPE_API_KEY") and not os.environ.get("OPENAI_API_KEY"):
        print("DASHSCOPE_API_KEY or OPENAI_API_KEY is required.", file=sys.stderr)
        return 1
    return asyncio.run(run_batch(args))


if __name__ == "__main__":
    raise SystemExit(main())
