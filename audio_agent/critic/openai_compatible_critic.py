"""OpenAI-compatible implementation of the final-answer critic."""

from __future__ import annotations

import asyncio
import json
import os
import re
import unicodedata
from typing import Any

from audio_agent.critic.base import BaseCritic
from audio_agent.core.errors import CriticError
from audio_agent.core.schemas import (
    CriticCheckResult,
    CriticResult,
    ToolCallRequest,
    TranscriptEdit,
)
from audio_agent.core.state import AgentState
from audio_agent.tools.executor import ToolExecutor
from audio_agent.tools.registry import ToolRegistry
from audio_agent.utils.prompt_io import load_prompt
from audio_agent.utils.transcript import extract_transcript_text


class OpenAICompatibleCritic(BaseCritic):
    """LLM-backed critic using an OpenAI-compatible chat completions API."""

    def __init__(
        self,
        model: str,
        api_key: str | None = None,
        base_url: str | None = None,
        api_key_env: str = "OPENAI_API_KEY",
        temperature: float = 0.0,
        max_tokens: int = 4096,
        timeout: float = 120.0,
    ) -> None:
        self._model = model
        self._api_key = api_key
        self._base_url = base_url
        self._api_key_env = api_key_env
        self._temperature = temperature
        self._max_tokens = max_tokens
        self._timeout = timeout
        self._client = self._initialize_client()

    @property
    def name(self) -> str:
        return f"openai_compatible_critic_{self._model}"

    async def critique(
        self,
        state: AgentState,
        executor: ToolExecutor,
        registry: ToolRegistry,
    ) -> CriticResult:
        decision = state.get("current_decision")
        proposed_answer = decision.draft_answer if decision else None
        if not proposed_answer:
            raise CriticError("Critic requires current_decision.draft_answer")

        final_transcript = extract_transcript_text(proposed_answer) or ""
        initial_transcript = state.get("initial_transcript")

        format_task = asyncio.create_task(
            self._run_format_check(state=state, proposed_answer=proposed_answer)
        )
        acoustic_task = asyncio.create_task(
            self._run_acoustic_check(
                initial_transcript=initial_transcript,
                final_transcript=final_transcript,
                state=state,
                executor=executor,
                registry=registry,
            )
        )
        image_task = asyncio.create_task(
            self._run_image_check(
                final_answer=proposed_answer,
                state=state,
                executor=executor,
                registry=registry,
            )
        )
        history_task = asyncio.create_task(
            self._run_history_check(
                final_transcript=final_transcript,
                final_answer=proposed_answer,
                state=state,
                executor=executor,
                registry=registry,
            )
        )

        format_check, acoustic_result, image_check, history_check = await asyncio.gather(
            format_task,
            acoustic_task,
            image_task,
            history_task,
        )
        edits, kw_check = acoustic_result
        checks = [format_check, kw_check, image_check, history_check]

        passed = self._passes_critic_rules(
            format_check=format_check,
            kw_check=kw_check,
            image_check=image_check,
            history_check=history_check,
        )
        reject_reasons = self._collect_reject_reasons(
            [format_check, kw_check, image_check, history_check]
        )
        reject_reason = None if passed else "\n".join(reason for reason in reject_reasons if reason)
        return CriticResult(
            passed=passed,
            critique=reject_reason,
            reject_reason=reject_reason,
            confidence=min((check.confidence for check in checks), default=0.0),
            checks=checks,
            transcript_edits=edits,
        )

    def _initialize_client(self):
        try:
            from openai import OpenAI
        except ImportError as exc:
            raise CriticError("Missing openai package. Install with: pip install openai") from exc

        api_key = self._api_key or os.environ.get(self._api_key_env)
        if not api_key:
            for alt_env in ["DASHSCOPE_API_KEY", "OPENAI_API_KEY"]:
                api_key = os.environ.get(alt_env)
                if api_key:
                    break
        if not api_key:
            raise CriticError(
                f"API key required for critic model {self._model}",
                details={"env": self._api_key_env},
            )

        kwargs: dict[str, Any] = {"api_key": api_key, "timeout": self._timeout}
        if self._base_url:
            kwargs["base_url"] = self._base_url
        return OpenAI(**kwargs)

    def _call_json(self, system_prompt: str, user_payload: dict[str, Any]) -> dict[str, Any]:
        try:
            response = self._client.chat.completions.create(
                model=self._model,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": json.dumps(user_payload, ensure_ascii=False)},
                ],
                temperature=self._temperature,
                max_tokens=self._max_tokens,
            )
        except Exception as exc:
            raise CriticError(f"Critic API call failed: {exc}") from exc

        if not response.choices:
            raise CriticError("Critic API returned no choices")
        content = response.choices[0].message.content or ""
        return self._parse_json_object(content)

    @staticmethod
    def _parse_json_object(text: str) -> dict[str, Any]:
        stripped = text.strip()
        if stripped.startswith("```"):
            stripped = stripped.strip("`").strip()
            if stripped.lower().startswith("json"):
                stripped = stripped[4:].strip()
        start = stripped.find("{")
        end = stripped.rfind("}")
        if start >= 0 and end >= start:
            stripped = stripped[start : end + 1]
        try:
            parsed = json.loads(stripped)
        except json.JSONDecodeError as exc:
            raise CriticError("Critic returned malformed JSON", details={"text": text[:500]}) from exc
        if not isinstance(parsed, dict):
            raise CriticError("Critic JSON output must be an object")
        return parsed

    async def _run_format_check(
        self,
        *,
        state: AgentState,
        proposed_answer: str,
    ) -> CriticCheckResult:
        return self._check_format(state, proposed_answer)

    async def _run_acoustic_check(
        self,
        *,
        initial_transcript: str | None,
        final_transcript: str,
        state: AgentState,
        executor: ToolExecutor,
        registry: ToolRegistry,
    ) -> tuple[list[TranscriptEdit], CriticCheckResult]:
        edits = await asyncio.to_thread(
            self._extract_edits,
            initial_transcript,
            final_transcript,
        )
        kw_check = await self._run_kw_verify_check(
            edits=edits,
            state=state,
            executor=executor,
            registry=registry,
        )
        return edits, kw_check

    def _check_format(self, state: AgentState, proposed_answer: str) -> CriticCheckResult:
        pattern = r'"transcription"\s*:'
        match = re.search(pattern, proposed_answer)
        parsed: dict[str, Any] | None = None
        parse_error = None
        if match:
            try:
                parsed = json.loads(proposed_answer)
            except json.JSONDecodeError as exc:
                parse_error = str(exc)

        passed = (
            isinstance(parsed, dict)
            and "transcription" in parsed
            and isinstance(parsed.get("transcription"), str)
        )
        if passed:
            reject_reason = None
            reason = "Format is valid."
            confidence = 1.0
        elif not match:
            reject_reason = "invalid format"
            reason = 'Final answer must be a JSON object containing a string "transcription" field.'
            confidence = 0.0
        else:
            reject_reason = "invalid JSON"
            reason = f"Final answer contained a transcription field but could not be parsed as JSON: {parse_error}"
            confidence = 0.0
        return CriticCheckResult(
            name="format",
            passed=passed,
            critique=reject_reason,
            reject_reason=reject_reason,
            confidence=confidence,
            metadata={
                "regex_matched": bool(match),
                "parsed": parsed,
                "reason": reason,
            },
        )

    def _extract_edits(
        self,
        initial_transcript: str | None,
        final_transcript: str,
    ) -> list[TranscriptEdit]:
        if not initial_transcript or not final_transcript:
            return []
        original_tokens = self._tokenize_transcript(initial_transcript)
        revised_tokens = self._tokenize_transcript(final_transcript)
        if not original_tokens or not revised_tokens:
            return []

        ops = self._align_tokens(original_tokens, revised_tokens)
        edits: list[TranscriptEdit] = []
        original_buffer: list[str] = []
        revised_buffer: list[str] = []
        operation_buffer: list[str] = []

        def flush_edit() -> None:
            if not operation_buffer:
                return
            raw_original_text = " ".join(original_buffer).strip()
            raw_revised_text = " ".join(revised_buffer).strip()
            original_text = raw_original_text or "<inserted>"
            revised_text = raw_revised_text or "<deleted>"
            if self._normalize_for_edit_compare(raw_original_text) != self._normalize_for_edit_compare(
                raw_revised_text
            ):
                edits.append(
                    TranscriptEdit(
                        original_text=original_text,
                        revised_text=revised_text,
                        rationale=(
                            "Detected by edit-distance alignment: "
                            f"{', '.join(dict.fromkeys(operation_buffer))}"
                        ),
                    )
                )
            original_buffer.clear()
            revised_buffer.clear()
            operation_buffer.clear()

        for operation, original_token, revised_token in ops:
            if operation == "equal":
                flush_edit()
                continue
            if original_token is not None:
                original_buffer.append(original_token)
            if revised_token is not None:
                revised_buffer.append(revised_token)
            operation_buffer.append(operation)
        flush_edit()
        return edits

    @staticmethod
    def _tokenize_transcript(text: str) -> list[str]:
        return re.findall(r"\S+", text)

    @staticmethod
    def _normalize_for_edit_compare(text: str) -> str:
        without_punctuation = "".join(
            char for char in text if not unicodedata.category(char).startswith("P")
        )
        return without_punctuation.casefold()

    @staticmethod
    def _align_tokens(
        original_tokens: list[str],
        revised_tokens: list[str],
    ) -> list[tuple[str, str | None, str | None]]:
        original_count = len(original_tokens)
        revised_count = len(revised_tokens)
        dp = [[0] * (revised_count + 1) for _ in range(original_count + 1)]

        for i in range(1, original_count + 1):
            dp[i][0] = i
        for j in range(1, revised_count + 1):
            dp[0][j] = j

        for i in range(1, original_count + 1):
            for j in range(1, revised_count + 1):
                original_token = OpenAICompatibleCritic._normalize_for_edit_compare(
                    original_tokens[i - 1]
                )
                revised_token = OpenAICompatibleCritic._normalize_for_edit_compare(
                    revised_tokens[j - 1]
                )
                substitution_cost = 0 if original_token == revised_token else 1
                dp[i][j] = min(
                    dp[i - 1][j - 1] + substitution_cost,
                    dp[i - 1][j] + 1,
                    dp[i][j - 1] + 1,
                )

        ops: list[tuple[str, str | None, str | None]] = []
        i = original_count
        j = revised_count
        while i > 0 or j > 0:
            if i > 0 and j > 0:
                original_token = OpenAICompatibleCritic._normalize_for_edit_compare(
                    original_tokens[i - 1]
                )
                revised_token = OpenAICompatibleCritic._normalize_for_edit_compare(
                    revised_tokens[j - 1]
                )
                substitution_cost = 0 if original_token == revised_token else 1
                if dp[i][j] == dp[i - 1][j - 1] + substitution_cost:
                    operation = "equal" if substitution_cost == 0 else "replace"
                    ops.append((operation, original_tokens[i - 1], revised_tokens[j - 1]))
                    i -= 1
                    j -= 1
                    continue
            if i > 0 and dp[i][j] == dp[i - 1][j] + 1:
                ops.append(("delete", original_tokens[i - 1], None))
                i -= 1
                continue
            ops.append(("insert", None, revised_tokens[j - 1]))
            j -= 1

        ops.reverse()
        return ops

    @staticmethod
    def _passes_critic_rules(
        *,
        format_check: CriticCheckResult,
        kw_check: CriticCheckResult,
        image_check: CriticCheckResult,
        history_check: CriticCheckResult,
    ) -> bool:
        """Apply final critic rules after all checks return pass/reason values."""
        if not format_check.passed or not kw_check.passed:
            return False
        if not image_check.passed and not history_check.passed:
            return False
        return True

    @staticmethod
    def _collect_reject_reasons(checks: list[CriticCheckResult]) -> list[str]:
        reasons = []
        for check in checks:
            if check.passed:
                continue
            reason = (
                check.reject_reason
                or check.critique
                or check.metadata.get("reason")
                or f"{check.name} failed"
            )
            reasons.append(str(reason))
        return reasons

    async def _run_kw_verify_check(
        self,
        edits: list[TranscriptEdit],
        state: AgentState,
        executor: ToolExecutor,
        registry: ToolRegistry,
    ) -> CriticCheckResult:
        if not edits:
            return CriticCheckResult(
                name="kw_verify",
                passed=True,
                critique=None,
                confidence=1.0,
                metadata={"skipped": "no transcript edits", "reason": "No transcript edits to verify."},
            )
        verifiable_edits = [edit for edit in edits if edit.revised_text != "<deleted>"]
        if not verifiable_edits:
            return CriticCheckResult(
                name="kw_verify",
                passed=True,
                critique=None,
                confidence=1.0,
                metadata={
                    "skipped": "deletion-only transcript edits",
                    "reason": "No revised transcript text to verify acoustically.",
                },
            )
        if "kw_verify" not in registry:
            return CriticCheckResult(
                name="kw_verify",
                passed=True,
                critique=None,
                confidence=0.0,
                metadata={"skipped": "tool not registered", "reason": "kw_verify is unavailable."},
            )

        original_audios = [a for a in state.get("audio_list", []) if a.source == "original"]
        edit_results = []
        missing_words = []
        for edit in verifiable_edits:
            tool_outputs = []
            passed_any = False
            confidences = []
            for audio in original_audios:
                result = await executor.execute(
                    ToolCallRequest(
                        tool_name="kw_verify",
                        args={"audio_path": audio.path, "target_text": edit.revised_text},
                        context={"critic": self.name, "audio_id": audio.audio_id},
                    )
                )
                tool_outputs.append(result.output)
                parsed = result.output.get("parsed_data", {})
                exists = bool(parsed.get("exists", result.success))
                confidence = float(parsed.get("confidence", 0.0))
                passed_any = passed_any or (result.success and exists)
                confidences.append(confidence)
            edit_results.append(
                {
                    "edit": edit.model_dump(mode="json"),
                    "passed": passed_any,
                    "confidence": max(confidences, default=0.0),
                    "reason": None
                    if passed_any
                    else f"Revised text not verified in audio: {edit.revised_text}",
                    "tool_outputs": tool_outputs,
                }
            )
            if not passed_any:
                missing_words.append(edit.revised_text)
        passed = all(item["passed"] for item in edit_results)
        confidence = min((item["confidence"] for item in edit_results), default=0.0)
        reject_reason = (
            "The refined transcript contains the following words: "
            f"{missing_words}, which do not appear in the original audio."
            if missing_words
            else None
        )
        return CriticCheckResult(
            name="kw_verify",
            passed=passed,
            critique=reject_reason,
            reject_reason=reject_reason,
            confidence=confidence,
            metadata={
                "edits": edit_results,
                "missing_words": missing_words,
                "reason": reject_reason or "All transcript edits verified.",
            },
        )

    async def _run_image_check(
        self,
        final_answer: str,
        state: AgentState,
        executor: ToolExecutor,
        registry: ToolRegistry,
    ) -> CriticCheckResult:
        image_list = state.get("image_list", [])
        if not image_list:
            return CriticCheckResult(
                name="image_qa",
                passed=True,
                confidence=1.0,
                metadata={"skipped": "no reference images", "reason": "No reference images to check."},
            )
        if "image_qa" not in registry:
            return CriticCheckResult(
                name="image_qa",
                passed=True,
                confidence=0.0,
                metadata={"skipped": "tool not registered", "reason": "image_qa is unavailable."},
            )

        image_results = []
        for image in image_list:
            question = (
                "Does this proposed final answer contradict anything visible in the image? "
                "Return only valid JSON with this schema: "
                '{"passed": true|false, "reason": "<concise reason>"}. '
                "Use passed=false only if the image clearly contradicts the final answer.\n\n"
                f"Final answer:\n{final_answer}"
            )
            result = await executor.execute(
                ToolCallRequest(
                    tool_name="image_qa",
                    args={"image_path": image.path, "question": question},
                    context={"critic": self.name, "image_id": image.image_id},
                )
            )
            parsed = self._extract_result_dict(result.output)
            passed = bool(parsed.get("passed", result.success))
            reason = str(
                parsed.get("reason")
                or parsed.get("answer")
                or result.error_message
                or "image_qa returned no reason."
            )
            image_results.append(
                {
                    "image_id": image.image_id,
                    "passed": passed,
                    "confidence": float(parsed.get("confidence", 0.5 if result.success else 0.0)),
                    "reason": reason,
                    "tool_output": result.output,
                    "parsed_data": parsed,
                    "error_message": result.error_message,
                }
            )
        passed = all(item["passed"] for item in image_results)
        confidence = min((item["confidence"] for item in image_results), default=0.0)
        reject_reason = None if passed else "; ".join(
            str(item["reason"]) for item in image_results if not item["passed"]
        )
        return CriticCheckResult(
            name="image_qa",
            passed=passed,
            critique=reject_reason,
            reject_reason=reject_reason,
            confidence=confidence,
            metadata={
                "images": image_results,
                "reason": reject_reason or "image_qa evidence collected for final judgement.",
            },
        )

    async def _run_history_check(
        self,
        final_transcript: str,
        final_answer: str,
        state: AgentState,
        executor: ToolExecutor,
        registry: ToolRegistry,
    ) -> CriticCheckResult:
        if "external_memory_retrieve" not in registry:
            check = CriticCheckResult(
                name="history",
                passed=True,
                confidence=0.0,
                metadata={
                    "skipped": "tool not registered",
                    "reason": "external_memory_retrieve is unavailable.",
                },
            )
            return check
        result = await executor.execute(
            ToolCallRequest(
                tool_name="external_memory_retrieve",
                args={"query": state.get("question", "")},
                context={"critic": self.name},
            )
        )
        memory_text = result.output.get("text", "")
        memory_payload = self._try_parse_json(memory_text)
        if not result.success:
            reject_reason = result.error_message or "external memory retrieval failed"
            check = CriticCheckResult(
                name="history",
                passed=False,
                critique=reject_reason,
                reject_reason=reject_reason,
                confidence=0.0,
                metadata={"tool_output": result.output, "reason": reject_reason},
            )
            return check
        history = memory_text
        if isinstance(memory_payload, dict):
            history = str(memory_payload.get("memory", "") or "")
        if not history.strip():
            check = CriticCheckResult(
                name="history",
                passed=True,
                confidence=1.0,
                metadata={
                    "skipped": "empty history",
                    "tool_output": memory_payload if isinstance(memory_payload, dict) else memory_text,
                    "reason": "History is empty.",
                },
            )
            return check
        history_check = await asyncio.to_thread(
            self._judge_history_alignment,
            question=state.get("question", ""),
            final_answer=final_answer,
            final_transcript=final_transcript,
            history=history,
        )
        history_check.metadata.update(
            {
                "tool_output": memory_payload if isinstance(memory_payload, dict) else memory_text,
                "history": history[:2000],
            }
        )
        return history_check

    @staticmethod
    def _extract_result_dict(output: dict[str, Any]) -> dict[str, Any]:
        parsed = output.get("parsed_data")
        if isinstance(parsed, dict):
            return parsed
        text = output.get("text", "")
        if isinstance(text, str) and text.strip():
            try:
                parsed_text = json.loads(text)
            except json.JSONDecodeError:
                return {}
            if isinstance(parsed_text, dict):
                return parsed_text
        return {}

    def _judge_history_alignment(
        self,
        *,
        question: str,
        final_answer: str,
        final_transcript: str,
        history: str,
    ) -> CriticCheckResult:
        payload = {
            "task": "history_check",
            "question": question,
            "final_answer": final_answer,
            "final_transcript": final_transcript,
            "history": history,
            "instruction": (
                "Check whether the final answer conflicts with the supplied history. "
                "Return JSON: {\"passed\": bool, \"reason\": string, \"confidence\": number}."
            ),
        }
        data = self._call_json(load_prompt("critic_history_check_system"), payload)
        passed = bool(data.get("passed", False))
        reason = str(data.get("reason") or data.get("reject_reason") or data.get("critique") or "")
        reject_reason = None if passed else reason
        return CriticCheckResult(
            name="history",
            passed=passed,
            critique=reject_reason,
            reject_reason=reject_reason,
            confidence=float(data.get("confidence", 0.0)),
            metadata={
                "reason": reason,
                "question": question,
            },
        )

    @staticmethod
    def _try_parse_json(text: str) -> Any:
        try:
            return json.loads(text)
        except json.JSONDecodeError:
            return None
