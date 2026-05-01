"""OpenAI-compatible implementation of the final-answer critic."""

from __future__ import annotations

import json
import os
from typing import Any

from audio_agent.critic.base import BaseCritic
from audio_agent.core.errors import CriticError
from audio_agent.core.logging import log_info
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

        final_transcript = extract_transcript_text(proposed_answer) or proposed_answer
        initial_transcript = state.get("initial_transcript")
        checks: list[CriticCheckResult] = []

        format_check = self._check_format(state, proposed_answer)
        checks.append(format_check)

        edits = self._extract_edits(initial_transcript, final_transcript)
        kw_check = await self._run_kw_verify_check(
            edits=edits,
            state=state,
            executor=executor,
            registry=registry,
        )
        checks.append(kw_check)

        image_check = await self._run_image_check(
            final_answer=proposed_answer,
            state=state,
            executor=executor,
            registry=registry,
        )
        checks.append(image_check)

        history_check = await self._run_history_check(
            final_transcript=final_transcript,
            final_answer=proposed_answer,
            state=state,
            executor=executor,
            registry=registry,
        )
        checks.append(history_check)

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

    def _check_format(self, state: AgentState, proposed_answer: str) -> CriticCheckResult:
        initial_plan = state.get("initial_plan")
        payload = {
            "task": "format_check",
            "question": state.get("question", ""),
            "expected_format": state.get("expected_output_format"),
            "requires_audio_output": bool(initial_plan.requires_audio_output) if initial_plan else False,
            "proposed_answer": proposed_answer,
        }
        data = self._call_json(load_prompt("critic_system"), payload)
        format_data = data.get("format_check", data)
        passed = bool(format_data.get("passed", False))
        reject_reason = None if passed else "invalid format"
        return CriticCheckResult(
            name="format",
            passed=passed,
            critique=reject_reason,
            reject_reason=reject_reason,
            confidence=float(format_data.get("confidence", 0.0)),
            metadata={
                "raw": format_data,
                "reason": reject_reason or "Format is valid.",
            },
        )

    def _extract_edits(
        self,
        initial_transcript: str | None,
        final_transcript: str,
    ) -> list[TranscriptEdit]:
        if not initial_transcript or not final_transcript:
            return []
        payload = {
            "task": "extract_transcript_edits",
            "initial_transcript": initial_transcript,
            "final_transcript": final_transcript,
        }
        data = self._call_json(load_prompt("critic_system"), payload)
        edits = data.get("edits", [])
        if not isinstance(edits, list):
            return []
        normalized: list[TranscriptEdit] = []
        for item in edits:
            if not isinstance(item, dict):
                continue
            original = item.get("original_text") or item.get("a")
            revised = item.get("revised_text") or item.get("b")
            if original and revised and str(original).strip() != str(revised).strip():
                normalized.append(
                    TranscriptEdit(
                        original_text=str(original),
                        revised_text=str(revised),
                        rationale=item.get("rationale"),
                    )
                )
        return normalized

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

    @staticmethod
    def _log_history_check_result(check: CriticCheckResult) -> None:
        log_info(
            "critic_history_check",
            {
                "passed": check.passed,
                "reason": (
                    check.reject_reason
                    or check.critique
                    or check.metadata.get("reason")
                    or "No history conflict."
                ),
                "confidence": check.confidence,
            },
        )

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
        for edit in edits:
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
            self._log_history_check_result(check)
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
            self._log_history_check_result(check)
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
            self._log_history_check_result(check)
            return check
        history_check = self._judge_history_alignment(
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
        self._log_history_check_result(history_check)
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
        data = self._call_json(load_prompt("critic_system"), payload)
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

    def _judge_tool_text(
        self,
        task: str,
        tool_text: str,
        final_answer: str,
        extra_context: dict[str, Any],
    ) -> CriticCheckResult:
        payload = {
            "task": task,
            "tool_evidence": tool_text,
            "final_answer": final_answer,
            "context": extra_context,
            "instruction": (
                "Return JSON: {\"passed\": bool, \"reject_reason\": string|null, "
                "\"confidence\": number}. Use reject_reason only when passed is false."
            ),
        }
        data = self._call_json(load_prompt("critic_system"), payload)
        reject_reason = data.get("reject_reason") or data.get("critique")
        return CriticCheckResult(
            name=task,
            passed=bool(data.get("passed", False)),
            critique=reject_reason,
            reject_reason=reject_reason,
            confidence=float(data.get("confidence", 0.0)),
            metadata={
                "tool_evidence": tool_text[:2000],
                "reject_reason": reject_reason,
                **extra_context,
            },
        )
