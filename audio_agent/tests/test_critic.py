"""Tests for critic schemas and transcript helpers."""

from audio_agent.core.schemas import CriticCheckResult, CriticResult, TranscriptEdit
from audio_agent.critic.openai_compatible_critic import OpenAICompatibleCritic
from audio_agent.utils.transcript import extract_transcript_text


def test_extract_transcript_text_from_json_pred():
    text = '```json\n{"pred": "hello   world"}\n```'
    assert extract_transcript_text(text) == "hello world"


def test_extract_transcript_text_falls_back_to_plain_text():
    assert extract_transcript_text(" hello\nworld ") == "hello world"


def test_critic_result_schema_accepts_checks_and_edits():
    edit = TranscriptEdit(original_text="helo", revised_text="hello", rationale="spelling")
    check = CriticCheckResult(name="kw_verify", passed=True, confidence=0.9)
    result = CriticResult(
        passed=True,
        critique=None,
        reject_reason=None,
        confidence=0.9,
        checks=[check],
        transcript_edits=[edit],
    )

    assert result.passed is True
    assert result.transcript_edits[0].revised_text == "hello"


def test_failed_critic_result_requires_critique():
    try:
        CriticResult(passed=False, confidence=0.5)
    except ValueError as exc:
        assert "reject_reason" in str(exc)
    else:
        raise AssertionError("Expected failed CriticResult without reject_reason to be invalid")


def test_failed_critic_result_accepts_reject_reason():
    result = CriticResult(passed=False, reject_reason="unsupported transcript edit", confidence=0.5)

    assert result.reject_reason == "unsupported transcript edit"
    assert result.critique == "unsupported transcript edit"


def test_critic_rules_reject_format_or_keyword_failures():
    passed = OpenAICompatibleCritic._passes_critic_rules(
        format_check=CriticCheckResult(name="format", passed=False, confidence=0.9),
        kw_check=CriticCheckResult(name="kw_verify", passed=True, confidence=0.9),
        image_check=CriticCheckResult(name="image_qa", passed=True, confidence=0.9),
        history_check=CriticCheckResult(name="history", passed=True, confidence=0.9),
    )
    assert passed is False

    passed = OpenAICompatibleCritic._passes_critic_rules(
        format_check=CriticCheckResult(name="format", passed=True, confidence=0.9),
        kw_check=CriticCheckResult(name="kw_verify", passed=False, confidence=0.9),
        image_check=CriticCheckResult(name="image_qa", passed=True, confidence=0.9),
        history_check=CriticCheckResult(name="history", passed=True, confidence=0.9),
    )
    assert passed is False


def test_critic_rules_reject_only_when_image_and_history_both_fail():
    passed = OpenAICompatibleCritic._passes_critic_rules(
        format_check=CriticCheckResult(name="format", passed=True, confidence=0.9),
        kw_check=CriticCheckResult(name="kw_verify", passed=True, confidence=0.9),
        image_check=CriticCheckResult(name="image_qa", passed=False, confidence=0.9),
        history_check=CriticCheckResult(name="history", passed=True, confidence=0.9),
    )
    assert passed is True

    passed = OpenAICompatibleCritic._passes_critic_rules(
        format_check=CriticCheckResult(name="format", passed=True, confidence=0.9),
        kw_check=CriticCheckResult(name="kw_verify", passed=True, confidence=0.9),
        image_check=CriticCheckResult(name="image_qa", passed=False, confidence=0.9),
        history_check=CriticCheckResult(name="history", passed=False, confidence=0.9),
    )
    assert passed is False
