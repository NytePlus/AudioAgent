"""Tests for critic schemas and transcript helpers."""

import asyncio

from audio_agent.core.schemas import CriticCheckResult, CriticResult, TranscriptEdit
from audio_agent.critic.openai_compatible_critic import OpenAICompatibleCritic
from audio_agent.utils.transcript import extract_transcript_text, normalize_transcription_answer


def test_extract_transcript_text_from_json_pred():
    text = '```json\n{"pred": "hello   world"}\n```'
    assert extract_transcript_text(text) == "hello world"


def test_extract_transcript_text_uses_transcription_and_ignores_extra_fields():
    text = (
        '{\n'
        '  "transcription": "hello   world",\n'
        '  "domain_specific_terms": ["not spoken"],\n'
        '  "notes": "also not spoken"\n'
        '}'
    )
    assert extract_transcript_text(text) == "hello world"


def test_extract_transcript_text_does_not_fallback_to_json_shell():
    text = '{"domain_specific_terms": ["not spoken"]}'
    assert extract_transcript_text(text) is None


def test_extract_transcript_text_falls_back_to_plain_text():
    assert extract_transcript_text(" hello\nworld ") == "hello world"


def test_normalize_transcription_answer_uses_canonical_key():
    text = '{"corrected_transcription": "hello   world", "domain_specific_terms": ["ignored"]}'
    assert normalize_transcription_answer(text) == '{"transcription": "hello world"}'


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


def test_format_check_accepts_transcription_json_without_model_call():
    critic = OpenAICompatibleCritic.__new__(OpenAICompatibleCritic)
    result = critic._check_format(
        {},
        (
            '{\n'
            '  "transcription": "Yes, and we represent the German sports youth.",\n'
            '  "confidence": 0.9\n'
            '}'
        ),
    )

    assert result.passed is True
    assert result.confidence == 1.0
    assert result.metadata["parsed"] == {
        "transcription": "Yes, and we represent the German sports youth.",
        "confidence": 0.9,
    }


def test_format_check_rejects_non_transcription_json():
    critic = OpenAICompatibleCritic.__new__(OpenAICompatibleCritic)
    result = critic._check_format({}, '{"answer": "hello"}')

    assert result.passed is False
    assert result.reject_reason == "invalid format"


def test_extract_edits_uses_local_alignment_for_replacements():
    critic = OpenAICompatibleCritic.__new__(OpenAICompatibleCritic)
    edits = critic._extract_edits(
        "we drive the project iCoachKids forward",
        "we drive the project i Coach Kids forward",
    )

    assert [(edit.original_text, edit.revised_text) for edit in edits] == [
        ("iCoachKids", "i Coach Kids")
    ]
    assert "replace" in (edits[0].rationale or "")


def test_extract_edits_uses_local_alignment_for_insertions_and_deletions():
    critic = OpenAICompatibleCritic.__new__(OpenAICompatibleCritic)
    insert_edits = critic._extract_edits("hello world", "hello brave world")
    delete_edits = critic._extract_edits("hello brave world", "hello world")

    assert [(edit.original_text, edit.revised_text) for edit in insert_edits] == [
        ("<inserted>", "brave")
    ]
    assert [(edit.original_text, edit.revised_text) for edit in delete_edits] == [
        ("brave", "<deleted>")
    ]


def test_extract_edits_ignores_case_and_punctuation_changes():
    critic = OpenAICompatibleCritic.__new__(OpenAICompatibleCritic)
    edits = critic._extract_edits(
        "German Sports Youth, and I Coach Kids.",
        "german sports youth and i coach kids",
    )

    assert edits == []


def test_kw_verify_skips_deletion_only_edits():
    critic = OpenAICompatibleCritic.__new__(OpenAICompatibleCritic)
    check = asyncio.run(
        critic._run_kw_verify_check(
            edits=[TranscriptEdit(original_text="brave", revised_text="<deleted>")],
            state={},
            executor=None,
            registry={},
        )
    )

    assert check.passed is True
    assert check.metadata["skipped"] == "deletion-only transcript edits"


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
