"""Skill reference loading and rendering for initial planning."""

from pathlib import Path
from typing import Any


PROMPTS_DIR = Path(__file__).parent.parent / "prompts"
TASK_SKILLS_PATH = PROMPTS_DIR / "task_skills.yaml"


def _load_yaml_safe(path: Path) -> dict[str, Any] | None:
    try:
        import yaml
    except ImportError:  # pragma: no cover
        return None
    if not path.exists():
        return None
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def render_skills_reference() -> str:
    """
    Render task_skills.yaml as a markdown reference for the planner.

    Returns an empty string if the file is missing or YAML is unavailable.
    """
    data = _load_yaml_safe(TASK_SKILLS_PATH)
    if not data:
        return ""

    skills = data.get("skills", [])
    if not skills:
        return ""
    abstract_slots = data.get("abstract_slots", {})

    lines = [
        "## Task Skills Reference",
        "",
        "The following skills describe useful abstract tool chains and guardrails for common audio tasks. "
        "Use them as guidance when forming your approach, focus_points, possible_tool_types, and detailed_plan. "
        "You are not required to follow them rigidly. If you draw on a specific skill, mention it in `notes`.",
        "",
    ]

    for skill in skills:
        skill_id = skill.get("id", "unknown")
        task = skill.get("task", "")
        chain = skill.get("chain", [])
        guardrails = skill.get("guardrails", [])
        bindings = skill.get("default_bindings_from_ref_flows", {})

        lines.append(f"### {skill_id}")
        if task:
            lines.append(f"- **Task**: {task}")

        if chain:
            chain_parts = []
            for step in chain:
                slot = step.get("slot", "")
                optional = step.get("optional", False)
                if slot:
                    chain_parts.append(f"{slot} (optional)" if optional else slot)
            if chain_parts:
                lines.append(f"- **Chain**: {' → '.join(chain_parts)}")

            slot_notes = []
            for step in chain:
                slot = step.get("slot", "")
                slot_data = abstract_slots.get(slot, {})
                contract = slot_data.get("contract")
                defaults = slot_data.get("current_catalog_defaults", [])
                if contract or defaults:
                    default_text = f"; defaults: {', '.join(defaults)}" if defaults else ""
                    slot_notes.append(f"{slot}: {contract or 'No contract'}{default_text}")
            if slot_notes:
                lines.append("- **Slots**:")
                for note in slot_notes:
                    lines.append(f"  - {note}")

        if guardrails:
            lines.append("- **Guardrails**:")
            for g in guardrails:
                lines.append(f"  - {g}")

        if bindings:
            lines.append("- **Suggested concrete tools**:")
            for slot, items in bindings.items():
                tool_names = [item.get("tool", "") for item in items if item.get("tool")]
                if tool_names:
                    lines.append(f"  - {slot}: {', '.join(tool_names)}")

        lines.append("")

    return "\n".join(lines)
