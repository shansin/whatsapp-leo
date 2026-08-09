"""Instruction loading and template building for WhatsApp Leo agent."""

import os

from logging_setup import logger


def _load_instructions():
    """Load and parse instruction sections from instructions.txt."""
    instr_path = os.path.join(os.path.dirname(__file__), "instructions.txt")
    if not os.path.exists(instr_path):
        logger.warning(f"instructions.txt not found at {instr_path}")
        return "", "", "", "", ""

    with open(instr_path) as f:
        content = f.read()

    sections = {}
    current_section = None
    lines = []

    for line in content.splitlines():
        if line.startswith("[") and line.endswith("]"):
            if current_section:
                sections[current_section] = "\n".join(lines).strip()
            current_section = line[1:-1]
            lines = []
        else:
            lines.append(line)

    if current_section:
        sections[current_section] = "\n".join(lines).strip()

    return (
        sections.get("BASE_INSTRUCTIONS", "") + "\n",
        "\n" + sections.get("PRIVILEDGED_INSTRUCTIONS", "") + "\n",
        "\n" + sections.get("COMMON_RULES", ""),
        sections.get("REMINDER_INSTRUCTIONS", ""),
        "\n" + sections.get("MEMORY_INSTRUCTIONS", ""),
    )


# Load once at import time
(
    _BASE_INSTRUCTION_TEMPLATE,
    _PRIVILEDGED_INSTRUCTIONS,
    _COMMON_RULES,
    _REMINDER_INSTRUCTIONS_TEMPLATE,
    _MEMORY_INSTRUCTIONS,
) = _load_instructions()

# Pre-built instruction templates (only {current_time} needs filling at message time)
INSTRUCTIONS_PRIVILEGED_TEMPLATE = (
    _BASE_INSTRUCTION_TEMPLATE + _PRIVILEDGED_INSTRUCTIONS + _COMMON_RULES + _MEMORY_INSTRUCTIONS
)
INSTRUCTIONS_BASIC_TEMPLATE = _BASE_INSTRUCTION_TEMPLATE + _COMMON_RULES + _MEMORY_INSTRUCTIONS
REMINDER_INSTRUCTIONS_TEMPLATE = _REMINDER_INSTRUCTIONS_TEMPLATE
