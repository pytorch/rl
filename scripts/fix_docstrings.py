#!/usr/bin/env python3
from __future__ import annotations

import re

from pathlib import Path

# Common verbs that need to be fixed when they appear in plural form
VERBS_TO_FIX = {
    "applies": "apply",
    "empties": "empty",
    "launches": "launch",
    "copies": "copy",
    "initializes": "initialize",
    "appends": "append",
    "is": "be",
    "it is": "be",
    "this is": "be",
    "a": "use",  # "A method to..." -> "Use method to..."
    "an": "use",  # "An initializer for..." -> "Use initializer for..."
    "the": "use",  # "The forward method" -> "Use forward method"
    "default": "use",  # "Default configuration" -> "Use configuration"
    "helper": "use",  # "Helper function" -> "Use function"
    "custom": "use",  # "Custom weight init" -> "Use weight init"
    "number": "count",  # "Number of dimensions" -> "Count dimensions"
}


def fix_first_sentence(sentence):
    # Split into words
    words = sentence.split()
    if not words:
        return sentence

    # Handle special cases first
    lower_sentence = sentence.lower()
    if lower_sentence.startswith(("it is ", "this is ")):
        return "Be" + sentence[sentence.lower().find(" is ") + 4 :]

    # Handle articles and descriptive words
    if words[0].lower() in ("a", "an", "the"):
        if len(words) > 1 and words[1].lower() in VERBS_TO_FIX:
            words[0] = VERBS_TO_FIX[words[0].lower()]
            words[1] = VERBS_TO_FIX[words[1].lower()]
            return " ".join(words)
        else:
            words[0] = "Use"
            return " ".join(words)

    # Handle other cases
    first_word = words[0].lower()
    if first_word in VERBS_TO_FIX:
        words[0] = VERBS_TO_FIX[first_word]

    return " ".join(words)


def fix_docstring(content):
    # Pattern to match docstrings that start with triple quotes
    pattern = r'("""|\'\'\')(.*?)(\1)'

    def fix_first_line(match):
        quote = match.group(1)
        docstring = match.group(2)

        # Split into first line and rest
        lines = docstring.split("\n", 1)
        first_line = lines[0].strip()
        rest = lines[1] if len(lines) > 1 else ""

        # Fix the first line
        fixed_first_line = fix_first_sentence(first_line)

        # Reconstruct docstring
        if rest:
            return f"{quote}{fixed_first_line}\n{rest}{quote}"
        return f"{quote}{fixed_first_line}{quote}"

    return re.sub(pattern, fix_first_line, content, flags=re.DOTALL)


def process_file(file_path):
    with open(file_path) as f:
        content = f.read()

    fixed_content = fix_docstring(content)

    if fixed_content != content:
        with open(file_path, "w") as f:
            f.write(fixed_content)


def main():
    # Process all Python files
    for py_file in Path(".").rglob("*.py"):
        if not str(py_file).startswith((".", "venv")):  # Skip hidden and venv files
            process_file(py_file)


if __name__ == "__main__":
    main()
