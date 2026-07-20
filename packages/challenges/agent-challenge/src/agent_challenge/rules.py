from __future__ import annotations

import hashlib
from dataclasses import dataclass
from pathlib import Path


class RulesLoadError(RuntimeError):
    pass


@dataclass(frozen=True)
class RulesBundle:
    rules_version: str
    files: list[str]
    policy_text: str


def load_rules(repository_root: Path | str | None = None) -> RulesBundle:
    root = Path(repository_root) if repository_root is not None else Path(__file__).parents[2]
    rules_dir = root / ".rules"
    if not rules_dir.is_dir():
        raise RulesLoadError(f"rules directory not found: {rules_dir}")

    rule_paths = sorted(path for path in rules_dir.glob("*.md") if path.is_file())
    if not rule_paths:
        raise RulesLoadError(f"rules directory has no Markdown rules: {rules_dir}")

    digest = hashlib.sha256()
    files: list[str] = []
    policy_sections: list[str] = []
    for path in rule_paths:
        relative_path = path.relative_to(root).as_posix()
        contents = path.read_bytes()
        digest.update(relative_path.encode("utf-8"))
        digest.update(b"\0")
        digest.update(contents)
        digest.update(b"\0")
        files.append(relative_path)
        policy_sections.append(contents.decode("utf-8"))

    return RulesBundle(
        rules_version=digest.hexdigest(),
        files=files,
        policy_text="\n\n".join(section.rstrip() for section in policy_sections) + "\n",
    )


load_rules_bundle = load_rules
