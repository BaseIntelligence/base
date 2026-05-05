"""Utilities for rendering Platform Network repository templates."""

from __future__ import annotations

import re
import shutil
from dataclasses import dataclass
from pathlib import Path

_PLACEHOLDER_RE = re.compile(r"{{\s*([a-zA-Z_][a-zA-Z0-9_]*)\s*}}")


@dataclass(frozen=True)
class ChallengeTemplateContext:
    """Context used to render a challenge repository template."""

    slug: str
    name: str
    package_name: str
    ghcr_image: str
    challenge_version: str = "0.1.0"
    api_version: str = "1.0"
    sdk_version: str = "1.0.0"
    python_version: str = "3.12"

    @classmethod
    def from_slug(
        cls,
        slug: str,
        name: str | None = None,
        ghcr_image: str | None = None,
        challenge_version: str = "0.1.0",
    ) -> ChallengeTemplateContext:
        """Build a render context from a challenge slug."""

        package_name = _slug_to_package_name(slug)
        display_name = name or slug.replace("-", " ").replace("_", " ").title()
        image = ghcr_image or f"ghcr.io/platformnetwork/{slug}:latest"
        return cls(
            slug=slug,
            name=display_name,
            package_name=package_name,
            ghcr_image=image,
            challenge_version=challenge_version,
        )

    def as_dict(self) -> dict[str, str]:
        """Return the context as string values for template rendering."""

        return {
            "slug": self.slug,
            "name": self.name,
            "package_name": self.package_name,
            "ghcr_image": self.ghcr_image,
            "challenge_version": self.challenge_version,
            "api_version": self.api_version,
            "sdk_version": self.sdk_version,
            "python_version": self.python_version,
        }


def render_challenge_template(
    output_dir: Path,
    context: ChallengeTemplateContext,
    template_dir: Path | None = None,
    overwrite: bool = False,
) -> list[Path]:
    """Render the challenge template into ``output_dir``.

    Args:
        output_dir: Destination repository directory.
        context: Challenge template variables.
        template_dir: Optional source template directory.
        overwrite: Whether existing files may be overwritten.

    Returns:
        Paths written relative to ``output_dir``.
    """

    source_dir = template_dir or Path(__file__).parent / "templates" / "challenge"
    if not source_dir.is_dir():
        raise FileNotFoundError(f"Challenge template directory not found: {source_dir}")

    output_dir.mkdir(parents=True, exist_ok=True)
    variables = context.as_dict()
    written: list[Path] = []

    for source_path in sorted(source_dir.rglob("*")):
        if source_path.is_dir():
            continue

        relative_source = source_path.relative_to(source_dir)
        if relative_source == Path("__init__.py"):
            continue
        relative_target = _render_path(relative_source, variables)
        if relative_target.suffix == ".j2":
            relative_target = relative_target.with_suffix("")

        target_path = output_dir / relative_target
        if target_path.exists() and not overwrite:
            raise FileExistsError(f"Refusing to overwrite existing file: {target_path}")

        target_path.parent.mkdir(parents=True, exist_ok=True)
        if source_path.suffix == ".j2":
            rendered = _render_text(source_path.read_text(encoding="utf-8"), variables)
            target_path.write_text(rendered, encoding="utf-8")
        else:
            shutil.copyfile(source_path, target_path)
        written.append(relative_target)

    return written


def _slug_to_package_name(slug: str) -> str:
    package_name = re.sub(r"[^a-zA-Z0-9_]+", "_", slug).strip("_").lower()
    if not package_name:
        raise ValueError(
            "Challenge slug must contain at least one alphanumeric character"
        )
    if package_name[0].isdigit():
        package_name = f"challenge_{package_name}"
    return package_name


def _render_path(path: Path, variables: dict[str, str]) -> Path:
    rendered_parts = [_render_text(part, variables) for part in path.parts]
    return Path(*rendered_parts)


def _render_text(template: str, variables: dict[str, str]) -> str:
    def replace(match: re.Match[str]) -> str:
        key = match.group(1)
        return variables.get(key, match.group(0))

    rendered = _PLACEHOLDER_RE.sub(replace, template)
    for key, value in variables.items():
        rendered = rendered.replace(f"__{key}__", value)
    return rendered
