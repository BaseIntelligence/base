"""SWE-Forge dataset loading and deterministic task selection."""

from __future__ import annotations

import hashlib
import json
import logging
import random
from dataclasses import dataclass
from urllib.error import HTTPError, URLError
from urllib.request import urlopen

from ..core.config import settings

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class SweForgeTask:
    """Minimal task metadata required by the evaluator."""

    task_id: str
    docker_image: str
    prompt: str = ""


FALLBACK_TASK_IDS: tuple[str, ...] = (
    "Amulet-Team-Amulet-Map-Editor-1337",
    "ArduPilot-ardupilot_wiki-7611",
    "ArduPilot-pymavlink-1199",
    "BerriAI-litellm-24938",
    "Bitmessage-PyBitmessage-2318",
    "CelestoAI-SmolVM-119",
    "FuzzingLabs-secpipe-53",
    "HHS-simpler-grants-gov-9410",
    "HKUDS-OpenSpace-61",
    "HKUDS-nanobot-2835",
    "Knowledgator-GLinker-22",
    "MISP-misp-objects-498",
    "NVIDIA-NVFlare-4411",
    "NVIDIA-NeMo-Curator-1693",
    "NVIDIA-TensorRT-LLM-12804",
    "NVIDIA-cuEquivariance-263",
    "NousResearch-hermes-agent-5427",
    "NousResearch-hermes-agent-5577",
    "PyAV-Org-PyAV-2225",
    "Significant-Gravitas-AutoGPT-12695",
)


FALLBACK_TASKS: tuple[SweForgeTask, ...] = tuple(
    SweForgeTask(
        task_id=task_id,
        docker_image=f"{settings.swe_forge_image_prefix}:{task_id}",
        prompt="Fallback SWE-Forge task from the public dataset tree.",
    )
    for task_id in FALLBACK_TASK_IDS
)


def load_swe_forge_tasks(tree_url: str | None = None) -> list[SweForgeTask]:
    """Load and validate SWE-Forge task metadata from the Hugging Face tree API."""

    url = tree_url or settings.swe_forge_tree_url
    try:
        with urlopen(url, timeout=15) as response:
            raw = response.read().decode("utf-8")
        records = json.loads(raw)
    except (HTTPError, OSError, URLError, TimeoutError, json.JSONDecodeError) as exc:
        logger.warning(
            "SWE-Forge tree fetch from %s failed (%s); substituting %d hardcoded fallback tasks",
            url,
            exc,
            len(FALLBACK_TASKS),
        )
        return list(FALLBACK_TASKS)

    tasks = _tasks_from_tree(records if isinstance(records, list) else [])
    if not tasks:
        logger.warning(
            "SWE-Forge tree at %s yielded no usable tasks; using %d hardcoded fallback tasks",
            url,
            len(FALLBACK_TASKS),
        )
        return list(FALLBACK_TASKS)
    return tasks


def select_tasks(
    tasks: list[SweForgeTask],
    *,
    agent_hash: str,
    count: int,
) -> list[SweForgeTask]:
    """Select a reproducible subset of tasks from an agent hash."""

    if count <= 0:
        return []
    selected = list(tasks)
    seed = int.from_bytes(hashlib.sha256(agent_hash.encode("utf-8")).digest()[:8], "big")
    random.Random(seed).shuffle(selected)
    return selected[: min(count, len(selected))]


def tasks_to_json(tasks: list[SweForgeTask]) -> str:
    """Serialize selected tasks for database storage."""

    return json.dumps([task.__dict__ for task in tasks], separators=(",", ":"))


def tasks_from_json(raw: str) -> list[SweForgeTask]:
    """Deserialize selected tasks from database storage."""

    data = json.loads(raw)
    return [SweForgeTask(**item) for item in data]


def _tasks_from_tree(records: list[object]) -> list[SweForgeTask]:
    required = {"workspace.yaml", "patch.diff", "evaluate.sh"}
    files_by_task: dict[str, set[str]] = {}
    for record in records:
        if not isinstance(record, dict) or record.get("type") != "file":
            continue
        path = str(record.get("path") or "")
        parts = path.split("/")
        if len(parts) < 3 or parts[0] != "tasks":
            continue
        files_by_task.setdefault(parts[1], set()).add(parts[2])

    tasks: list[SweForgeTask] = []
    for task_id in sorted(files_by_task):
        if not required.issubset(files_by_task[task_id]):
            continue
        tasks.append(
            SweForgeTask(
                task_id=task_id,
                docker_image=f"{settings.swe_forge_image_prefix}:{task_id}",
                prompt=f"SWE-Forge task {task_id}",
            )
        )
    return tasks
