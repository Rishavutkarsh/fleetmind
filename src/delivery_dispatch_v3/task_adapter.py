from __future__ import annotations

PUBLIC_TO_INTERNAL_TASK_ID: dict[str, str] = {
    "easy_dispatch": "v3_easy_dispatch",
    "medium_dispatch": "v3_medium_dispatch",
    "hard_dispatch": "v3_hard_dispatch",
}

INTERNAL_TO_PUBLIC_TASK_ID: dict[str, str] = {
    internal: public for public, internal in PUBLIC_TO_INTERNAL_TASK_ID.items()
}

PUBLIC_TASK_IDS: tuple[str, ...] = tuple(PUBLIC_TO_INTERNAL_TASK_ID)


def to_internal_task_id(task_id: str) -> str:
    return PUBLIC_TO_INTERNAL_TASK_ID.get(task_id, task_id)


def to_public_task_id(task_id: str) -> str:
    return INTERNAL_TO_PUBLIC_TASK_ID.get(task_id, task_id)


def is_public_task_id(task_id: str) -> bool:
    return task_id in PUBLIC_TO_INTERNAL_TASK_ID
