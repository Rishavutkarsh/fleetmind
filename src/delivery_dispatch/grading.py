from __future__ import annotations

from dataclasses import dataclass, asdict
from typing import Any


@dataclass(frozen=True)
class GraderResult:
    task_id: str
    raw_reward: float
    baseline_reward: float
    target_reward: float
    score: float
    completed_orders: int | None = None
    on_time_orders: int | None = None
    late_orders: int | None = None
    expired_orders: int | None = None
    rejected_orders: int | None = None
    invalid_actions: int | None = None
    service_rate: float | None = None
    on_time_rate: float | None = None

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def clamp(value: float, low: float = 0.0, high: float = 1.0) -> float:
    return max(low, min(high, value))


def normalized_score(raw_reward: float, baseline_reward: float, target_reward: float) -> float:
    if target_reward <= baseline_reward:
        return 1.0 if raw_reward >= target_reward else 0.0
    return clamp((raw_reward - baseline_reward) / (target_reward - baseline_reward))


def grade_task(
    task_id: str,
    raw_reward: float,
    baseline_reward: float,
    target_reward: float,
    *,
    completed_orders: int | None = None,
    on_time_orders: int | None = None,
    late_orders: int | None = None,
    expired_orders: int | None = None,
    rejected_orders: int | None = None,
    invalid_actions: int | None = None,
) -> GraderResult:
    total_resolved = (completed_orders or 0) + (expired_orders or 0) + (rejected_orders or 0)
    service_rate = ((completed_orders or 0) / total_resolved) if total_resolved else None
    on_time_rate = ((on_time_orders or 0) / (completed_orders or 1)) if completed_orders else None
    return GraderResult(
        task_id=task_id,
        raw_reward=float(raw_reward),
        baseline_reward=float(baseline_reward),
        target_reward=float(target_reward),
        score=normalized_score(raw_reward, baseline_reward, target_reward),
        completed_orders=completed_orders,
        on_time_orders=on_time_orders,
        late_orders=late_orders,
        expired_orders=expired_orders,
        rejected_orders=rejected_orders,
        invalid_actions=invalid_actions,
        service_rate=service_rate,
        on_time_rate=on_time_rate,
    )


def grade_trajectory(
    task_id: str,
    trajectory_reward: float,
    baseline_reward: float,
    target_reward: float,
    stats: dict[str, int] | None = None,
) -> GraderResult:
    stats = stats or {}
    return grade_task(
        task_id=task_id,
        raw_reward=trajectory_reward,
        baseline_reward=baseline_reward,
        target_reward=target_reward,
        completed_orders=stats.get("completed_orders"),
        on_time_orders=stats.get("on_time_orders"),
        late_orders=stats.get("late_orders"),
        expired_orders=stats.get("expired_orders"),
        rejected_orders=stats.get("rejected_orders"),
        invalid_actions=stats.get("invalid_actions"),
    )


def weighted_mean(results: list[GraderResult], weights: dict[str, float]) -> float:
    if not results:
        return 0.0
    weighted_total = 0.0
    weight_total = 0.0
    for result in results:
        weight = float(weights.get(result.task_id, 1.0))
        weighted_total += result.score * weight
        weight_total += weight
    return 0.0 if weight_total == 0 else weighted_total / weight_total


def summarize_results(results: list[GraderResult], weights: dict[str, float] | None = None) -> dict[str, Any]:
    weights = weights or {}
    return {
        "tasks": [result.to_dict() for result in results],
        "overall_score": weighted_mean(results, weights),
    }


__all__ = [
    "GraderResult",
    "clamp",
    "grade_task",
    "grade_trajectory",
    "normalized_score",
    "summarize_results",
    "weighted_mean",
]
