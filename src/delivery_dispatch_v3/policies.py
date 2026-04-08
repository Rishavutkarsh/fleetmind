from __future__ import annotations

from .dynamics import Counts, count_moved
from .models import V3Action, V3Observation


def baseline_policy(observation: V3Observation) -> V3Action:
    current_counts = tuple(zone.courier_count for zone in observation.zones)
    current_signal = [zone.visible_orders * zone.reward_per_order / zone.congestion_multiplier for zone in observation.zones]
    target_counts = _weighted_target_counts(
        current_counts=current_counts,
        weights=current_signal,
        total_couriers=observation.scenario_info.total_couriers,
        move_cap=observation.scenario_info.max_repositions_per_round,
    )
    return V3Action(
        target_allocations=[
            {"zone_id": zone.zone_id, "courier_count": target}
            for zone, target in zip(observation.zones, target_counts, strict=True)
        ]
    )


def heuristic_policy(observation: V3Observation) -> V3Action:
    current_counts = tuple(zone.courier_count for zone in observation.zones)
    weights = [
        0.65 * zone.visible_orders * zone.reward_per_order / zone.congestion_multiplier + 0.35 * zone.courier_count
        for zone in observation.zones
    ]
    target_counts = _weighted_target_counts(
        current_counts=current_counts,
        weights=weights,
        total_couriers=observation.scenario_info.total_couriers,
        move_cap=observation.scenario_info.max_repositions_per_round,
    )
    return V3Action(
        target_allocations=[
            {"zone_id": zone.zone_id, "courier_count": target}
            for zone, target in zip(observation.zones, target_counts, strict=True)
        ]
    )


def _weighted_target_counts(
    current_counts: Counts,
    weights: list[float],
    total_couriers: int,
    move_cap: int,
) -> Counts:
    safe_weights = [max(0.1, weight) for weight in weights]
    total_weight = sum(safe_weights)
    raw = [weight / total_weight * total_couriers for weight in safe_weights]
    counts = [int(value) for value in raw]
    while sum(counts) < total_couriers:
        best_index = max(range(len(counts)), key=lambda index: raw[index] - counts[index])
        counts[best_index] += 1
    while sum(counts) > total_couriers:
        worst_index = max(range(len(counts)), key=lambda index: counts[index] - raw[index])
        counts[worst_index] -= 1

    target = tuple(counts)
    if count_moved(current_counts, target) <= move_cap:
        return target

    mutable = list(current_counts)
    desired = list(target)
    while count_moved(tuple(mutable), tuple(desired)) > move_cap:
        donor = max(
            range(len(mutable)),
            key=lambda index: max(0, mutable[index] - desired[index]),
        )
        receiver = max(
            range(len(mutable)),
            key=lambda index: max(0, desired[index] - mutable[index]),
        )
        if mutable[donor] <= desired[donor] or desired[receiver] <= mutable[receiver]:
            break
        desired[receiver] -= 1
        desired[donor] += 1
    return tuple(desired)
