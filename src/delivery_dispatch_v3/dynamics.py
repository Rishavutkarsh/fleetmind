from __future__ import annotations

from functools import lru_cache

from .models import HiddenRecipe, RoundTemplate, V3Action, V3Observation, ZoneAllocation


Counts = tuple[int, ...]


def all_distributions(total_couriers: int, zone_count: int) -> tuple[Counts, ...]:
    return _all_distributions(total_couriers, zone_count)


@lru_cache(maxsize=None)
def _all_distributions(total_couriers: int, zone_count: int) -> tuple[Counts, ...]:
    if zone_count == 1:
        return ((total_couriers,),)
    layouts: list[Counts] = []
    for head in range(total_couriers + 1):
        for tail in _all_distributions(total_couriers - head, zone_count - 1):
            layouts.append((head, *tail))
    return tuple(layouts)


def parse_target_counts(observation: V3Observation, action: V3Action) -> Counts | None:
    zone_ids = [zone.zone_id for zone in observation.zones]
    counts_by_zone: dict[str, int] = {}
    for allocation in action.target_allocations:
        if allocation.zone_id in counts_by_zone:
            return None
        if allocation.courier_count < 0:
            return None
        counts_by_zone[allocation.zone_id] = allocation.courier_count
    if set(counts_by_zone) != set(zone_ids):
        return None
    counts = tuple(counts_by_zone[zone_id] for zone_id in zone_ids)
    if sum(counts) != observation.scenario_info.total_couriers:
        return None
    return counts


def allocations_from_counts(recipe: HiddenRecipe, counts: Counts) -> V3Action:
    return V3Action(
        target_allocations=[
            ZoneAllocation(zone_id=zone.zone_id, courier_count=count)
            for zone, count in zip(recipe.zone_specs, counts, strict=True)
        ]
    )


def count_moved(current_counts: Counts, target_counts: Counts) -> int:
    return sum(max(0, current - target) for current, target in zip(current_counts, target_counts, strict=True))


def round_service_reward(
    round_template: RoundTemplate,
    current_counts: Counts,
    missed_order_penalty: float,
) -> tuple[float, tuple[int, ...], tuple[int, ...]]:
    served = tuple(
        min(couriers, visible_orders)
        for couriers, visible_orders in zip(current_counts, round_template.visible_orders_by_zone, strict=True)
    )
    missed = tuple(
        max(0, visible_orders - couriers)
        for couriers, visible_orders in zip(current_counts, round_template.visible_orders_by_zone, strict=True)
    )
    reward = 0.0
    for served_orders, missed_orders, reward_per_order in zip(
        served,
        missed,
        round_template.reward_per_order_by_zone,
        strict=True,
    ):
        reward += served_orders * reward_per_order
        reward -= missed_orders * missed_order_penalty
    return reward, served, missed


def move_cost(recipe: HiddenRecipe, round_index: int, current_counts: Counts, target_counts: Counts) -> float:
    if current_counts == target_counts:
        return 0.0
    round_template = recipe.rounds[round_index]
    surplus = tuple(max(0, current - target) for current, target in zip(current_counts, target_counts, strict=True))
    deficit = tuple(max(0, target - current) for current, target in zip(current_counts, target_counts, strict=True))
    origin_slots = expand_counts(surplus)
    target_slots = expand_counts(deficit)
    return recipe.profile.move_cost_weight * _assignment_cost(
        tuple(origin_slots),
        tuple(target_slots),
        tuple(zone.position for zone in recipe.zone_specs),
        round_template.congestion_multiplier_by_zone,
    )


def legal_next_counts(recipe: HiddenRecipe, current_counts: Counts) -> tuple[Counts, ...]:
    all_counts = all_distributions(recipe.profile.courier_count, recipe.profile.zone_count)
    return tuple(
        counts
        for counts in all_counts
        if count_moved(current_counts, counts) <= recipe.profile.max_repositions_per_round
    )


def pressure_label(round_template: RoundTemplate, zone_specs: tuple) -> str:
    peak_index = max(range(len(zone_specs)), key=lambda index: round_template.visible_orders_by_zone[index])
    peak_zone = zone_specs[peak_index]
    peak_orders = round_template.visible_orders_by_zone[peak_index]
    return f"{peak_zone.label} leads with {peak_orders} visible orders"


def expand_counts(counts: Counts) -> list[int]:
    slots: list[int] = []
    for zone_index, count in enumerate(counts):
        slots.extend([zone_index] * count)
    return slots


@lru_cache(maxsize=None)
def _assignment_cost(
    origins: tuple[int, ...],
    targets: tuple[int, ...],
    positions: tuple[tuple[int, int], ...],
    congestion_multiplier_by_zone: tuple[float, ...],
) -> float:
    courier_count = len(origins)
    if courier_count == 0:
        return 0.0

    @lru_cache(maxsize=None)
    def solve(origin_index: int, used_mask: int) -> float:
        if origin_index >= courier_count:
            return 0.0
        best = float("inf")
        for target_index in range(courier_count):
            if used_mask & (1 << target_index):
                continue
            origin_zone = origins[origin_index]
            target_zone = targets[target_index]
            distance = manhattan(positions[origin_zone], positions[target_zone])
            congestion_multiplier = congestion_multiplier_by_zone[target_zone]
            candidate = distance * congestion_multiplier + solve(origin_index + 1, used_mask | (1 << target_index))
            if candidate < best:
                best = candidate
        return best

    return solve(0, 0)


def manhattan(a: tuple[int, int], b: tuple[int, int]) -> int:
    return abs(a[0] - b[0]) + abs(a[1] - b[1])
