from __future__ import annotations

import random

from .models import DifficultyProfile, HiddenRecipe, RoundTemplate, WorldRegime, ZoneSpec


PROFILES: dict[str, DifficultyProfile] = {
    "v3_easy_dispatch": DifficultyProfile(
        task_id="v3_easy_dispatch",
        zone_count=4,
        courier_count=5,
        total_rounds=6,
        max_repositions_per_round=2,
        missed_order_penalty=5.0,
        move_cost_weight=1.0,
        runtime_budget_ms=250.0,
    ),
    "v3_medium_dispatch": DifficultyProfile(
        task_id="v3_medium_dispatch",
        zone_count=4,
        courier_count=6,
        total_rounds=8,
        max_repositions_per_round=2,
        missed_order_penalty=5.5,
        move_cost_weight=1.1,
        runtime_budget_ms=400.0,
    ),
    "v3_hard_dispatch": DifficultyProfile(
        task_id="v3_hard_dispatch",
        zone_count=4,
        courier_count=6,
        total_rounds=10,
        max_repositions_per_round=3,
        missed_order_penalty=6.0,
        move_cost_weight=1.05,
        runtime_budget_ms=900.0,
    ),
}

WORLD_REGIMES: tuple[WorldRegime, ...] = (
    "visible_ramp",
    "decoy_then_shift",
    "premium_late_surge",
    "congested_pivot",
)


def generate_recipe(task_id: str, seed: int) -> HiddenRecipe:
    profile = PROFILES[task_id]
    rng = random.Random(f"{task_id}:{seed}")
    zone_specs = _zone_specs(profile.zone_count)
    indices = list(range(profile.zone_count))
    hot_zone_index = rng.randrange(profile.zone_count)
    decoy_choices = [index for index in indices if index != hot_zone_index]
    decoy_zone_index = rng.choice(decoy_choices)
    support_choices = [index for index in decoy_choices if index != decoy_zone_index]
    support_zone_index = rng.choice(support_choices)
    premium_zone_index = hot_zone_index if rng.random() < 0.7 else support_zone_index
    world_regime = WORLD_REGIMES[seed % len(WORLD_REGIMES)]

    rounds = tuple(
        _build_round(
            profile=profile,
            round_index=round_index,
            rng=rng,
            world_regime=world_regime,
            hot_zone_index=hot_zone_index,
            decoy_zone_index=decoy_zone_index,
            support_zone_index=support_zone_index,
            premium_zone_index=premium_zone_index,
        )
        for round_index in range(profile.total_rounds)
    )
    initial_courier_counts = _initial_counts(profile.courier_count, profile.zone_count, hot_zone_index)
    return HiddenRecipe(
        task_id=task_id,
        seed=seed,
        profile=profile,
        world_regime=world_regime,
        hot_zone_index=hot_zone_index,
        decoy_zone_index=decoy_zone_index,
        support_zone_index=support_zone_index,
        premium_zone_index=premium_zone_index,
        zone_specs=zone_specs,
        initial_courier_counts=initial_courier_counts,
        rounds=rounds,
    )


def _zone_specs(zone_count: int) -> tuple[ZoneSpec, ...]:
    base = [
        ZoneSpec(zone_id="north", label="North", position=(0, 2)),
        ZoneSpec(zone_id="east", label="East", position=(2, 0)),
        ZoneSpec(zone_id="south", label="South", position=(4, 2)),
        ZoneSpec(zone_id="west", label="West", position=(2, 4)),
        ZoneSpec(zone_id="central", label="Central", position=(2, 2)),
    ]
    return tuple(base[:zone_count])


def _initial_counts(courier_count: int, zone_count: int, hot_zone_index: int) -> tuple[int, ...]:
    counts = [courier_count // zone_count] * zone_count
    for index in range(courier_count % zone_count):
        counts[index] += 1
    if zone_count > 1 and counts[hot_zone_index] > 0:
        shift_from = (hot_zone_index + 1) % zone_count
        if counts[shift_from] > 0:
            counts[shift_from] -= 1
            counts[hot_zone_index] += 1
    return tuple(counts)


def _build_round(
    profile: DifficultyProfile,
    round_index: int,
    rng: random.Random,
    world_regime: WorldRegime,
    hot_zone_index: int,
    decoy_zone_index: int,
    support_zone_index: int,
    premium_zone_index: int,
) -> RoundTemplate:
    progress = round_index / max(1, profile.total_rounds - 1)
    visible_orders: list[int] = []
    reward_per_order: list[float] = []
    congestion_multiplier: list[float] = []

    for zone_index in range(profile.zone_count):
        base = 1
        hot_signal = _hot_component(profile.task_id, progress, world_regime, zone_index == hot_zone_index)
        decoy_signal = _decoy_component(profile.task_id, progress, world_regime, zone_index == decoy_zone_index)
        support_signal = 1 if zone_index == support_zone_index and progress > 0.3 else 0
        noise = rng.randint(0, 1 if profile.task_id == "v3_easy_dispatch" else 2)
        demand = max(0, base + hot_signal + decoy_signal + support_signal + noise)
        visible_orders.append(demand)

        premium_bonus = 0.0
        if zone_index == premium_zone_index and progress >= (0.45 if profile.task_id == "v3_hard_dispatch" else 0.3):
            premium_bonus = 2.5 if profile.task_id == "v3_easy_dispatch" else 4.5
        reward_per_order.append(8.0 + premium_bonus)

        congestion = 1.0
        if world_regime == "congested_pivot" and progress >= 0.35 and zone_index in {decoy_zone_index, hot_zone_index}:
            if profile.task_id == "v3_hard_dispatch":
                congestion = 1.35 if zone_index == hot_zone_index and progress < 0.6 else 1.18
            else:
                congestion = 1.5 if zone_index == hot_zone_index and progress < 0.6 else 1.25
        elif world_regime != "congested_pivot" and zone_index == decoy_zone_index and progress < 0.4:
            congestion = 1.15
        congestion_multiplier.append(congestion)

    return RoundTemplate(
        round_index=round_index,
        visible_orders_by_zone=tuple(visible_orders),
        reward_per_order_by_zone=tuple(reward_per_order),
        congestion_multiplier_by_zone=tuple(congestion_multiplier),
    )


def _hot_component(task_id: str, progress: float, world_regime: WorldRegime, is_hot_zone: bool) -> int:
    if not is_hot_zone:
        return 0
    if task_id == "v3_easy_dispatch":
        return 1 + round(3 * progress)
    if task_id == "v3_medium_dispatch":
        if world_regime == "visible_ramp":
            return round(4 * progress)
        return max(0, round(5 * (progress - 0.25)))
    if world_regime in {"decoy_then_shift", "congested_pivot"}:
        return max(0, round(7 * (progress - 0.32)))
    return max(0, round(6 * (progress - 0.18)))


def _decoy_component(task_id: str, progress: float, world_regime: WorldRegime, is_decoy_zone: bool) -> int:
    if not is_decoy_zone:
        return 0
    if task_id == "v3_easy_dispatch":
        return 1 if progress < 0.35 else 0
    if task_id == "v3_medium_dispatch":
        return 2 if progress < 0.45 else 0
    if world_regime == "decoy_then_shift":
        return 3 if progress < 0.55 else 0
    if world_regime == "premium_late_surge":
        return 2 if progress < 0.4 else 1
    return 2 if progress < 0.5 else 0
