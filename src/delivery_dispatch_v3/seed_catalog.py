from __future__ import annotations

import secrets

from .generator import generate_recipe
from .grading import rollout_policy, timed_optimal_reward
from .models import SeedMetadata


TRAIN_SEEDS: dict[str, tuple[int, ...]] = {
    "v3_easy_dispatch": (143, 129, 111, 147, 155, 131, 107, 141, 151, 119, 139, 123),
    "v3_medium_dispatch": (231, 237, 243, 219, 203, 215, 209, 255, 227, 251, 235, 233),
    "v3_hard_dispatch": (311, 325, 343, 359, 315, 303, 345, 331, 337, 319, 339, 307),
}

EVAL_SEEDS: dict[str, tuple[int, ...]] = {
    "v3_easy_dispatch": (423, 409, 407, 401, 435, 427, 419, 403),
    "v3_medium_dispatch": (529, 511, 533, 505, 507, 539, 515, 537),
    "v3_hard_dispatch": (607, 625, 603, 615, 601, 635, 631, 623),
}

OFFICIAL_SEEDS: dict[str, tuple[int, ...]] = {
    "v3_easy_dispatch": (703, 737, 723, 709),
    "v3_medium_dispatch": (819, 837, 807, 827),
    "v3_hard_dispatch": (935, 913, 907, 919),
}

TEST_SEEDS: dict[str, tuple[int, ...]] = {
    "v3_easy_dispatch": (791, 753, 805, 767, 793, 775, 787, 771, 807, 779, 777, 819, 783, 815, 765, 751),
    "v3_medium_dispatch": (875, 857, 879, 867, 859, 849, 851, 893, 887, 843, 871, 885, 847, 855, 873, 863),
    "v3_hard_dispatch": (991, 981, 971, 1011, 975, 1019, 1013, 979, 973, 977, 989, 1017, 965, 1015, 1009, 993),
}

CANDIDATE_SEEDS: dict[str, dict[str, tuple[int, ...]]] = {
    "train": {
        "v3_easy_dispatch": tuple(range(101, 161, 2)),
        "v3_medium_dispatch": tuple(range(201, 261, 2)),
        "v3_hard_dispatch": tuple(range(301, 361, 2)),
    },
    "eval": {
        "v3_easy_dispatch": tuple(range(401, 441, 2)),
        "v3_medium_dispatch": tuple(range(501, 541, 2)),
        "v3_hard_dispatch": tuple(range(601, 641, 2)),
    },
    "official": {
        "v3_easy_dispatch": tuple(range(701, 741, 2)),
        "v3_medium_dispatch": tuple(range(801, 841, 2)),
        "v3_hard_dispatch": tuple(range(901, 941, 2)),
    },
    "test": {
        "v3_easy_dispatch": tuple(range(741, 821, 2)),
        "v3_medium_dispatch": tuple(range(841, 921, 2)),
        "v3_hard_dispatch": tuple(range(941, 1021, 2)),
    },
}

SEED_POOLS: dict[str, dict[str, tuple[int, ...]]] = {
    "train": TRAIN_SEEDS,
    "eval": EVAL_SEEDS,
    "official": OFFICIAL_SEEDS,
    "test": TEST_SEEDS,
}

TASK_IDS: tuple[str, ...] = (
    "v3_easy_dispatch",
    "v3_medium_dispatch",
    "v3_hard_dispatch",
)


def build_seed_metadata(task_id: str, seed: int) -> SeedMetadata:
    recipe = generate_recipe(task_id, seed)
    baseline_reward = rollout_policy(task_id, seed, policy_name="baseline")
    heuristic_reward = rollout_policy(task_id, seed, policy_name="heuristic")
    target_reward, runtime_ms = timed_optimal_reward(task_id, seed)
    admissible = is_seed_admissible(
        target_reward=target_reward,
        baseline_reward=baseline_reward,
        heuristic_reward=heuristic_reward,
        runtime_ms=runtime_ms,
        runtime_budget_ms=recipe.profile.runtime_budget_ms,
    )
    return SeedMetadata(
        task_id=task_id,
        seed=seed,
        world_regime=recipe.world_regime,
        hot_zone=recipe.zone_specs[recipe.hot_zone_index].label,
        decoy_zone=recipe.zone_specs[recipe.decoy_zone_index].label,
        premium_zone=recipe.zone_specs[recipe.premium_zone_index].label,
        baseline_reward=baseline_reward,
        heuristic_reward=heuristic_reward,
        target_reward=target_reward,
        score_gap=target_reward - baseline_reward,
        heuristic_gap=target_reward - heuristic_reward,
        solver_runtime_ms=runtime_ms,
        runtime_budget_ms=recipe.profile.runtime_budget_ms,
        admissible=admissible,
    )


def is_seed_admissible(
    target_reward: float,
    baseline_reward: float,
    heuristic_reward: float,
    runtime_ms: float,
    runtime_budget_ms: float,
) -> bool:
    return (
        target_reward - baseline_reward >= 12.0
        and target_reward - heuristic_reward >= 8.0
        and runtime_ms <= runtime_budget_ms
    )


def curate_seed_pool(task_id: str, candidate_seeds: tuple[int, ...], limit: int) -> tuple[int, ...]:
    metadatas = [build_seed_metadata(task_id, seed) for seed in candidate_seeds]
    admissible = [metadata for metadata in metadatas if metadata.admissible]
    admissible.sort(
        key=lambda metadata: (
            metadata.score_gap,
            metadata.heuristic_gap,
            -metadata.solver_runtime_ms,
        ),
        reverse=True,
    )
    chosen: list[int] = []
    seen_regimes: set[str] = set()
    for metadata in admissible:
        if metadata.world_regime not in seen_regimes or len(seen_regimes) >= 4:
            chosen.append(metadata.seed)
            seen_regimes.add(metadata.world_regime)
        if len(chosen) >= limit:
            return tuple(chosen)
    for metadata in admissible:
        if metadata.seed in chosen:
            continue
        chosen.append(metadata.seed)
        if len(chosen) >= limit:
            break
    return tuple(chosen)


def resolve_curated_seed(task_id: str, external_seed: int, pool_name: str = "test") -> int:
    pool = SEED_POOLS[pool_name][task_id]
    if not pool:
        raise ValueError(f"No curated seeds for {task_id} in pool '{pool_name}'")
    task_offset = sum(ord(character) for character in task_id)
    index = ((external_seed * 1315423911) + task_offset) % len(pool)
    return pool[index]


def resolve_task_id(external_seed: int) -> str:
    return TASK_IDS[external_seed % len(TASK_IDS)]


def choose_random_task_id() -> str:
    return secrets.choice(TASK_IDS)


def choose_random_curated_seed(task_id: str, pool_name: str = "test") -> int:
    pool = SEED_POOLS[pool_name][task_id]
    if not pool:
        raise ValueError(f"No curated seeds for {task_id} in pool '{pool_name}'")
    return secrets.choice(pool)
