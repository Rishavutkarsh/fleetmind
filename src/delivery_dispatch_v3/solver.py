from __future__ import annotations

import time
from collections.abc import Callable
from functools import lru_cache

from .dynamics import Counts, allocations_from_counts, legal_next_counts, move_cost, round_service_reward
from .models import HiddenRecipe, V3Action


ProgressCallback = Callable[[dict[str, object]], None]


def solve_exact(
    recipe: HiddenRecipe,
    start_round: int = 0,
    start_counts: Counts | None = None,
    progress_callback: ProgressCallback | None = None,
) -> tuple[float, list[V3Action]]:
    counts = start_counts or recipe.initial_courier_counts
    legal_cache: dict[Counts, tuple[Counts, ...]] = {}
    started_at = time.perf_counter()
    state_counter = {"visited": 0}

    @lru_cache(maxsize=None)
    def value(round_index: int, current_counts: Counts) -> tuple[float, tuple[Counts, ...]]:
        state_counter["visited"] += 1
        if progress_callback and state_counter["visited"] % 200 == 0:
            progress_callback(
                {
                    "round_index": round_index,
                    "elapsed_seconds": time.perf_counter() - started_at,
                    "states_visited": state_counter["visited"],
                }
            )

        round_template = recipe.rounds[round_index]
        if round_index >= recipe.profile.total_rounds - 1:
            reward, _, _ = round_service_reward(
                round_template=round_template,
                current_counts=current_counts,
                missed_order_penalty=recipe.profile.missed_order_penalty,
            )
            return reward, ()

        best_total = float("-inf")
        best_path: tuple[Counts, ...] = ()
        candidate_next_counts = legal_cache.get(current_counts)
        if candidate_next_counts is None:
            candidate_next_counts = legal_next_counts(recipe, current_counts)
            legal_cache[current_counts] = candidate_next_counts

        service_reward, _, _ = round_service_reward(
            round_template=round_template,
            current_counts=current_counts,
            missed_order_penalty=recipe.profile.missed_order_penalty,
        )
        for next_counts in candidate_next_counts:
            total = service_reward - move_cost(recipe, round_index, current_counts, next_counts)
            future_total, future_path = value(round_index + 1, next_counts)
            total += future_total
            if total > best_total:
                best_total = total
                best_path = (next_counts, *future_path)
        return best_total, best_path

    reward, path = value(start_round, counts)
    return reward, [allocations_from_counts(recipe, next_counts) for next_counts in path]


def best_action(recipe: HiddenRecipe, round_index: int, current_counts: Counts) -> V3Action:
    _, plan = solve_exact(recipe, start_round=round_index, start_counts=current_counts)
    return plan[0] if plan else allocations_from_counts(recipe, current_counts)
