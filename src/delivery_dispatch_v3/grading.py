from __future__ import annotations

import time
from collections.abc import Callable
from functools import lru_cache

from .environment import V3DeliveryDispatchEnv
from .models import V3TaskResult
from .policies import baseline_policy, heuristic_policy
from .solver import solve_exact

STRICT_SCORE_EPSILON = 1e-4
BASELINE_SCORE_ANCHOR = 0.05


def grade_episode(task_id: str, seed: int, raw_reward: float) -> V3TaskResult:
    baseline_reward = cached_rollout_policy(task_id, seed, policy_name="baseline")
    heuristic_reward = cached_rollout_policy(task_id, seed, policy_name="heuristic")
    target_reward = cached_optimal_reward(task_id, seed)
    score = normalize_score(raw_reward, baseline_reward, target_reward)
    return V3TaskResult(
        task_id=task_id,
        raw_reward=raw_reward,
        baseline_reward=baseline_reward,
        target_reward=target_reward,
        score=score,
        heuristic_reward=heuristic_reward,
    )


def rollout_policy(task_id: str, seed: int, policy_name: str = "baseline") -> float:
    env = V3DeliveryDispatchEnv(default_task_id=task_id)
    observation = env.reset_internal(task_id=task_id, internal_seed=seed)
    policy = baseline_policy if policy_name == "baseline" else heuristic_policy
    while not env.done:
        result = env.step(policy(observation), grade_terminal=False)
        observation = result.observation
    return env.cumulative_reward


@lru_cache(maxsize=512)
def cached_rollout_policy(task_id: str, seed: int, policy_name: str = "baseline") -> float:
    return rollout_policy(task_id, seed, policy_name=policy_name)


def optimal_reward(
    task_id: str,
    seed: int,
    progress_callback: Callable[[dict[str, object]], None] | None = None,
) -> float:
    env = V3DeliveryDispatchEnv(default_task_id=task_id)
    env.reset_internal(task_id=task_id, internal_seed=seed)
    reward, _ = solve_exact(
        recipe=env._require_recipe(),
        start_round=env.round_index,
        start_counts=env.courier_counts,
        progress_callback=progress_callback,
    )
    return reward


@lru_cache(maxsize=512)
def cached_optimal_reward(task_id: str, seed: int) -> float:
    return optimal_reward(task_id, seed)


def timed_optimal_reward(task_id: str, seed: int) -> tuple[float, float]:
    started_at = time.perf_counter()
    reward = optimal_reward(task_id, seed)
    runtime_ms = (time.perf_counter() - started_at) * 1000.0
    return reward, runtime_ms


def normalize_score(raw_reward: float, baseline_reward: float, target_reward: float) -> float:
    lower = STRICT_SCORE_EPSILON
    baseline_anchor = BASELINE_SCORE_ANCHOR
    upper = 1.0 - STRICT_SCORE_EPSILON
    if target_reward <= baseline_reward:
        return upper if raw_reward >= target_reward else baseline_anchor

    gap = target_reward - baseline_reward
    normalized = (raw_reward - baseline_reward) / gap
    if normalized >= 0.0:
        score = baseline_anchor + normalized * (upper - baseline_anchor)
    else:
        score = baseline_anchor + normalized * (baseline_anchor - lower)
    return max(lower, min(upper, score))
