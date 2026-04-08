from __future__ import annotations

import time
from collections.abc import Callable

from .environment import V3DeliveryDispatchEnv
from .models import V3TaskResult
from .policies import baseline_policy, heuristic_policy
from .solver import solve_exact


def grade_episode(task_id: str, seed: int, raw_reward: float) -> V3TaskResult:
    baseline_reward = rollout_policy(task_id, seed, policy_name="baseline")
    heuristic_reward = rollout_policy(task_id, seed, policy_name="heuristic")
    target_reward = optimal_reward(task_id, seed)
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


def timed_optimal_reward(task_id: str, seed: int) -> tuple[float, float]:
    started_at = time.perf_counter()
    reward = optimal_reward(task_id, seed)
    runtime_ms = (time.perf_counter() - started_at) * 1000.0
    return reward, runtime_ms


def normalize_score(raw_reward: float, baseline_reward: float, target_reward: float) -> float:
    if target_reward <= baseline_reward:
        return 1.0 if raw_reward >= target_reward else 0.0
    return max(0.0, min(1.0, (raw_reward - baseline_reward) / (target_reward - baseline_reward)))
