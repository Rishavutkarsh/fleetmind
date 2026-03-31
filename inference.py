from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Callable

sys.path.insert(0, str(Path(__file__).resolve().parent / "src"))

from delivery_dispatch.environment import DeliveryDispatchEnv
from delivery_dispatch.grading import grade_trajectory, summarize_results
from delivery_dispatch.llm import choose_action_with_llm, llm_configured
from delivery_dispatch.models import Observation
from delivery_dispatch.policies import ActionDict, baseline_policy, target_policy


PolicyFn = Callable[[dict], ActionDict]


def run_policy(task_id: str, policy: PolicyFn) -> tuple[float, dict[str, int]]:
    env = DeliveryDispatchEnv(scenario_name=task_id)
    observation = env.reset()

    done = False
    while not done:
        action = policy(observation.model_dump(mode="json"))
        result = env.step(action)
        observation = result.observation
        done = result.done

    return env.cumulative_reward, dict(env.stats)


def run_llm_policy(task_id: str) -> tuple[float, dict[str, int]]:
    env = DeliveryDispatchEnv(scenario_name=task_id)
    observation = env.reset()

    done = False
    while not done:
        action = choose_action_with_llm(observation)
        result = env.step(action)
        observation = result.observation
        done = result.done

    return env.cumulative_reward, dict(env.stats)


def score_tasks(policy_name: str) -> dict:
    weights = {
        "low_demand": 0.2,
        "high_demand": 0.3,
        "hotspot_congestion": 0.5,
    }
    task_results = []

    for task_id in weights:
        baseline_reward, _ = run_policy(task_id, baseline_policy)
        target_reward, _ = run_policy(task_id, target_policy)
        lower_bound = min(baseline_reward, target_reward)
        upper_bound = max(baseline_reward, target_reward)
        if policy_name == "llm":
            raw_reward, raw_stats = run_llm_policy(task_id)
        elif policy_name == "target":
            raw_reward, raw_stats = run_policy(task_id, target_policy)
        else:
            raw_reward, raw_stats = run_policy(task_id, baseline_policy)

        task_results.append(
            grade_trajectory(
                task_id=task_id,
                trajectory_reward=raw_reward,
                baseline_reward=lower_bound,
                target_reward=upper_bound,
                stats=raw_stats,
            )
        )

    return summarize_results(task_results, weights)


def main() -> None:
    policy_name = "llm" if llm_configured() else "baseline"
    print(json.dumps({"policy": policy_name, **score_tasks(policy_name)}, indent=2))


if __name__ == "__main__":
    main()
