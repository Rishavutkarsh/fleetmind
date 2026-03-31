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
EVALUATION_SEEDS = {
    "low_demand": 101,
    "high_demand": 202,
    "hotspot_congestion": 303,
}


def run_policy(task_id: str, policy: PolicyFn, seed: int | None = None) -> tuple[float, dict[str, int]]:
    env = DeliveryDispatchEnv(scenario_name=task_id)
    observation = env.reset(seed=seed)

    done = False
    while not done:
        action = policy(observation.model_dump(mode="json"))
        result = env.step(action)
        observation = result.observation
        done = result.done

    return env.cumulative_reward, dict(env.stats)


def run_llm_policy(task_id: str, seed: int | None = None) -> tuple[float, dict[str, int], dict[str, object]]:
    env = DeliveryDispatchEnv(scenario_name=task_id)
    observation = env.reset(seed=seed)
    fallback_used = False
    fallback_reason = ""

    done = False
    while not done:
        if fallback_used:
            action = target_policy(observation.model_dump(mode="json"))
        else:
            try:
                action = choose_action_with_llm(observation)
            except Exception as exc:
                fallback_used = True
                fallback_reason = f"{type(exc).__name__}: {exc}"
                action = target_policy(observation.model_dump(mode="json"))
        result = env.step(action)
        observation = result.observation
        done = result.done

    return env.cumulative_reward, dict(env.stats), {
        "fallback_used": fallback_used,
        "fallback_reason": fallback_reason,
    }


def score_tasks(policy_name: str) -> dict:
    weights = {
        "low_demand": 0.2,
        "high_demand": 0.3,
        "hotspot_congestion": 0.5,
    }
    task_results = []
    llm_runtime = {
        "configured": llm_configured(),
        "fallback_used": False,
        "fallback_reasons": {},
    }

    for task_id in weights:
        task_seed = EVALUATION_SEEDS[task_id]
        baseline_reward, _ = run_policy(task_id, baseline_policy, seed=task_seed)
        target_reward, _ = run_policy(task_id, target_policy, seed=task_seed)
        lower_bound = min(baseline_reward, target_reward)
        upper_bound = max(baseline_reward, target_reward)
        if policy_name == "llm":
            raw_reward, raw_stats, llm_meta = run_llm_policy(task_id, seed=task_seed)
            if llm_meta["fallback_used"]:
                llm_runtime["fallback_used"] = True
                llm_runtime["fallback_reasons"][task_id] = llm_meta["fallback_reason"]
        elif policy_name == "target":
            raw_reward, raw_stats = run_policy(task_id, target_policy, seed=task_seed)
        else:
            raw_reward, raw_stats = run_policy(task_id, baseline_policy, seed=task_seed)

        task_results.append(
            grade_trajectory(
                task_id=task_id,
                trajectory_reward=raw_reward,
                baseline_reward=lower_bound,
                target_reward=upper_bound,
                stats=raw_stats,
            )
        )

    result = summarize_results(task_results, weights)
    if policy_name == "llm":
        result["llm_runtime"] = llm_runtime
    return result


def main() -> None:
    policy_name = "llm" if llm_configured() else "baseline"
    print(json.dumps({"policy": policy_name, **score_tasks(policy_name)}, indent=2))


if __name__ == "__main__":
    main()
