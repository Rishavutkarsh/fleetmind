from __future__ import annotations

import json
import os
import sys
from pathlib import Path
from typing import Any, Callable

from openai import OpenAI

sys.path.insert(0, str(Path(__file__).resolve().parent / "src"))

from delivery_dispatch_v3.environment import V3DeliveryDispatchEnv
from delivery_dispatch_v3.models import V3Action, V3Observation
from delivery_dispatch_v3.policies import heuristic_policy


API_BASE_URL = os.getenv("API_BASE_URL")
MODEL_NAME = os.getenv("MODEL_NAME", "gpt-4.1-mini")
API_KEY = os.getenv("HF_TOKEN") or os.getenv("OPENAI_API_KEY")
SUCCESS_SCORE_THRESHOLD = 0.1

ENV_NAME = "fleetmind_v3"
EVALUATION_PUBLIC_SEEDS = {
    "easy_dispatch": 17031,
    "medium_dispatch": 27031,
    "hard_dispatch": 37031,
}
PolicyFn = Callable[[V3Observation], V3Action]
SYSTEM_PROMPT = (
    "You are playing a delivery fleet allocation benchmark. "
    "Return JSON only with the shape "
    "{\"target_allocations\": [{\"zone_id\": \"...\", \"courier_count\": 0}]}. "
    "You will be given zone-level courier counts, visible demand, per-order rewards, "
    "and congestion multipliers. Optimize long-term cumulative reward, not just immediate demand. "
    "Respect the total courier count and include every zone exactly once."
)


def _format_bool(value: bool) -> str:
    return "true" if value else "false"


def _format_reward(value: float) -> str:
    return f"{value:.2f}"


def _action_str(action: V3Action) -> str:
    return json.dumps(action.model_dump(mode="json"), separators=(",", ":"))


def _print_start(task_id: str) -> None:
    print(f"[START] task={task_id} env={ENV_NAME} model={MODEL_NAME}", flush=True)


def _print_step(step_index: int, action: V3Action, reward: float, done: bool, error: str | None) -> None:
    error_value = error if error is not None else "null"
    print(
        f"[STEP] step={step_index} action={_action_str(action)} "
        f"reward={_format_reward(reward)} done={_format_bool(done)} error={error_value}",
        flush=True,
    )


def _print_end(success: bool, rewards: list[float], score: float | None = None) -> None:
    reward_values = ",".join(_format_reward(value) for value in rewards)
    score_value = "null" if score is None else f"{score:.4f}"
    print(
        f"[END] success={_format_bool(success)} steps={len(rewards)} score={score_value} rewards={reward_values}",
        flush=True,
    )


def llm_configured() -> bool:
    return bool(API_KEY and MODEL_NAME)


def build_client() -> OpenAI:
    kwargs: dict[str, Any] = {"api_key": API_KEY}
    if API_BASE_URL:
        kwargs["base_url"] = API_BASE_URL
    return OpenAI(**kwargs)


def parse_action(raw_text: str) -> V3Action:
    try:
        payload: dict[str, Any] = json.loads(raw_text)
    except json.JSONDecodeError:
        start = raw_text.find("{")
        end = raw_text.rfind("}")
        if start == -1 or end == -1 or end <= start:
            return V3Action()
        try:
            payload = json.loads(raw_text[start : end + 1])
        except json.JSONDecodeError:
            return V3Action()
    try:
        return V3Action.model_validate(payload)
    except Exception:
        return V3Action()


def choose_action_with_llm(observation: V3Observation) -> V3Action:
    client = build_client()
    response = client.chat.completions.create(
        model=MODEL_NAME,
        messages=[
            {"role": "system", "content": SYSTEM_PROMPT},
            {
                "role": "user",
                "content": json.dumps(observation.model_dump(mode="json"), separators=(",", ":")),
            },
        ],
        temperature=0.1,
    )
    raw_text = response.choices[0].message.content or ""
    return parse_action(raw_text)


def fallback_policy(observation: V3Observation) -> V3Action:
    return heuristic_policy(observation)


def choose_action(observation: V3Observation, prefer_llm: bool) -> tuple[V3Action, str | None]:
    if not prefer_llm or not llm_configured():
        return fallback_policy(observation), None if not prefer_llm else "LLM config missing; using deterministic fallback"
    try:
        return choose_action_with_llm(observation), None
    except Exception as exc:
        return fallback_policy(observation), str(exc)


def run_task(task_id: str, seed: int, prefer_llm: bool = True) -> dict[str, Any]:
    env = V3DeliveryDispatchEnv(default_task_id=task_id)
    observation = env.reset(task_id=task_id, seed=seed)
    rewards: list[float] = []
    step_index = 0
    success = False
    final_summary: dict[str, Any] | None = None

    _print_start(task_id)

    try:
        done = False
        while not done:
            step_index += 1
            action, error = choose_action(observation, prefer_llm=prefer_llm)
            result = env.step(action)
            observation = result.observation
            rewards.append(result.reward.step_reward)
            done = result.done
            if done:
                final_summary = result.info.get("episode_summary") if isinstance(result.info, dict) else None
            _print_step(step_index, action, result.reward.step_reward, done, error)
        success = True
    except Exception as exc:
        fallback_action = V3Action()
        _print_step(step_index + 1, fallback_action, 0.0, True, str(exc))
    finally:
        score = None if final_summary is None else float(final_summary["graded_score"])
        success = success and score is not None and score >= SUCCESS_SCORE_THRESHOLD
        _print_end(success, rewards, score=score)

    return {
        "task_id": task_id,
        "seed": seed,
        "raw_reward": 0.0 if final_summary is None else float(final_summary["raw_reward"]),
        "baseline_reward": 0.0 if final_summary is None else float(final_summary["baseline_reward"]),
        "target_reward": 0.0 if final_summary is None else float(final_summary["target_reward"]),
        "score": 0.0 if final_summary is None else float(final_summary["graded_score"]),
        "heuristic_reward": None if final_summary is None else final_summary.get("heuristic_reward"),
    }


def score_tasks(policy_name: str = "baseline") -> dict[str, Any]:
    prefer_llm = policy_name != "baseline"
    task_results: list[dict[str, Any]] = []
    for task_id, seed in EVALUATION_PUBLIC_SEEDS.items():
        task_results.append(run_task(task_id=task_id, seed=seed, prefer_llm=prefer_llm))
    overall_score = sum(task["score"] for task in task_results) / len(task_results)
    return {
        "tasks": task_results,
        "overall_score": overall_score,
        "mode": "llm-first" if prefer_llm else "deterministic-fallback",
    }


def main() -> None:
    for task_id, seed in EVALUATION_PUBLIC_SEEDS.items():
        run_task(task_id=task_id, seed=seed, prefer_llm=True)


if __name__ == "__main__":
    main()
