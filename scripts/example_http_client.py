from __future__ import annotations

import argparse
from typing import Any

import requests


def choose_action(observation: dict[str, Any]) -> dict[str, Any]:
    agents = [agent for agent in observation["agents"] if agent["status"] == "idle"]
    orders = [
        order
        for order in observation["orders"]
        if order["status"] == "unassigned"
    ]

    def manhattan(a: tuple[int, int], b: tuple[int, int]) -> int:
        return abs(a[0] - b[0]) + abs(a[1] - b[1])

    assignments: list[dict[str, str]] = []
    used_orders: set[str] = set()

    ranked_orders = sorted(
        orders,
        key=lambda order: (
            order["deadline"],
            -order["reward_value"],
        ),
    )

    for agent in agents:
        agent_location = tuple(agent["location"])
        best_order = None
        best_score = None
        for order in ranked_orders:
            if order["order_id"] in used_orders:
                continue
            pickup = tuple(order["pickup_location"])
            drop = tuple(order["drop_location"])
            score = (
                manhattan(agent_location, pickup)
                + manhattan(pickup, drop)
                - 0.35 * float(order["reward_value"])
            )
            if best_score is None or score < best_score:
                best_score = score
                best_order = order

        if best_order is None:
            continue

        assignments.append(
            {"agent_id": agent["agent_id"], "order_id": best_order["order_id"]}
        )
        used_orders.add(best_order["order_id"])

    return {"assignments": assignments, "rejections": []}


def main() -> None:
    parser = argparse.ArgumentParser(description="Minimal Fleetmind HTTP API client.")
    parser.add_argument("--base_url", default="http://127.0.0.1:8000")
    parser.add_argument("--task_id", default="high_demand")
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--max_decision_steps", type=int, default=None)
    args = parser.parse_args()

    params = {"task_id": args.task_id}
    if args.seed is not None:
        params["seed"] = args.seed
    if args.max_decision_steps is not None:
        params["max_decision_steps"] = args.max_decision_steps

    reset_response = requests.post(f"{args.base_url}/reset", params=params, timeout=30)
    reset_response.raise_for_status()
    observation = reset_response.json()

    print(
        {
            "event": "reset",
            "task_id": args.task_id,
            "used_seed": observation["scenario_info"].get("used_seed"),
            "max_decision_steps": observation.get("max_decision_steps"),
        }
    )

    done = False
    while not done:
        action = choose_action(observation)
        step_response = requests.post(
            f"{args.base_url}/step",
            json=action,
            timeout=30,
        )
        step_response.raise_for_status()
        payload = step_response.json()
        observation = payload["observation"]
        done = payload["done"]
        print(
            {
                "decision_step": observation["decision_step"],
                "step_reward": payload["reward"]["step_reward"],
                "cumulative_reward": payload["reward"]["cumulative_reward"],
                "errors": observation["feedback"]["error_summary"],
            }
        )

    print(
        {
            "event": "done",
            "task_id": observation["task_id"],
            "used_seed": observation["scenario_info"].get("used_seed"),
            "metrics": observation["metrics"],
            "cumulative_reward": observation["feedback"]["cumulative_reward"],
        }
    )


if __name__ == "__main__":
    main()
