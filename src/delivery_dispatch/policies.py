from __future__ import annotations

from dataclasses import dataclass
from typing import Any


ActionDict = dict[str, list[dict[str, str]]]
Point = tuple[int, int]


@dataclass(frozen=True)
class CandidateAssignment:
    agent_id: str
    order_id: str
    score: float
    estimated_cost: int
    finish_time: int
    slack: int
    reward_value: float
    reward_density: float
    feasible_now: bool


def _as_point(value: Any) -> Point:
    x, y = value
    return int(x), int(y)


def _grid_congested_set(state: dict[str, Any]) -> set[Point]:
    grid = state.get("grid", {})
    return {_as_point(point) for point in grid.get("congested_zones", [])}


def _hotspot_points(state: dict[str, Any]) -> tuple[Point, ...]:
    grid = state.get("grid", {})
    return tuple(_as_point(point) for point in grid.get("hotspots", []))


def _point_cost(point: Point, congested_zones: set[Point]) -> int:
    return 2 if point in congested_zones else 1


def _route_cost(start: Point, end: Point, congested_zones: set[Point]) -> int:
    if start == end:
        return 0

    x1, y1 = start
    x2, y2 = end

    def walk(horizontal_first: bool) -> int:
        cost = 0
        x, y = x1, y1

        def step_to(nx: int, ny: int) -> None:
            nonlocal x, y, cost
            if (nx, ny) == (x, y):
                return
            x, y = nx, ny
            cost += _point_cost((x, y), congested_zones)

        if horizontal_first:
            dx = 1 if x2 >= x else -1
            while x != x2:
                step_to(x + dx, y)
            dy = 1 if y2 >= y else -1
            while y != y2:
                step_to(x, y + dy)
        else:
            dy = 1 if y2 >= y else -1
            while y != y2:
                step_to(x, y + dy)
            dx = 1 if x2 >= x else -1
            while x != x2:
                step_to(x + dx, y)

        return cost

    return min(walk(True), walk(False))


def estimate_job_cost(
    agent_location: Point,
    pickup_location: Point,
    drop_location: Point,
    congested_zones: set[Point],
    service_time: int = 1,
) -> int:
    return (
        _route_cost(agent_location, pickup_location, congested_zones)
        + _route_cost(pickup_location, drop_location, congested_zones)
        + service_time
    )


def _distance_to_nearest_hotspot(point: Point, hotspots: tuple[Point, ...]) -> int:
    if not hotspots:
        return 0
    x, y = point
    return min(abs(x - hx) + abs(y - hy) for hx, hy in hotspots)


def _visible_orders(state: dict[str, Any]) -> list[dict[str, Any]]:
    return [order for order in state.get("orders", []) if order.get("status") == "unassigned"]


def _idle_agents(state: dict[str, Any]) -> list[dict[str, Any]]:
    return [agent for agent in state.get("agents", []) if agent.get("status") == "idle"]


def _baseline_candidate(
    agent: dict[str, Any],
    order: dict[str, Any],
    congested_zones: set[Point],
    current_time: int,
) -> tuple[int, int, float, str, str]:
    agent_location = _as_point(agent["location"])
    pickup = _as_point(order["pickup_location"])
    drop = _as_point(order["drop_location"])
    estimated_cost = estimate_job_cost(agent_location, pickup, drop, congested_zones)
    slack = int(order["deadline"]) - (current_time + estimated_cost)
    return (
        int(order["deadline"]),
        estimated_cost,
        -float(order["reward_value"]),
        str(order["order_id"]),
        str(agent["agent_id"]),
    )


def _score_candidate(
    agent: dict[str, Any],
    order: dict[str, Any],
    congested_zones: set[Point],
    hotspots: tuple[Point, ...],
    current_time: int,
) -> CandidateAssignment:
    agent_location = _as_point(agent["location"])
    pickup = _as_point(order["pickup_location"])
    drop = _as_point(order["drop_location"])
    reward_value = float(order["reward_value"])
    deadline = int(order["deadline"])
    estimated_cost = estimate_job_cost(agent_location, pickup, drop, congested_zones)
    finish_time = current_time + estimated_cost
    slack = deadline - finish_time
    feasible_now = slack >= 0
    reward_density = reward_value / max(estimated_cost, 1)

    lateness_penalty = abs(slack) * 4.0 if slack < 0 else 0.0
    urgency_bonus = max(0.0, min(slack, 6)) * 0.8
    feasible_bonus = 8.0 if feasible_now else -6.0
    hotspot_bonus = max(0, 6 - _distance_to_nearest_hotspot(drop, hotspots)) * 0.7
    congestion_drag = max(0, estimated_cost - (_route_cost(agent_location, pickup, set()) + _route_cost(pickup, drop, set()) + 1))

    score = (
        (2.6 * reward_value)
        + (11.0 * reward_density)
        + urgency_bonus
        + feasible_bonus
        + hotspot_bonus
        - (1.45 * estimated_cost)
        - (1.2 * congestion_drag)
        - lateness_penalty
    )

    return CandidateAssignment(
        agent_id=str(agent["agent_id"]),
        order_id=str(order["order_id"]),
        score=score,
        estimated_cost=estimated_cost,
        finish_time=finish_time,
        slack=slack,
        reward_value=reward_value,
        reward_density=reward_density,
        feasible_now=feasible_now,
    )


def build_action(assignments: list[tuple[str, str]]) -> ActionDict:
    return {
        "assignments": [
            {"agent_id": agent_id, "order_id": order_id}
            for agent_id, order_id in assignments
        ]
    }


def baseline_policy(state: dict[str, Any]) -> ActionDict:
    current_time = int(state.get("time", 0))
    congested_zones = _grid_congested_set(state)
    idle_agents = sorted(_idle_agents(state), key=lambda agent: str(agent["agent_id"]))
    remaining_orders = list(_visible_orders(state))
    assignments: list[tuple[str, str]] = []

    for agent in idle_agents:
        if not remaining_orders:
            break
        ranked_orders = sorted(
            remaining_orders,
            key=lambda order: _baseline_candidate(agent, order, congested_zones, current_time),
        )
        chosen = ranked_orders[0]
        assignments.append((str(agent["agent_id"]), str(chosen["order_id"])))
        remaining_orders = [
            order for order in remaining_orders if str(order["order_id"]) != str(chosen["order_id"])
        ]

    return build_action(assignments)


def target_policy(state: dict[str, Any]) -> ActionDict:
    current_time = int(state.get("time", 0))
    congested_zones = _grid_congested_set(state)
    hotspots = _hotspot_points(state)
    idle_agents = list(_idle_agents(state))
    available_orders = list(_visible_orders(state))

    candidates = [
        _score_candidate(agent, order, congested_zones, hotspots, current_time)
        for agent in idle_agents
        for order in available_orders
    ]
    candidates.sort(
        key=lambda item: (
            -item.score,
            not item.feasible_now,
            -item.reward_density,
            item.estimated_cost,
            item.agent_id,
            item.order_id,
        )
    )

    chosen_agents: set[str] = set()
    chosen_orders: set[str] = set()
    assignments: list[tuple[str, str]] = []

    for candidate in candidates:
        if candidate.agent_id in chosen_agents or candidate.order_id in chosen_orders:
            continue
        if candidate.score < 0 and assignments:
            continue
        chosen_agents.add(candidate.agent_id)
        chosen_orders.add(candidate.order_id)
        assignments.append((candidate.agent_id, candidate.order_id))

    return build_action(assignments)


__all__ = [
    "ActionDict",
    "CandidateAssignment",
    "baseline_policy",
    "build_action",
    "estimate_job_cost",
    "target_policy",
]
