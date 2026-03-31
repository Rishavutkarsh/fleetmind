from __future__ import annotations

from random import Random

from .models import AgentState, GridConfig, OrderState, Scenario, ZonePhase


def _clamp(value: int, low: int, high: int) -> int:
    return max(low, min(high, value))


def _shift_point(
    point: tuple[int, int],
    width: int,
    height: int,
    rng: Random,
    max_shift: int,
) -> tuple[int, int]:
    x, y = point
    return (
        _clamp(x + rng.randint(-max_shift, max_shift), 0, width - 1),
        _clamp(y + rng.randint(-max_shift, max_shift), 0, height - 1),
    )


def _vary_order(
    order: OrderState,
    grid: GridConfig,
    rng: Random,
    *,
    time_shift: int,
    spatial_shift: int,
    reward_shift: int,
    deadline_shift: int,
) -> OrderState:
    created_at = max(0, order.created_at + rng.randint(-time_shift, time_shift))
    pickup = _shift_point(order.pickup_location, grid.width, grid.height, rng, spatial_shift)
    drop = _shift_point(order.drop_location, grid.width, grid.height, rng, spatial_shift)
    reward_value = max(4.0, order.reward_value + rng.randint(-reward_shift, reward_shift))
    deadline = max(created_at + 4, order.deadline + rng.randint(-deadline_shift, deadline_shift))
    return order.model_copy(
        update={
            "created_at": created_at,
            "pickup_location": pickup,
            "drop_location": drop,
            "reward_value": float(reward_value),
            "deadline": deadline,
        },
        deep=True,
    )


def _vary_scenario(
    scenario: Scenario,
    seed: int | None,
    *,
    hotspot_shift: int,
    time_shift: int,
    spatial_shift: int,
    reward_shift: int,
    deadline_shift: int,
) -> Scenario:
    if seed is None:
        return scenario

    rng = Random(f"{scenario.name}:{seed}")
    varied_grid = scenario.grid.model_copy(
        update={
            "hotspots": tuple(
                _shift_point(point, scenario.grid.width, scenario.grid.height, rng, hotspot_shift)
                for point in scenario.grid.hotspots
            )
        },
        deep=True,
    )
    varied_orders = tuple(
        sorted(
            (
                _vary_order(
                    order,
                    varied_grid,
                    rng,
                    time_shift=time_shift,
                    spatial_shift=spatial_shift,
                    reward_shift=reward_shift,
                    deadline_shift=deadline_shift,
                )
                for order in scenario.orders
            ),
            key=lambda item: (item.created_at, item.order_id),
        )
    )
    varied_hotspot_phases = tuple(
        phase.model_copy(
            update={
                "points": tuple(
                    _shift_point(point, scenario.grid.width, scenario.grid.height, rng, hotspot_shift)
                    for point in phase.points
                )
            },
            deep=True,
        )
        for phase in scenario.hotspot_phases
    )
    varied_congestion_phases = tuple(
        phase.model_copy(
            update={
                "points": tuple(
                    _shift_point(point, scenario.grid.width, scenario.grid.height, rng, 1)
                    for point in phase.points
                )
            },
            deep=True,
        )
        for phase in scenario.congestion_phases
    )
    return scenario.model_copy(
        update={
            "grid": varied_grid,
            "orders": varied_orders,
            "hotspot_phases": varied_hotspot_phases,
            "congestion_phases": varied_congestion_phases,
        },
        deep=True,
    )


def build_low_demand_scenario(seed: int | None = None) -> Scenario:
    scenario = Scenario(
        name="low_demand",
        grid=GridConfig(width=8, height=8, hotspots=((5, 5), (6, 5), (5, 6))),
        agents=(
            AgentState(agent_id="a1", location=(1, 1)),
            AgentState(agent_id="a2", location=(4, 2)),
            AgentState(agent_id="a3", location=(6, 6)),
        ),
        orders=(
            OrderState(order_id="o1", created_at=0, pickup_location=(1, 2), drop_location=(3, 3), reward_value=10, deadline=10),
            OrderState(order_id="o2", created_at=2, pickup_location=(5, 2), drop_location=(6, 4), reward_value=12, deadline=13),
            OrderState(order_id="o3", created_at=5, pickup_location=(2, 6), drop_location=(1, 7), reward_value=8, deadline=17),
            OrderState(order_id="o4", created_at=8, pickup_location=(5, 5), drop_location=(7, 6), reward_value=11, deadline=19),
            OrderState(order_id="o5", created_at=12, pickup_location=(2, 1), drop_location=(4, 1), reward_value=9, deadline=23),
            OrderState(order_id="o6", created_at=16, pickup_location=(6, 5), drop_location=(7, 7), reward_value=13, deadline=28),
            OrderState(order_id="o7", created_at=21, pickup_location=(3, 4), drop_location=(1, 5), reward_value=8, deadline=31),
            OrderState(order_id="o8", created_at=26, pickup_location=(4, 6), drop_location=(6, 7), reward_value=14, deadline=36),
        ),
        episode_horizon=40,
        default_max_decision_steps=20,
        hotspot_phases=(
            ZonePhase(start_time=0, points=((5, 5), (6, 5), (5, 6))),
            ZonePhase(start_time=24, points=((5, 5), (6, 5), (6, 6))),
        ),
        briefing="Sparse city with generous deadlines. Most orders are feasible, so wasted idle time and unnecessary detours matter.",
        dispatch_objective="Prefer clean on-time execution and avoid leaving easy nearby work untouched.",
        known_future_signal="",
    )
    return _vary_scenario(
        scenario,
        seed,
        hotspot_shift=0,
        time_shift=1,
        spatial_shift=0,
        reward_shift=1,
        deadline_shift=1,
    )


def build_high_demand_scenario(seed: int | None = None) -> Scenario:
    scenario = Scenario(
        name="high_demand",
        grid=GridConfig(
            width=10,
            height=10,
            congested_zones=((4, 4), (4, 5), (5, 4), (5, 5), (6, 5)),
            hotspots=((7, 7), (7, 8), (8, 7), (8, 8)),
        ),
        agents=(
            AgentState(agent_id="a1", location=(1, 1)),
            AgentState(agent_id="a2", location=(2, 8)),
            AgentState(agent_id="a3", location=(8, 2)),
            AgentState(agent_id="a4", location=(6, 6)),
        ),
        orders=(
            OrderState(order_id="o1", created_at=0, pickup_location=(7, 7), drop_location=(9, 8), reward_value=16, deadline=12),
            OrderState(order_id="o2", created_at=1, pickup_location=(6, 7), drop_location=(7, 9), reward_value=12, deadline=11),
            OrderState(order_id="o3", created_at=2, pickup_location=(3, 3), drop_location=(1, 5), reward_value=9, deadline=10),
            OrderState(order_id="o4", created_at=4, pickup_location=(8, 7), drop_location=(9, 9), reward_value=15, deadline=15),
            OrderState(order_id="o5", created_at=6, pickup_location=(2, 8), drop_location=(4, 9), reward_value=10, deadline=16),
            OrderState(order_id="o6", created_at=8, pickup_location=(7, 8), drop_location=(9, 6), reward_value=18, deadline=19),
            OrderState(order_id="o7", created_at=11, pickup_location=(5, 2), drop_location=(2, 2), reward_value=11, deadline=20),
            OrderState(order_id="o8", created_at=13, pickup_location=(8, 8), drop_location=(6, 9), reward_value=14, deadline=23),
            OrderState(order_id="o9", created_at=17, pickup_location=(1, 7), drop_location=(3, 9), reward_value=9, deadline=26),
            OrderState(order_id="o10", created_at=17, pickup_location=(7, 7), drop_location=(8, 9), reward_value=17, deadline=24),
            OrderState(order_id="o11", created_at=18, pickup_location=(8, 8), drop_location=(9, 7), reward_value=15, deadline=24),
            OrderState(order_id="o12", created_at=22, pickup_location=(6, 6), drop_location=(8, 4), reward_value=13, deadline=31),
            OrderState(order_id="o13", created_at=22, pickup_location=(7, 9), drop_location=(9, 9), reward_value=16, deadline=29),
            OrderState(order_id="o14", created_at=28, pickup_location=(7, 7), drop_location=(9, 7), reward_value=12, deadline=37),
            OrderState(order_id="o15", created_at=34, pickup_location=(4, 1), drop_location=(5, 3), reward_value=8, deadline=42),
            OrderState(order_id="o16", created_at=36, pickup_location=(8, 7), drop_location=(9, 5), reward_value=14, deadline=43),
            OrderState(order_id="o17", created_at=19, pickup_location=(1, 1), drop_location=(9, 9), reward_value=5, deadline=25),
            OrderState(order_id="o18", created_at=15, pickup_location=(0, 8), drop_location=(9, 0), reward_value=6, deadline=22),
        ),
        episode_horizon=60,
        default_max_decision_steps=25,
        hotspot_phases=(
            ZonePhase(start_time=0, points=((7, 7), (7, 8), (8, 7), (8, 8))),
            ZonePhase(start_time=18, points=((6, 7), (7, 7), (7, 8), (8, 8))),
            ZonePhase(start_time=36, points=((5, 6), (6, 6), (6, 7), (7, 7))),
        ),
        congestion_phases=(
            ZonePhase(start_time=0, points=((4, 4), (4, 5), (5, 4), (5, 5), (6, 5))),
            ZonePhase(start_time=32, points=((5, 5), (5, 6), (6, 5), (6, 6), (7, 6))),
        ),
        briefing="Demand arrives faster than the fleet can comfortably absorb, with clustered premium bursts and low-yield long-haul distractions.",
        dispatch_objective="Maximize cumulative reward under sustained capacity pressure; serving everything greedily should no longer be obviously best.",
        known_future_signal="",
    )
    return _vary_scenario(
        scenario,
        seed,
        hotspot_shift=1,
        time_shift=2,
        spatial_shift=1,
        reward_shift=2,
        deadline_shift=2,
    )


def build_hotspot_congestion_scenario(seed: int | None = None) -> Scenario:
    scenario = Scenario(
        name="hotspot_congestion",
        grid=GridConfig(
            width=15,
            height=15,
            congested_zones=(
                (6, 6), (6, 7), (6, 8),
                (7, 6), (7, 7), (7, 8),
                (8, 6), (8, 7), (8, 8),
                (10, 10), (10, 11), (11, 10),
            ),
            hotspots=((11, 11), (11, 12), (12, 11), (12, 12), (13, 12)),
        ),
        agents=(
            AgentState(agent_id="a1", location=(2, 2)),
            AgentState(agent_id="a2", location=(3, 10)),
            AgentState(agent_id="a3", location=(10, 4)),
            AgentState(agent_id="a4", location=(13, 13)),
            AgentState(agent_id="a5", location=(8, 12)),
        ),
        orders=(
            OrderState(order_id="o1", created_at=0, pickup_location=(11, 11), drop_location=(14, 14), reward_value=20, deadline=16),
            OrderState(order_id="o2", created_at=1, pickup_location=(12, 11), drop_location=(10, 14), reward_value=18, deadline=15),
            OrderState(order_id="o3", created_at=3, pickup_location=(4, 5), drop_location=(2, 9), reward_value=10, deadline=15),
            OrderState(order_id="o4", created_at=5, pickup_location=(11, 12), drop_location=(13, 10), reward_value=17, deadline=18),
            OrderState(order_id="o5", created_at=7, pickup_location=(9, 11), drop_location=(12, 13), reward_value=16, deadline=20),
            OrderState(order_id="o6", created_at=10, pickup_location=(6, 4), drop_location=(3, 2), reward_value=11, deadline=21),
            OrderState(order_id="o7", created_at=14, pickup_location=(13, 12), drop_location=(14, 9), reward_value=19, deadline=25),
            OrderState(order_id="o8", created_at=18, pickup_location=(10, 11), drop_location=(7, 13), reward_value=14, deadline=28),
            OrderState(order_id="o9", created_at=18, pickup_location=(12, 12), drop_location=(14, 10), reward_value=22, deadline=27),
            OrderState(order_id="o10", created_at=19, pickup_location=(11, 11), drop_location=(13, 14), reward_value=18, deadline=26),
            OrderState(order_id="o11", created_at=24, pickup_location=(3, 12), drop_location=(1, 14), reward_value=9, deadline=34),
            OrderState(order_id="o12", created_at=31, pickup_location=(12, 12), drop_location=(14, 8), reward_value=20, deadline=41),
            OrderState(order_id="o13", created_at=31, pickup_location=(10, 12), drop_location=(13, 13), reward_value=17, deadline=39),
            OrderState(order_id="o14", created_at=39, pickup_location=(11, 10), drop_location=(8, 11), reward_value=13, deadline=48),
            OrderState(order_id="o15", created_at=47, pickup_location=(5, 11), drop_location=(8, 14), reward_value=12, deadline=58),
            OrderState(order_id="o16", created_at=47, pickup_location=(12, 11), drop_location=(14, 14), reward_value=21, deadline=56),
            OrderState(order_id="o17", created_at=56, pickup_location=(13, 11), drop_location=(14, 13), reward_value=21, deadline=66),
            OrderState(order_id="o18", created_at=63, pickup_location=(10, 3), drop_location=(6, 1), reward_value=10, deadline=74),
            OrderState(order_id="o19", created_at=32, pickup_location=(1, 2), drop_location=(14, 14), reward_value=6, deadline=40),
            OrderState(order_id="o20", created_at=26, pickup_location=(0, 0), drop_location=(14, 13), reward_value=7, deadline=35),
        ),
        episode_horizon=80,
        default_max_decision_steps=30,
        hotspot_phases=(
            ZonePhase(start_time=0, points=((11, 11), (11, 12), (12, 11), (12, 12), (13, 12))),
            ZonePhase(start_time=20, points=((9, 10), (10, 10), (10, 11), (11, 11), (11, 12))),
            ZonePhase(start_time=45, points=((6, 11), (7, 11), (7, 12), (8, 12), (8, 13))),
        ),
        congestion_phases=(
            ZonePhase(
                start_time=0,
                points=((6, 6), (6, 7), (6, 8), (7, 6), (7, 7), (7, 8), (8, 6), (8, 7), (8, 8)),
            ),
            ZonePhase(
                start_time=24,
                points=((8, 8), (8, 9), (9, 8), (9, 9), (10, 9), (10, 10), (10, 11), (11, 10)),
            ),
            ZonePhase(
                start_time=50,
                points=((5, 10), (5, 11), (6, 10), (6, 11), (7, 11), (8, 11), (8, 12)),
            ),
        ),
        briefing="Large city with moving hotspot pressure, congestion pockets, premium spikes, and multiple low-yield long-haul traps.",
        dispatch_objective="Force strategic tradeoffs: a strong policy should balance premium hotspot work against detours, congestion, and selective rejection.",
        known_future_signal="",
    )
    return _vary_scenario(
        scenario,
        seed,
        hotspot_shift=1,
        time_shift=3,
        spatial_shift=1,
        reward_shift=3,
        deadline_shift=3,
    )


SCENARIO_BUILDERS = {
    "low_demand": build_low_demand_scenario,
    "high_demand": build_high_demand_scenario,
    "hotspot_congestion": build_hotspot_congestion_scenario,
}
