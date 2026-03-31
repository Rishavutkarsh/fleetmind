from __future__ import annotations

from .models import AgentState, GridConfig, OrderState, Scenario


def build_low_demand_scenario() -> Scenario:
    return Scenario(
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
        briefing="Sparse city with generous deadlines. Most orders are feasible, so wasted idle time and unnecessary detours matter.",
        dispatch_objective="Prefer clean on-time execution and avoid leaving easy nearby work untouched.",
        known_future_signal="Future demand stays light with occasional orders around the upper-right hotspot.",
    )


def build_high_demand_scenario() -> Scenario:
    return Scenario(
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
        ),
        episode_horizon=60,
        briefing="Demand arrives faster than the fleet can comfortably absorb, including short hotspot bursts, premium clusters, and occasional low-value long-haul distractions.",
        dispatch_objective="Balance urgency, value density, and capacity; skipping the wrong job should hurt later throughput, but some requests are not worth tying up the fleet for.",
        known_future_signal="Expect bursty arrivals around the upper-right hotspot near t=17-22 and again late in the episode, with one low-yield cross-city distraction in the middle.",
    )


def build_hotspot_congestion_scenario() -> Scenario:
    return Scenario(
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
        ),
        episode_horizon=80,
        briefing="Large city with hotspot bursts, tight premium orders, fixed congestion pockets, and a few low-yield long-haul traps. Positioning and selective commitments matter.",
        dispatch_objective="Trade off local urgent jobs against future hotspot bursts while avoiding long congested detours and low-value commitments that block the fleet.",
        known_future_signal="Expect premium hotspot spikes near t=18-20 and t=47, plus an unattractive cross-city request around t=32 and other long-haul work later.",
    )


SCENARIO_BUILDERS = {
    "low_demand": build_low_demand_scenario,
    "high_demand": build_high_demand_scenario,
    "hotspot_congestion": build_hotspot_congestion_scenario,
}
