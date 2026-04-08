from __future__ import annotations

from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field


WorldRegime = Literal[
    "visible_ramp",
    "decoy_then_shift",
    "premium_late_surge",
    "congested_pivot",
]


class DifficultyProfile(BaseModel):
    model_config = ConfigDict(frozen=True)

    task_id: str
    zone_count: int
    courier_count: int
    total_rounds: int
    max_repositions_per_round: int
    missed_order_penalty: float
    move_cost_weight: float
    runtime_budget_ms: float


class ZoneSpec(BaseModel):
    model_config = ConfigDict(frozen=True)

    zone_id: str
    label: str
    position: tuple[int, int]


class RoundTemplate(BaseModel):
    model_config = ConfigDict(frozen=True)

    round_index: int
    visible_orders_by_zone: tuple[int, ...]
    reward_per_order_by_zone: tuple[float, ...]
    congestion_multiplier_by_zone: tuple[float, ...]


class HiddenRecipe(BaseModel):
    model_config = ConfigDict(frozen=True)

    task_id: str
    seed: int
    profile: DifficultyProfile
    world_regime: WorldRegime
    hot_zone_index: int
    decoy_zone_index: int
    support_zone_index: int
    premium_zone_index: int
    zone_specs: tuple[ZoneSpec, ...]
    initial_courier_counts: tuple[int, ...]
    rounds: tuple[RoundTemplate, ...]


class ZoneSnapshot(BaseModel):
    model_config = ConfigDict(frozen=True)

    zone_id: str
    label: str
    courier_count: int
    visible_orders: int
    reward_per_order: float
    congestion_multiplier: float


class ZoneAllocation(BaseModel):
    model_config = ConfigDict(frozen=True)

    zone_id: str
    courier_count: int


class V3Action(BaseModel):
    target_allocations: list[ZoneAllocation] = Field(default_factory=list)


class V3Reward(BaseModel):
    step_reward: float
    cumulative_reward: float


class V3Feedback(BaseModel):
    last_step_reward: float = 0.0
    cumulative_reward: float = 0.0
    recent_events: list[str] = Field(default_factory=list)
    current_pressure: str = ""


class V3ScenarioInfo(BaseModel):
    task_id: str
    used_seed: int
    total_rounds: int
    total_couriers: int
    max_repositions_per_round: int


class V3Observation(BaseModel):
    round_index: int
    remaining_rounds: int
    task_id: str
    zones: list[ZoneSnapshot]
    feedback: V3Feedback
    scenario_info: V3ScenarioInfo


class V3StepResult(BaseModel):
    observation: V3Observation
    reward: V3Reward
    done: bool
    info: dict[str, Any]


class V3TaskResult(BaseModel):
    task_id: str
    raw_reward: float
    baseline_reward: float
    target_reward: float
    score: float
    heuristic_reward: float | None = None


class SeedMetadata(BaseModel):
    task_id: str
    seed: int
    world_regime: str
    hot_zone: str
    decoy_zone: str
    premium_zone: str
    baseline_reward: float
    heuristic_reward: float
    target_reward: float
    score_gap: float
    heuristic_gap: float
    solver_runtime_ms: float
    runtime_budget_ms: float
    admissible: bool
