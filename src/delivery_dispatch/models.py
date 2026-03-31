from __future__ import annotations

from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field


Point = tuple[int, int]
AgentStatus = Literal["idle", "busy"]
OrderStatus = Literal["unassigned", "assigned", "completed", "expired", "rejected"]


class GridConfig(BaseModel):
    model_config = ConfigDict(frozen=True)

    width: int
    height: int
    congested_zones: tuple[Point, ...] = ()
    hotspots: tuple[Point, ...] = ()


class AgentState(BaseModel):
    model_config = ConfigDict(validate_assignment=True)

    agent_id: str
    location: Point
    status: AgentStatus = "idle"
    busy_until: int = 0
    assigned_order_id: str | None = None
    availability_in: int = 0
    idle_now: bool = True


class OrderState(BaseModel):
    model_config = ConfigDict(validate_assignment=True)

    order_id: str
    created_at: int
    pickup_location: Point
    drop_location: Point
    reward_value: float
    deadline: int
    status: OrderStatus = "unassigned"
    assigned_agent_id: str | None = None
    scheduled_completion_time: int | None = None
    completed_at: int | None = None
    rejected_at: int | None = None
    service_cutoff_time: int | None = None
    nearest_agent_id: str | None = None
    estimated_service_time: int | None = None
    estimated_finish_time: int | None = None
    slack_time: int | None = None
    feasible_now: bool | None = None


class Scenario(BaseModel):
    model_config = ConfigDict(frozen=True)

    name: str
    grid: GridConfig
    agents: tuple[AgentState, ...]
    orders: tuple[OrderState, ...]
    episode_horizon: int
    default_max_decision_steps: int = 100
    briefing: str = ""
    dispatch_objective: str = ""
    known_future_signal: str = ""


class Assignment(BaseModel):
    model_config = ConfigDict(frozen=True)

    agent_id: str
    order_id: str


class Action(BaseModel):
    assignments: list[Assignment] = Field(default_factory=list)
    rejections: list[str] = Field(default_factory=list)


class Reward(BaseModel):
    step_reward: float
    cumulative_reward: float


class Feedback(BaseModel):
    last_step_reward: float = 0.0
    cumulative_reward: float = 0.0
    recent_events: list[str] = Field(default_factory=list)
    reward_breakdown: dict[str, float] = Field(default_factory=dict)
    error_summary: dict[str, int] = Field(default_factory=dict)
    current_pressure: str = ""


class Metrics(BaseModel):
    completed_orders: int = 0
    on_time_orders: int = 0
    late_orders: int = 0
    expired_orders: int = 0
    rejected_orders: int = 0
    invalid_actions: int = 0
    active_orders: int = 0
    pending_orders: int = 0
    idle_agents: int = 0
    busy_agents: int = 0


class ScenarioInfo(BaseModel):
    name: str
    episode_horizon: int
    default_max_decision_steps: int = 100
    briefing: str = ""
    dispatch_objective: str = ""
    known_future_signal: str = ""


class Observation(BaseModel):
    time: int
    decision_step: int
    max_decision_steps: int
    task_id: str
    episode_horizon: int
    grid: GridConfig
    agents: list[AgentState]
    orders: list[OrderState]
    feedback: Feedback
    metrics: Metrics
    scenario_info: ScenarioInfo


class StepResult(BaseModel):
    observation: Observation
    reward: Reward
    done: bool
    info: dict[str, Any]


class TaskResult(BaseModel):
    task_id: str
    raw_reward: float
    baseline_reward: float
    target_reward: float
    score: float
    completed_orders: int = 0
    on_time_orders: int = 0
    late_orders: int = 0
    expired_orders: int = 0
    rejected_orders: int = 0
    invalid_actions: int = 0
