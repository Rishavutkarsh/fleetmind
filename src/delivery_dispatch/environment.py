from __future__ import annotations

from copy import deepcopy
from random import SystemRandom
from typing import Any

from .models import (
    Action,
    AgentState,
    Feedback,
    Metrics,
    Observation,
    OrderState,
    Reward,
    Scenario,
    ScenarioInfo,
    StepResult,
    ZonePhase,
)
from .policies import estimate_job_cost
from .scenarios import SCENARIO_BUILDERS


class DeliveryDispatchEnv:
    """Deterministic event-driven delivery dispatch simulator."""

    invalid_assignment_penalty = -1.0
    idle_penalty = -0.35
    service_time = 1
    missed_order_penalty_multiplier = 0.75
    feasible_assignment_bonus = 0.5
    infeasible_assignment_penalty = -1.5
    rejection_penalty_multiplier = 0.4
    service_grace_window = 4
    early_bonus_per_tick = 0.45
    early_bonus_cap = 4
    late_linear_penalty = 1.4
    late_quadratic_penalty = 0.35
    high_value_threshold = 16.0

    def __init__(
        self,
        scenario_name: str = "low_demand",
        max_decision_steps: int | None = None,
        seed: int | None = None,
    ) -> None:
        if scenario_name not in SCENARIO_BUILDERS:
            raise ValueError(f"Unknown scenario: {scenario_name}")
        self._scenario_name = scenario_name
        self._configured_max_decision_steps = max_decision_steps
        self._configured_seed = seed
        self._scenario: Scenario | None = None
        self.current_time = 0
        self.decision_step = 0
        self.max_decision_steps = max_decision_steps or 100
        self.arrival_freeze_time: int | None = None
        self.agents: list[AgentState] = []
        self.orders: list[OrderState] = []
        self.cumulative_reward = 0.0
        self.last_step_reward = 0.0
        self.recent_events: list[str] = []
        self.last_reward_breakdown: dict[str, float] = {}
        self.stats = {
            "completed_orders": 0,
            "on_time_orders": 0,
            "late_orders": 0,
            "expired_orders": 0,
            "rejected_orders": 0,
            "invalid_actions": 0,
        }
        self.last_error_summary: dict[str, int] = {}
        self.used_seed: int | None = None

    def reset(
        self,
        task_id: str | None = None,
        max_decision_steps: int | None = None,
        seed: int | None = None,
    ) -> Observation:
        if task_id is not None:
            if task_id not in SCENARIO_BUILDERS:
                raise ValueError(f"Unknown scenario: {task_id}")
            self._scenario_name = task_id

        scenario_seed = seed if seed is not None else self._configured_seed
        if scenario_seed is None:
            scenario_seed = SystemRandom().randint(1, 10_000_000)
        self.used_seed = scenario_seed
        self._scenario = deepcopy(SCENARIO_BUILDERS[self._scenario_name](scenario_seed))
        self.current_time = 0
        self.decision_step = 0
        self.max_decision_steps = (
            max_decision_steps
            or self._configured_max_decision_steps
            or self._scenario.default_max_decision_steps
        )
        self.agents = list(self._scenario.agents)
        self.orders = list(self._scenario.orders)
        self.cumulative_reward = 0.0
        self.last_step_reward = 0.0
        self.arrival_freeze_time = None
        self.recent_events = ["environment reset"]
        self.last_reward_breakdown = {}
        self.stats = {
            "completed_orders": 0,
            "on_time_orders": 0,
            "late_orders": 0,
            "expired_orders": 0,
            "rejected_orders": 0,
            "invalid_actions": 0,
        }
        self.last_error_summary = {}
        return self.state()

    def state(self) -> Observation:
        scenario = self._require_scenario()
        visible_orders = self._visible_orders()
        agent_views = [self._agent_view(agent) for agent in self.agents]
        order_views = [self._order_view(order) for order in visible_orders]
        metrics = Metrics(
            completed_orders=self.stats["completed_orders"],
            on_time_orders=self.stats["on_time_orders"],
            late_orders=self.stats["late_orders"],
            expired_orders=self.stats["expired_orders"],
            rejected_orders=self.stats["rejected_orders"],
            invalid_actions=self.stats["invalid_actions"],
            active_orders=len([order for order in visible_orders if order.status in {"unassigned", "assigned"}]),
            pending_orders=len([order for order in visible_orders if order.status == "unassigned"]),
            idle_agents=len([agent for agent in self.agents if agent.status == "idle"]),
            busy_agents=len([agent for agent in self.agents if agent.status == "busy"]),
        )
        public_error_summary = {
            key: value
            for key, value in self.last_error_summary.items()
            if key in {"expired_orders", "late_deliveries", "rejected_orders", "invalid_actions"}
        }
        return Observation(
            time=self.current_time,
            decision_step=self.decision_step,
            max_decision_steps=self.max_decision_steps,
            task_id=scenario.name,
            episode_horizon=scenario.episode_horizon,
            grid=self._current_grid(),
            agents=agent_views,
            orders=order_views,
            feedback=Feedback(
                last_step_reward=self.last_step_reward,
                cumulative_reward=self.cumulative_reward,
                recent_events=[],
                reward_breakdown={},
                error_summary=public_error_summary,
                current_pressure="",
            ),
            metrics=metrics,
            scenario_info=ScenarioInfo(
                name=scenario.name,
                episode_horizon=scenario.episode_horizon,
                default_max_decision_steps=scenario.default_max_decision_steps,
                used_seed=self.used_seed,
                briefing=scenario.briefing,
                dispatch_objective=scenario.dispatch_objective,
                known_future_signal="",
            ),
        )

    def step(self, action: Action | dict[str, Any]) -> StepResult:
        scenario = self._require_scenario()
        if self.decision_step >= self.max_decision_steps:
            return StepResult(
                observation=self.state(),
                reward=Reward(step_reward=0.0, cumulative_reward=round(self.cumulative_reward, 3)),
                done=True,
                info={
                    "accepted_assignments": [],
                    "rejected_orders": [],
                    "invalid_assignments": [],
                    "time_advanced_to": self.current_time,
                    "stats": dict(self.stats),
                    "reward_breakdown": {},
                    "error_summary": {},
                    "current_pressure": "",
                },
            )
        parsed_action = action if isinstance(action, Action) else Action.model_validate(action)
        step_reward = 0.0
        reward_breakdown = {
            "rejection_penalty": 0.0,
            "valid_assignment_bonus": 0.0,
            "infeasible_assignment_penalty": 0.0,
            "invalid_assignment_penalty": 0.0,
            "idle_penalty": 0.0,
            "completion_reward": 0.0,
            "expiry_penalty": 0.0,
        }
        error_summary = {
            "rejected_orders": 0,
            "late_deliveries": 0,
            "expired_orders": 0,
            "high_value_orders_missed": 0,
            "urgent_orders_unassigned": 0,
        }
        accepted_assignments: list[dict[str, str]] = []
        invalid_assignments: list[dict[str, str]] = []
        rejected_orders: list[str] = []
        self.recent_events = []

        busy_agents = {agent.agent_id for agent in self.agents if agent.status == "busy"}
        claimed_orders: set[str] = set()

        for order_id in parsed_action.rejections:
            order = self._find_order(order_id)
            if order is None or order.status != "unassigned" or order.created_at > self.current_time:
                step_reward += self.invalid_assignment_penalty
                reward_breakdown["invalid_assignment_penalty"] += self.invalid_assignment_penalty
                self.stats["invalid_actions"] += 1
                invalid_assignments.append({"agent_id": "reject", "order_id": order_id})
                self.recent_events.append(f"ignored invalid rejection for {order_id}")
                continue

            rejection_penalty = -self._rejection_penalty(order)
            step_reward += rejection_penalty
            reward_breakdown["rejection_penalty"] += rejection_penalty
            order.status = "rejected"
            order.rejected_at = self.current_time
            self.stats["rejected_orders"] += 1
            error_summary["rejected_orders"] += 1
            rejected_orders.append(order.order_id)
            self.recent_events.append(f"rejected {order.order_id}")

        for assignment in parsed_action.assignments:
            agent = self._find_agent(assignment.agent_id)
            order = self._find_order(assignment.order_id)
            valid = True

            if agent is None or order is None:
                valid = False
            elif agent.agent_id in busy_agents or agent.status != "idle":
                valid = False
            elif assignment.order_id in claimed_orders:
                valid = False
            elif order.status != "unassigned" or order.created_at > self.current_time:
                valid = False

            if not valid:
                step_reward += self.invalid_assignment_penalty
                reward_breakdown["invalid_assignment_penalty"] += self.invalid_assignment_penalty
                self.stats["invalid_actions"] += 1
                invalid_assignments.append(assignment.model_dump())
                self.recent_events.append(
                    f"ignored invalid assignment {assignment.agent_id}->{assignment.order_id}"
                )
                continue

            job_time = self._job_time(agent, order)
            agent.status = "busy"
            agent.assigned_order_id = order.order_id
            agent.busy_until = self.current_time + job_time
            order.status = "assigned"
            order.assigned_agent_id = agent.agent_id
            order.scheduled_completion_time = agent.busy_until
            claimed_orders.add(order.order_id)
            busy_agents.add(agent.agent_id)
            accepted_assignments.append(assignment.model_dump())
            estimated_finish = self.current_time + job_time
            if estimated_finish <= self._service_cutoff(order):
                step_reward += self.feasible_assignment_bonus
                reward_breakdown["valid_assignment_bonus"] += self.feasible_assignment_bonus
            else:
                step_reward += self.infeasible_assignment_penalty
                reward_breakdown["infeasible_assignment_penalty"] += self.infeasible_assignment_penalty
            self.recent_events.append(
                f"assigned {order.order_id} to {agent.agent_id} until t={agent.busy_until}"
            )

        avoidable_idle_slots = self._avoidable_idle_slots()
        if avoidable_idle_slots > 0:
            idle_agents = len([agent for agent in self.agents if agent.status == "idle"])
            idle_penalty = self.idle_penalty * min(idle_agents, avoidable_idle_slots)
            step_reward += idle_penalty
            reward_breakdown["idle_penalty"] += idle_penalty
            self.recent_events.append("avoidable idle capacity remained")

        next_time = self._next_event_time()
        if next_time is None:
            next_time = scenario.episode_horizon
        self.current_time = min(next_time, scenario.episode_horizon)

        completion_reward, completed_orders, late_deliveries = self._resolve_completions()
        step_reward += completion_reward
        reward_breakdown["completion_reward"] += completion_reward
        expiry_penalty, expired_orders, high_value_missed = self._expire_orders()
        step_reward += expiry_penalty
        reward_breakdown["expiry_penalty"] += expiry_penalty
        error_summary["late_deliveries"] += late_deliveries
        error_summary["expired_orders"] += len(expired_orders)
        error_summary["high_value_orders_missed"] += high_value_missed
        error_summary["urgent_orders_unassigned"] = self._count_urgent_unassigned_orders()

        if not self.recent_events:
            self.recent_events.append("no state change")

        if completed_orders:
            self.recent_events.extend(completed_orders)
        if expired_orders:
            self.recent_events.extend(expired_orders)

        self.decision_step += 1
        terminal_info: dict[str, int] = {}
        if self.decision_step >= self.max_decision_steps:
            (
                terminal_reward,
                terminal_events,
                terminal_error_summary,
                terminal_info,
            ) = self._finalize_terminal_state()
            step_reward += terminal_reward
            for key, value in terminal_error_summary.items():
                error_summary[key] = error_summary.get(key, 0) + value
            if terminal_events:
                self.recent_events.extend(terminal_events)

        self.cumulative_reward += step_reward
        self.last_step_reward = step_reward
        self.last_reward_breakdown = {
            key: round(value, 3)
            for key, value in reward_breakdown.items()
            if abs(value) > 1e-9
        }
        self.last_error_summary = {key: value for key, value in error_summary.items() if value}

        done = self._is_done()
        public_error_summary = {
            key: value
            for key, value in self.last_error_summary.items()
            if key in {"expired_orders", "late_deliveries", "rejected_orders", "invalid_actions"}
        }
        info = {
            "accepted_assignments": accepted_assignments,
            "rejected_orders": rejected_orders,
            "invalid_assignments": invalid_assignments,
            "time_advanced_to": self.current_time,
            "stats": dict(self.stats),
            "reward_breakdown": {},
            "error_summary": public_error_summary,
            "current_pressure": "",
            "terminal_resolution": terminal_info,
        }
        return StepResult(
            observation=self.state(),
            reward=Reward(
                step_reward=round(step_reward, 3),
                cumulative_reward=round(self.cumulative_reward, 3),
            ),
            done=done,
            info=info,
        )

    def _require_scenario(self) -> Scenario:
        if self._scenario is None:
            raise RuntimeError("Environment must be reset before use.")
        return self._scenario

    def _visible_orders(self) -> list[OrderState]:
        visibility_time = self.arrival_freeze_time if self.arrival_freeze_time is not None else self.current_time
        return [
            order
            for order in self.orders
            if order.created_at <= visibility_time and order.status not in {"completed", "expired", "rejected"}
        ]

    def _find_agent(self, agent_id: str) -> AgentState | None:
        return next((agent for agent in self.agents if agent.agent_id == agent_id), None)

    def _find_order(self, order_id: str) -> OrderState | None:
        return next((order for order in self.orders if order.order_id == order_id), None)

    def _agent_view(self, agent: AgentState) -> AgentState:
        availability_in = max(agent.busy_until - self.current_time, 0) if agent.status == "busy" else 0
        return agent.model_copy(
            update={
                "availability_in": availability_in,
                "idle_now": agent.status == "idle",
            },
            deep=True,
        )

    def _order_view(self, order: OrderState) -> OrderState:
        return order.model_copy(update={"service_cutoff_time": None}, deep=True)

    def _nearest_idle_agent(self, order: OrderState) -> AgentState | None:
        idle_agents = [agent for agent in self.agents if agent.status == "idle"]
        if not idle_agents:
            return None
        ranked = sorted(
            idle_agents,
            key=lambda agent: (
                self._job_time(agent, order),
                agent.agent_id,
            ),
        )
        return ranked[0]

    def _job_time(self, agent: AgentState, order: OrderState) -> int:
        congested = set(self._current_grid().congested_zones)
        return estimate_job_cost(
            agent.location,
            order.pickup_location,
            order.drop_location,
            congested,
            self.service_time,
        )

    def _service_cutoff(self, order: OrderState) -> int:
        return order.deadline + self.service_grace_window

    def _next_event_time(self) -> int | None:
        scenario = self._require_scenario()
        completion_times = [
            agent.busy_until
            for agent in self.agents
            if agent.status == "busy" and agent.busy_until > self.current_time
        ]
        arrival_times = [
            order.created_at
            for order in self.orders
            if order.status == "unassigned" and order.created_at > self.current_time
        ]
        cutoff_times = [
            self._service_cutoff(order) + 1
            for order in self.orders
            if order.status == "unassigned" and order.created_at <= self.current_time
        ]
        candidates = completion_times + arrival_times + cutoff_times + [scenario.episode_horizon]
        candidates = [candidate for candidate in candidates if candidate > self.current_time]
        return min(candidates) if candidates else None

    def _resolve_completions(self) -> tuple[float, list[str], int]:
        reward = 0.0
        events: list[str] = []
        late_deliveries = 0
        for agent in self.agents:
            if agent.status != "busy" or agent.busy_until > self.current_time:
                continue
            if agent.assigned_order_id is None:
                agent.status = "idle"
                agent.busy_until = self.current_time
                continue

            order = self._find_order(agent.assigned_order_id)
            if order is None:
                agent.status = "idle"
                agent.assigned_order_id = None
                agent.busy_until = self.current_time
                continue

            order.completed_at = self.current_time
            order.status = "completed"
            agent.location = order.drop_location
            agent.status = "idle"
            agent.busy_until = self.current_time
            agent.assigned_order_id = None

            lateness = max(order.completed_at - order.deadline, 0)
            order_reward = self._completion_reward(order, order.completed_at)

            if lateness == 0:
                early_ticks = max(order.deadline - order.completed_at, 0)
                if early_ticks > 0:
                    events.append(f"order {order.order_id} completed early by {early_ticks}")
                else:
                    events.append(f"order {order.order_id} completed on time")
                self.stats["on_time_orders"] += 1
            else:
                self.stats["late_orders"] += 1
                late_deliveries += 1
                if order.completed_at > self._service_cutoff(order):
                    events.append(f"order {order.order_id} completed beyond service cutoff")
                else:
                    events.append(f"order {order.order_id} completed late by {lateness}")

            reward += order_reward
            self.stats["completed_orders"] += 1
        return reward, events, late_deliveries

    def _expire_orders(self) -> tuple[float, list[str], int]:
        penalty = 0.0
        events: list[str] = []
        high_value_missed = 0
        for order in self.orders:
            if order.status != "unassigned":
                continue
            if order.created_at > self.current_time:
                continue
            if self._service_cutoff(order) < self.current_time:
                order.status = "expired"
                self.stats["expired_orders"] += 1
                penalty -= self._missed_order_penalty(order, self.current_time)
                events.append(f"order {order.order_id} expired")
                if order.reward_value >= self.high_value_threshold:
                    high_value_missed += 1
        return penalty, events, high_value_missed

    def _completion_reward(self, order: OrderState, completed_at: int) -> float:
        early_ticks = max(order.deadline - completed_at, 0)
        if completed_at <= order.deadline:
            early_bonus = self.early_bonus_per_tick * min(early_ticks, self.early_bonus_cap)
            return order.reward_value + early_bonus

        lateness = completed_at - order.deadline
        penalty = (self.late_linear_penalty * lateness) + (
            self.late_quadratic_penalty * (lateness ** 2)
        )
        return max(0.0, order.reward_value - penalty)

    def _count_urgent_unassigned_orders(self) -> int:
        return len(
            [
                order
                for order in self._visible_orders()
                if order.status == "unassigned" and (order.deadline - self.current_time) <= 3
            ]
        )

    def _finalize_terminal_state(self) -> tuple[float, list[str], dict[str, int], dict[str, int]]:
        reward = 0.0
        events: list[str] = []
        freeze_time = self.current_time
        self.arrival_freeze_time = freeze_time
        error_summary = {
            "expired_orders": 0,
            "high_value_orders_missed": 0,
            "late_deliveries": 0,
        }
        terminal_info = {
            "resolved_assigned_orders": 0,
            "terminal_expired_unassigned": 0,
            "terminal_expired_assigned": 0,
        }

        assigned_orders = [
            order for order in self.orders if order.status == "assigned"
        ]
        terminal_time = max(
            [self.current_time]
            + [
                order.scheduled_completion_time or self.current_time
                for order in assigned_orders
                if (order.scheduled_completion_time or self.current_time) <= self._service_cutoff(order)
            ]
        )

        for order in assigned_orders:
            agent = self._find_agent(order.assigned_agent_id) if order.assigned_agent_id else None
            finish_time = order.scheduled_completion_time or self.current_time
            if finish_time <= self._service_cutoff(order):
                order.completed_at = finish_time
                order.status = "completed"
                if agent is not None:
                    agent.location = order.drop_location
                    agent.status = "idle"
                    agent.busy_until = finish_time
                    agent.assigned_order_id = None
                order_reward = self._completion_reward(order, finish_time)
                reward += order_reward
                self.stats["completed_orders"] += 1
                terminal_info["resolved_assigned_orders"] += 1

                if finish_time <= order.deadline:
                    self.stats["on_time_orders"] += 1
                    if finish_time < order.deadline:
                        events.append(
                            f"terminal rollout completed {order.order_id} early by {order.deadline - finish_time}"
                        )
                    else:
                        events.append(f"terminal rollout completed {order.order_id} on time")
                else:
                    self.stats["late_orders"] += 1
                    error_summary["late_deliveries"] += 1
                    events.append(
                        f"terminal rollout completed {order.order_id} late by {finish_time - order.deadline}"
                    )
            else:
                order.status = "expired"
                self.stats["expired_orders"] += 1
                terminal_info["terminal_expired_assigned"] += 1
                penalty = -(self.missed_order_penalty_multiplier * order.reward_value)
                reward += penalty
                error_summary["expired_orders"] += 1
                if order.reward_value >= self.high_value_threshold:
                    error_summary["high_value_orders_missed"] += 1
                if agent is not None:
                    agent.status = "idle"
                    agent.busy_until = self.current_time
                    agent.assigned_order_id = None
                events.append(f"terminal expiry for assigned order {order.order_id}")

        for order in self.orders:
            if order.status != "unassigned":
                continue
            if order.created_at > freeze_time:
                continue
            order.status = "expired"
            self.stats["expired_orders"] += 1
            terminal_info["terminal_expired_unassigned"] += 1
            penalty = -(self.missed_order_penalty_multiplier * order.reward_value)
            reward += penalty
            error_summary["expired_orders"] += 1
            if order.reward_value >= self.high_value_threshold:
                error_summary["high_value_orders_missed"] += 1
            events.append(f"terminal expiry for unassigned order {order.order_id}")

        for agent in self.agents:
            if agent.status == "busy":
                agent.status = "idle"
                agent.assigned_order_id = None
                agent.busy_until = terminal_time

        self.current_time = terminal_time
        return reward, events, error_summary, terminal_info

    def _avoidable_idle_slots(self) -> int:
        idle_agents = [agent for agent in self.agents if agent.status == "idle"]
        if not idle_agents:
            return 0

        worthwhile_orders = [
            order
            for order in self._visible_orders()
            if order.status == "unassigned" and self._is_worth_serving_now(order, idle_agents)
        ]
        return len(worthwhile_orders)

    def _is_done(self) -> bool:
        scenario = self._require_scenario()
        if self.decision_step >= self.max_decision_steps:
            return True
        if self.current_time >= scenario.episode_horizon:
            return True
        pending_orders = [
            order
            for order in self.orders
            if order.status in {"unassigned", "assigned"}
            and (order.created_at <= self.current_time or order.created_at <= scenario.episode_horizon)
        ]
        any_busy = any(agent.status == "busy" for agent in self.agents)
        future_orders = any(
            order.status == "unassigned" and order.created_at > self.current_time
            for order in self.orders
        )
        return not pending_orders and not any_busy and not future_orders

    def _pressure_summary(self) -> str:
        visible_orders = self._visible_orders()
        unassigned_orders = [order for order in visible_orders if order.status == "unassigned"]
        idle_agents = [agent for agent in self.agents if agent.status == "idle"]
        current_grid = self._current_grid()
        urgent_orders = [
            order for order in unassigned_orders
            if self._service_cutoff(order) - self.current_time <= 6
        ]
        hotspot_orders = [
            order for order in unassigned_orders
            if order.pickup_location in current_grid.hotspots
        ]

        if urgent_orders and len(idle_agents) < len(urgent_orders):
            return "high urgency pressure: more urgent orders than idle agents"
        if hotspot_orders and idle_agents:
            return "hotspot pressure: current demand is concentrated near hotspot zones"
        if unassigned_orders and not idle_agents:
            return "capacity pressure: all agents are currently occupied"
        if not unassigned_orders:
            return "low pressure: no unassigned visible orders right now"
        return "moderate pressure: feasible work is available without immediate overload"

    def _current_grid(self) -> Any:
        scenario = self._require_scenario()
        return scenario.grid.model_copy(
            update={
                "hotspots": self._phase_points(scenario.hotspot_phases, scenario.grid.hotspots),
                "congested_zones": self._phase_points(
                    scenario.congestion_phases, scenario.grid.congested_zones
                ),
            },
            deep=True,
        )

    def _phase_points(
        self,
        phases: tuple[ZonePhase, ...],
        fallback: tuple[tuple[int, int], ...],
    ) -> tuple[tuple[int, int], ...]:
        if not phases:
            return fallback
        chosen = fallback
        for phase in sorted(phases, key=lambda item: item.start_time):
            if self.current_time >= phase.start_time:
                chosen = phase.points
            else:
                break
        return chosen

    def _best_idle_finish_time(self, order: OrderState, idle_agents: list[AgentState] | None = None) -> int | None:
        idle_agents = idle_agents or [agent for agent in self.agents if agent.status == "idle"]
        if not idle_agents:
            return None
        best_cost = min(self._job_time(agent, order) for agent in idle_agents)
        return self.current_time + best_cost

    def _priority_multiplier(self, order: OrderState, reference_time: int) -> float:
        urgency = order.deadline - reference_time
        multiplier = 1.0
        if order.reward_value >= self.high_value_threshold:
            multiplier += 0.3
        elif order.reward_value >= 12:
            multiplier += 0.15

        if urgency <= 3:
            multiplier += 0.3
        elif urgency <= 6:
            multiplier += 0.15
        return multiplier

    def _missed_order_penalty(self, order: OrderState, reference_time: int) -> float:
        return self.missed_order_penalty_multiplier * order.reward_value * self._priority_multiplier(
            order, reference_time
        )

    def _rejection_penalty(self, order: OrderState) -> float:
        expiry_penalty = self._missed_order_penalty(order, self.current_time)
        best_finish = self._best_idle_finish_time(order)
        if best_finish is None:
            ratio = 0.55
        elif best_finish > self._service_cutoff(order):
            ratio = 0.5
        elif best_finish > order.deadline:
            ratio = 0.65
        else:
            ratio = 0.8
        return max(1.0, expiry_penalty * ratio, self.rejection_penalty_multiplier * order.reward_value)

    def _is_worth_serving_now(self, order: OrderState, idle_agents: list[AgentState]) -> bool:
        best_finish = self._best_idle_finish_time(order, idle_agents)
        if best_finish is None or best_finish > self._service_cutoff(order):
            return False
        delivery_value = self._completion_reward(order, best_finish)
        return delivery_value >= max(3.0, self._rejection_penalty(order) * 1.2)
