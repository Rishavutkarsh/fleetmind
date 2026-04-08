from __future__ import annotations

import secrets

from .dynamics import Counts, count_moved, move_cost, parse_target_counts, pressure_label, round_service_reward
from .generator import generate_recipe
from .models import (
    HiddenRecipe,
    V3Action,
    V3Feedback,
    V3Observation,
    V3Reward,
    V3ScenarioInfo,
    V3StepResult,
    ZoneSnapshot,
)
from .task_adapter import to_internal_task_id, to_public_task_id


class V3DeliveryDispatchEnv:
    def __init__(self, default_task_id: str = "medium_dispatch") -> None:
        self.default_task_id = default_task_id
        self.recipe: HiddenRecipe | None = None
        self.task_id = default_task_id
        self.internal_task_id = to_internal_task_id(default_task_id)
        self.public_seed = 0
        self.internal_seed = 0
        self.pool_name = "test"
        self.round_index = 0
        self.courier_counts: Counts = ()
        self.cumulative_reward = 0.0
        self.last_step_reward = 0.0
        self.recent_events: list[str] = []
        self.done = False

    def reset(
        self,
        task_id: str | None = None,
        seed: int | None = None,
        pool_name: str = "test",
    ) -> V3Observation:
        self.pool_name = pool_name
        self.public_seed = seed if seed is not None else secrets.randbelow(1_000_000_000)
        from .seed_catalog import (
            choose_random_curated_seed,
            choose_random_task_id,
            resolve_curated_seed,
            resolve_task_id,
        )

        if task_id is None:
            self.internal_task_id = (
                resolve_task_id(self.public_seed) if seed is not None else choose_random_task_id()
            )
        else:
            self.internal_task_id = to_internal_task_id(task_id)
        self.task_id = to_public_task_id(self.internal_task_id)

        if seed is None:
            self.internal_seed = choose_random_curated_seed(self.internal_task_id, pool_name=self.pool_name)
        else:
            self.internal_seed = resolve_curated_seed(
                self.internal_task_id,
                self.public_seed,
                pool_name=self.pool_name,
            )
        self.recipe = generate_recipe(self.internal_task_id, self.internal_seed)
        self.round_index = 0
        self.courier_counts = self.recipe.initial_courier_counts
        self.cumulative_reward = 0.0
        self.last_step_reward = 0.0
        self.recent_events = ["environment reset"]
        self.done = False
        return self.state()

    def reset_internal(
        self,
        task_id: str,
        internal_seed: int,
        public_seed: int | None = None,
        pool_name: str = "test",
    ) -> V3Observation:
        self.internal_task_id = to_internal_task_id(task_id)
        self.task_id = to_public_task_id(self.internal_task_id)
        self.pool_name = pool_name
        self.public_seed = internal_seed if public_seed is None else public_seed
        self.internal_seed = internal_seed
        self.recipe = generate_recipe(self.internal_task_id, self.internal_seed)
        self.round_index = 0
        self.courier_counts = self.recipe.initial_courier_counts
        self.cumulative_reward = 0.0
        self.last_step_reward = 0.0
        self.recent_events = ["environment reset"]
        self.done = False
        return self.state()

    def state(self) -> V3Observation:
        recipe = self._require_recipe()
        if self.done:
            round_template = recipe.rounds[-1]
            remaining_rounds = 0
        else:
            round_template = recipe.rounds[self.round_index]
            remaining_rounds = recipe.profile.total_rounds - self.round_index
        return V3Observation(
            round_index=self.round_index,
            remaining_rounds=remaining_rounds,
            task_id=self.task_id,
            zones=[
                ZoneSnapshot(
                    zone_id=zone.zone_id,
                    label=zone.label,
                    courier_count=self.courier_counts[index],
                    visible_orders=round_template.visible_orders_by_zone[index],
                    reward_per_order=round_template.reward_per_order_by_zone[index],
                    congestion_multiplier=round_template.congestion_multiplier_by_zone[index],
                )
                for index, zone in enumerate(recipe.zone_specs)
            ],
            feedback=V3Feedback(
                last_step_reward=self.last_step_reward,
                cumulative_reward=self.cumulative_reward,
                recent_events=list(self.recent_events),
                current_pressure=pressure_label(round_template, recipe.zone_specs),
            ),
            scenario_info=V3ScenarioInfo(
                task_id=self.task_id,
                used_seed=self.public_seed,
                total_rounds=recipe.profile.total_rounds,
                total_couriers=recipe.profile.courier_count,
                max_repositions_per_round=recipe.profile.max_repositions_per_round,
            ),
        )

    def step(self, action: V3Action, grade_terminal: bool = True) -> V3StepResult:
        if self.done:
            return V3StepResult(
                observation=self.state(),
                reward=V3Reward(step_reward=0.0, cumulative_reward=self.cumulative_reward),
                done=True,
                info={"message": "episode already finished"},
            )

        recipe = self._require_recipe()
        round_template = recipe.rounds[self.round_index]
        step_reward, served, missed = round_service_reward(
            round_template=round_template,
            current_counts=self.courier_counts,
            missed_order_penalty=recipe.profile.missed_order_penalty,
        )

        events = [f"served {sum(served)} orders, missed {sum(missed)}"]
        invalid_penalty = 0.0
        next_counts = self.courier_counts
        if self.round_index < recipe.profile.total_rounds - 1:
            parsed = parse_target_counts(self.state(), action)
            if parsed is None:
                invalid_penalty -= 8.0
                events.append("invalid target allocation; kept current fleet distribution")
            elif count_moved(self.courier_counts, parsed) > recipe.profile.max_repositions_per_round:
                invalid_penalty -= 6.0
                events.append("target allocation exceeded reposition cap; kept current fleet distribution")
            else:
                next_counts = parsed
        else:
            events.append("final round; ignored rebalancing target")

        movement_cost = 0.0
        if self.round_index < recipe.profile.total_rounds - 1 and next_counts != self.courier_counts:
            movement_cost = move_cost(recipe, self.round_index, self.courier_counts, next_counts)
            step_reward -= movement_cost
            events.append(f"rebalanced fleet for {movement_cost:.1f} movement cost")

        step_reward += invalid_penalty
        self.cumulative_reward += step_reward
        self.last_step_reward = step_reward
        self.recent_events = events
        self.courier_counts = next_counts
        self.round_index += 1
        self.done = self.round_index >= recipe.profile.total_rounds

        info: dict[str, object] = {}
        if self.done and grade_terminal:
            from .grading import grade_episode

            task_result = grade_episode(
                task_id=self.internal_task_id,
                seed=self.internal_seed,
                raw_reward=self.cumulative_reward,
            )
            info["episode_summary"] = {
                "raw_reward": round(task_result.raw_reward, 3),
                "baseline_reward": round(task_result.baseline_reward, 3),
                "target_reward": round(task_result.target_reward, 3),
                "heuristic_reward": None if task_result.heuristic_reward is None else round(task_result.heuristic_reward, 3),
                "graded_score": round(task_result.score, 4),
            }

        return V3StepResult(
            observation=self.state(),
            reward=V3Reward(step_reward=step_reward, cumulative_reward=self.cumulative_reward),
            done=self.done,
            info=info,
        )

    def clone(self) -> "V3DeliveryDispatchEnv":
        clone = V3DeliveryDispatchEnv(default_task_id=self.default_task_id)
        clone.recipe = self.recipe
        clone.task_id = self.task_id
        clone.internal_task_id = self.internal_task_id
        clone.public_seed = self.public_seed
        clone.internal_seed = self.internal_seed
        clone.pool_name = self.pool_name
        clone.round_index = self.round_index
        clone.courier_counts = self.courier_counts
        clone.cumulative_reward = self.cumulative_reward
        clone.last_step_reward = self.last_step_reward
        clone.recent_events = list(self.recent_events)
        clone.done = self.done
        return clone

    def _require_recipe(self) -> HiddenRecipe:
        if self.recipe is None:
            self.reset(self.default_task_id, 101)
        assert self.recipe is not None
        return self.recipe
