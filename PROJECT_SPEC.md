# LLM-Driven Last-Mile Delivery Dispatch Environment

## Problem Statement

Build an `OpenEnv` environment that simulates a last-mile delivery dispatch system in a simplified city grid. A fixed fleet of delivery agents must be assigned to dynamically arriving orders. Each order has a pickup location, drop location, reward value, and delivery deadline. Some regions of the city may also be demand hotspots, and some zones may have fixed congestion that increases travel cost.

At every decision step, an LLM acts as the central dispatcher. It observes the current environment state and decides which idle agents should be assigned to which active orders. The objective is to maximize total earned reward while minimizing lateness, missed orders, invalid assignments, and poor fleet utilization.

This environment is designed to represent a simplified version of real-world dispatch systems used in logistics and food delivery platforms. The challenge is not low-level route control, but high-level sequential decision-making under limited resources, time pressure, and spatial constraints.

Implementation note:
- the environment is intentionally designed for LLM-style decision making
- the submission runtime should still remain self-contained and reproducible if external model credentials are unavailable
- optional external model execution may be supported, but the environment should remain meaningful and runnable without relying on paid provider credits

## One-Line Summary

An LLM-driven delivery dispatch simulator where a model assigns limited agents to dynamic orders under time, distance, deadline, reward, and congestion constraints.

## Environment Design

The city is represented as a 2D grid.

Each episode contains:
- a fixed grid layout
- a fixed number of delivery agents
- a deterministic schedule of incoming orders
- optional hotspot zones where more orders originate
- optional fixed congestion zones that increase movement cost

The environment should remain deterministic and reproducible for grading.

## Core Entities

### Agents

Each agent has:
- `agent_id`
- `location: (x, y)`
- `status: idle | busy`
- `busy_until`
- `assigned_order_id | null`

Rules:
- an agent can handle at most one order at a time
- only idle agents can be assigned new orders
- agents become available again when their current job is completed

### Orders

Each order has:
- `order_id`
- `created_at`
- `pickup_location: (x, y)`
- `drop_location: (x, y)`
- `reward_value`
- `deadline`
- `status: unassigned | assigned | completed | expired`

Rules:
- an order can be assigned to only one agent
- an order expires if not completed before the scenario's expiration rule
- expired orders cannot be reassigned

### Zones

Each cell in the grid may be one of:
- `normal`: movement cost `1`
- `congested`: movement cost `2`

Optional metadata:
- `hotspot: true | false`

Hotspots affect order generation frequency, not movement directly.

## State Representation

The `state()` output should be compact, structured, and easy for an LLM to read.

Recommended state schema:

```json
{
  "time": 12,
  "grid": {
    "width": 15,
    "height": 15,
    "congested_zones": [[6, 6], [6, 7], [7, 6], [7, 7]],
    "hotspots": [[11, 11], [12, 11], [11, 12]]
  },
  "agents": [
    {
      "agent_id": "a1",
      "location": [2, 3],
      "status": "idle",
      "busy_until": 12,
      "assigned_order_id": null
    }
  ],
  "orders": [
    {
      "order_id": "o4",
      "created_at": 10,
      "pickup_location": [11, 12],
      "drop_location": [13, 14],
      "reward_value": 18,
      "deadline": 20,
      "status": "unassigned"
    }
  ],
  "scenario_info": {
    "name": "hotspot_congestion",
    "episode_horizon": 40
  }
}
```

## Action Space

At each step, the dispatcher returns assignment decisions in strict JSON.

Recommended format:

```json
{
  "assignments": [
    {"agent_id": "a1", "order_id": "o4"},
    {"agent_id": "a2", "order_id": "o2"}
  ]
}
```

Action rules:
- only idle agents may be assigned
- each agent may appear at most once
- each order may appear at most once
- omitted idle agents remain idle
- malformed, duplicate, or infeasible assignments are ignored and penalized lightly

## Step Semantics

Each `step()` does the following:
1. receives the LLM assignment action
2. validates all assignments
3. assigns valid `(agent, order)` pairs
4. computes each assigned job's total completion time
5. updates agents to `busy`
6. advances the simulator to the next event
7. resolves completed jobs
8. expires overdue orders
9. injects any new scheduled orders
10. returns updated state, reward, done flag, and info

Episodes are also bounded by a configurable `max_decision_steps` limit. When that limit is reached:
- no further decisions are accepted
- future unseen orders are ignored
- already assigned orders are deterministically rolled forward and scored
- visible unassigned orders are terminally expired and penalized
- the environment returns a final summary through `done = true`

## Movement and Travel Cost

Base movement uses grid travel.

For `v1`, use shortest-path travel cost over the grid where:
- entering a `normal` cell costs `1`
- entering a `congested` cell costs `2`

Delivery time for an assigned order:

```text
job_time = travel(agent -> pickup) + travel(pickup -> drop) + service_time
```

Use:
- `service_time = 1`

To keep implementation simple and valid:
- use deterministic shortest-path cost
- no random traffic
- no dynamic road closures
- no stochastic delays

## Time Advancement

Use event-based advancement.

After each dispatch action, advance time to the earliest of:
- next agent becoming available
- next order arrival
- episode horizon reached

This keeps the episode efficient and avoids unnecessary empty steps.

## Reward Function

Reward should be shaped and interpretable.

Recommended components:
- `+ reward_value` for on-time delivery
- `+ early_bonus` if delivered well before deadline
- `- lateness_penalty` for late completion
- `- missed_penalty` for expired orders
- `- invalid_action_penalty` for invalid assignments
- `- idle_penalty` when idle agents exist and feasible orders remain

Suggested concrete version:
- on-time completion: `+reward_value`
- early completion bonus: `+0.1 * reward_value` if completed with slack `>= 3`
- late completion: `+0.3 * reward_value - lateness`
- expired order: `-0.5 * reward_value`
- invalid assignment: `-1`
- avoidable idle penalty: `-0.5`

This creates meaningful feedback without making the score hard to interpret.

## Objective

Maximize cumulative episode reward.

A strong dispatcher should:
- choose assignments that finish more orders on time
- avoid wasting agents on low-value or infeasible jobs
- handle congestion-aware tradeoffs
- position the fleet effectively in hotspot-heavy scenarios

## Scenario Suite

Use 3 deterministic tasks.

### Task 1: Low Demand

Purpose:
- verify core assignment logic
- reward simple nearest-feasible reasoning

Setup:
- grid: `8x8`
- agents: `3`
- orders: `8-10`
- congestion: none or minimal
- deadlines: generous
- hotspot effect: low

Expected behavior:
- most orders should be serviceable
- mistakes come mostly from poor assignment choices

### Task 2: High Demand

Purpose:
- test prioritization under scarcity

Setup:
- grid: `10x10` or `12x12`
- agents: `3-4`
- orders: `18-25`
- congestion: a few fixed zones
- deadlines: moderate
- hotspot effect: medium

Expected behavior:
- not all orders can be served
- agent must prioritize based on reward, distance, and urgency

### Task 3: Hotspot + Congestion

Purpose:
- test strategic dispatch in the richest setting

Setup:
- grid: `15x15`
- agents: `4-5`
- orders: `20-28`
- hotspot zones: concentrated demand in selected regions
- congestion zones: fixed cells with movement cost `2`
- deadlines: mixed, with some tight high-value orders

Expected behavior:
- the LLM must trade off:
  - high-value but congested orders
  - urgent nearby orders
  - long-term fleet positioning around hotspots

## Scenario Generation Rules

To preserve reproducibility:
- use fixed seeds or fully predefined schedules
- use deterministic order arrival times
- use deterministic congestion maps
- use fixed hotspot coordinates per scenario

Hotspots should influence where pickups appear more often, especially in the hard task.

## Grading

Each task gets a normalized score in `[0, 1]`.

Recommended formula:

```text
score = clamp((agent_reward - baseline_reward) / (target_reward - baseline_reward), 0, 1)
```

Where:
- `baseline_reward` is produced by a naive deterministic policy
- `target_reward` is produced by a stronger heuristic dispatch policy

### Baseline Policy

Simple rule-based dispatcher:
- sort active orders by earliest deadline
- assign nearest idle agent greedily

### Target Policy

Stronger heuristic dispatcher:
- score orders using reward, travel cost, deadline slack, and congestion-adjusted feasibility
- greedily assign the best feasible matches

Example heuristic:

```text
priority = 1.5 * reward_value - 1.0 * travel_cost - 2.0 * urgency_penalty
```

This gives stable anchors for normalization and makes scores interpretable across tasks.

## Per-Task Grader Output

Recommended output:

```json
{
  "task_id": "hotspot_congestion",
  "raw_reward": 57.0,
  "baseline_reward": 24.0,
  "target_reward": 68.0,
  "score": 0.75
}
```

## Terminal Resolution

When the configured decision-step budget is exhausted:
- assigned orders that would still finish before their service cutoff are resolved and scored
- assigned orders that would miss cutoff are expired
- visible unassigned orders are expired immediately
- not-yet-visible future orders are ignored

This keeps episode length bounded while preserving deterministic final scoring.

## Overall Score

Recommended weighted average:
- low demand: `0.2`
- high demand: `0.3`
- hotspot + congestion: `0.5`

This makes the hardest and most realistic task matter most.

## Submission Requirements Alignment

This spec is designed to support:
- real-world environment framing
- clear `step()`, `reset()`, `state()` behavior
- 3 distinct tasks
- reproducible graders
- meaningful reward shaping
- lightweight execution under hackathon constraints

## Why This Version Is Good

This version is strong because it stays:
- realistic
- deterministic
- judge-friendly
- fast enough to validate
- rich enough to show non-trivial LLM reasoning

It also avoids the common trap of overbuilding simulation complexity before the environment core is solid.
