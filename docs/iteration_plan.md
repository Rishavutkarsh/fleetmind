# Fleetmind Iteration Plan

This document captures the next disciplined improvement loop for Fleetmind.

## Iteration Principles

1. Change one thing at a time.
2. Evaluate on fixed train seeds and held-out eval seeds.
3. Judge progress by both score and behavior.
4. Keep the environment black-box from the agent's perspective.

## Seed Discipline

Use small repeated seed sets instead of one-off runs.

Suggested split:
- train seeds: `1, 2, 3`
- eval seeds: `101, 102, 103`

Use the same seeds before and after each change.

## Curriculum

Run experiments in this order:
- `low_demand` for API and policy sanity
- `high_demand` for strategy formation
- `hotspot_congestion` for dynamic robustness

The goal is not to learn only on the hardest task.

## Iteration 1: Better End-of-Run Learning Signal

Hypothesis:
- Agents need a clearer terminal summary to improve across repeated episodes.

Success criteria:
- agents can explain what caused score loss
- policy revisions become more targeted

Candidate additions:
- average lateness
- reward lost to expiry
- reward lost to rejection
- cumulative idle penalty incurred

## Iteration 2: Reduce Waiting / No-Op Exploits

Hypothesis:
- repeated empty steps with worthwhile visible work are still too attractive.

Success criteria:
- agents stop gaining from passive waiting loops
- short strategic holding remains viable

Candidate changes:
- track repeated no-op steps
- increase penalty only when worthwhile serviceable work exists
- reset the counter after meaningful assignments or rejections

## Iteration 3: Make Medium/Hard More Distinct Strategically

Hypothesis:
- `high_demand` and `hotspot_congestion` should differ not only in pressure, but in the kind of planning they demand.

Success criteria:
- medium rewards stable prioritization
- hard rewards adaptation to evolving demand and congestion

Candidate changes:
- keep `high_demand` steady but capacity-constrained
- make `hotspot_congestion` reward anticipation of shifting hotspot phases more strongly

## Evaluation Protocol

For each iteration:

1. Run baseline and target on train/eval seeds.
2. Run one black-box agent with the prompt in `docs/agent_eval_prompt.md`.
3. Compare:
- cumulative reward
- on-time rate
- expiries
- rejections
- qualitative strategy

Keep the change only if both behavior and metrics improve.
