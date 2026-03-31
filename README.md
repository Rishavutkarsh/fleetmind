---
title: Delivery Dispatch OpenEnv
sdk: docker
app_port: 7860
---

# Delivery Dispatch OpenEnv

Deterministic last-mile delivery dispatch environment for the Meta x Scaler OpenEnv hackathon.

This environment is designed for LLM-driven dispatch decision making: an agent assigns limited delivery agents to dynamic orders under deadlines, congestion, and reward tradeoffs.

The submission is also self-contained and reproducible:
- it runs without private API keys
- it does not hard-fail when model env vars are missing
- it supports optional external model usage through environment variables

## Submission Notes

- keep the GitHub repository public for evaluation
- keep the Hugging Face Space public for evaluation
- do not commit private API keys
- local development may use a `.env` file, but production/evaluation config should come from environment variables

## Tasks

- `low_demand`: sparse arrivals, generous deadlines
- `high_demand`: more demand than agents can comfortably serve
- `hotspot_congestion`: larger map with concentrated demand and fixed congestion zones

## Observation Space

Each observation includes:
- simulation time and episode horizon
- current decision step and configured max decision steps
- grid dimensions, congestion zones, and hotspots
- agent states
- active visible orders
- minimal reward/error feedback
- compact task metrics

Design note:
- observations are intentionally closer to raw environment state than to a planner helper API
- derived hints such as nearest-agent suggestions or feasibility flags are not populated for the agent
- future demand schedules are not exposed through the API

Typed model:
- `delivery_dispatch.models.Observation`

## Action Space

The policy returns strict JSON:

```json
{
  "assignments": [
    {"agent_id": "a1", "order_id": "o4"}
  ],
  "rejections": ["o8"]
}
```

Rules:
- only idle agents can be assigned
- each agent can take at most one order
- each order can be assigned at most once
- rejections explicitly remove visible unassigned orders from play
- invalid assignments are ignored and penalized

Typed model:
- `delivery_dispatch.models.Action`

## Reward

Reward is shaped over the episode:
- positive reward for on-time completion
- small marginal bonus for earlier completion
- smoothly decayed reward for late completion, capped at zero for served orders
- penalties for explicit order rejection
- penalties for expired orders that pass the service cutoff
- penalties for invalid actions
- penalties for avoidable idle capacity

Feedback is intentionally sparse:
- `last_step_reward`
- `cumulative_reward`
- compact error counts such as late deliveries, expired orders, and rejected orders

Typed model:
- `delivery_dispatch.models.Reward`

## Local Setup

Install dependencies:

```bash
python -m pip install -r requirements.txt
```

Run deterministic scoring locally:

```bash
python inference.py
```

Run preflight validation:

```bash
python validate_submission.py
```

Run the thin API locally:

```bash
uvicorn app:app --host 0.0.0.0 --port 8000
```

You can start a fresh episode with a custom decision budget through:

```bash
curl -X POST "http://127.0.0.1:8000/reset?task_id=high_demand&max_decision_steps=40"
```

You can also request a reproducible seeded variant of a task:

```bash
curl -X POST "http://127.0.0.1:8000/reset?task_id=high_demand&max_decision_steps=40&seed=7"
```

## Inference Modes

### Default self-contained mode

By default, `inference.py` runs a deterministic local dispatch policy so the submission stays reproducible and does not depend on external credentials.

### Optional external model mode

If these environment variables are set, `inference.py` can use the OpenAI client path instead:
- `API_BASE_URL`
- `MODEL_NAME`
- `HF_TOKEN`

Without them, the submission still runs using the built-in policy.

For local development, you can copy `.env.example` and set variables in your shell or local `.env` workflow. Do not commit secrets.

Practical note:
- the external Hugging Face/OpenAI-compatible path is implemented and can be wired with environment variables
- live provider usage may still depend on available Hugging Face inference credits or billing
- because of that, the self-contained mode remains the default submission-safe path

## Docker

Build:

```bash
docker build -t delivery-dispatch-openenv .
```

Run:

```bash
docker run -p 7860:7860 delivery-dispatch-openenv
```

The container serves the API with:
- `GET /health`
- `POST /reset`
- `GET /state`
- `POST /step`

Episode termination:
- episodes end when the environment reaches `done = true`
- a configurable `max_decision_steps` cap can be passed to `/reset`
- when the decision cap is reached, future unseen orders are ignored, assigned work is terminally resolved, and visible unassigned work is expired for final scoring

## Hugging Face Spaces

This repo is prepared for a Docker-based Space:
- root `Dockerfile`
- root `app.py`
- README front matter with `sdk: docker`
- service port `7860`

Recommended Space configuration:
- SDK: `Docker`
- app port: `7860`
- visibility: `Public`
- environment variables:
  - `API_BASE_URL`
  - `MODEL_NAME`
  - `HF_TOKEN`

## Files

- `PROJECT_SPEC.md`: environment design spec
- `HACKATHON_REQUIREMENTS.md`: distilled submission requirements
- `.env.example`: local env var template without secrets
- `openenv.yaml`: OpenEnv submission metadata
- `inference.py`: root scoring entrypoint
- `validate_submission.py`: local preflight validator
- `src/delivery_dispatch/`: simulator, models, policies, grading, and API
