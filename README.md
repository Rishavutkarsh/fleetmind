---
title: Fleetmind V3 OpenEnv
sdk: docker
app_port: 7860
---

# Fleetmind V3 OpenEnv

Fleetmind V3 is a delivery benchmark for the Meta x Scaler OpenEnv hackathon. Agents allocate couriers across zones under noisy visible demand while the evaluator grades against an exact privileged dynamic program over the hidden future.

The root submission path is now `v3`:
- root `app.py`
- root `openenv.yaml`
- root `inference.py`
- root `validate_submission.py`

## Public Tasks

- `easy_dispatch`
- `medium_dispatch`
- `hard_dispatch`

These public task ids map internally to curated `v3` seed pools. Agents only see the public task id and public seed.

## API

Endpoints:
- `GET /health`
- `POST /reset`
- `GET /state`
- `POST /step`

`POST /reset` behavior:
- with `task_id`, starts a fresh episode in that tier
- with `seed`, deterministically maps the public seed to a hidden curated test case
- with no `seed`, randomly selects a hidden curated test case
- with no `task_id`, randomly chooses one of the three public tasks

Observations include:
- `task_id`
- `round_index`
- `remaining_rounds`
- current zone snapshots
- feedback
- `scenario_info` with public seed and fleet limits

Actions use strict JSON:

```json
{
  "target_allocations": [
    {"zone_id": "north", "courier_count": 2},
    {"zone_id": "east", "courier_count": 1},
    {"zone_id": "south", "courier_count": 1},
    {"zone_id": "west", "courier_count": 2}
  ]
}
```

Rules:
- include every zone exactly once
- courier counts must sum to the total courier count
- negative counts are invalid
- moves above the per-round reposition cap are penalized and ignored

## Grading

Each completed episode returns terminal grading in `info.episode_summary`:
- `raw_reward`
- `baseline_reward`
- `target_reward`
- `heuristic_reward`
- `graded_score`

`graded_score` is normalized to `[0.0, 1.0]`.

## Inference Contract

The root `inference.py` is the required hackathon baseline script.

It is:
- LLM-first when env vars are available
- deterministic-fallback when model config is missing or provider calls fail
- structured-stdout compliant with `[START]`, `[STEP]`, and `[END]`

Environment variable handling:
- API key precedence: `HF_TOKEN`, then `OPENAI_API_KEY`
- base URL: `API_BASE_URL` if present
- model: `MODEL_NAME` if present

The baseline always uses the OpenAI client library for LLM calls.

## Local Setup

Install dependencies:

```bash
python -m pip install -r requirements.txt
```

Run the submission baseline:

```bash
python inference.py
```

Run the pre-submission validator:

```bash
python validate_submission.py
```

Run the API locally:

```bash
uvicorn app:app --host 0.0.0.0 --port 7860
```

Example resets:

```bash
curl -X POST "http://127.0.0.1:7860/reset?task_id=easy_dispatch"
curl -X POST "http://127.0.0.1:7860/reset?task_id=hard_dispatch&seed=123456"
curl -X POST "http://127.0.0.1:7860/reset"
```

## Hugging Face Space

This repo is prepared for a Docker Space:
- SDK: `Docker`
- app port: `7860`
- visibility: `Public`

Recommended environment variables:
- `HF_TOKEN`
- `OPENAI_API_KEY`
- `API_BASE_URL`
- `MODEL_NAME`

## Files

- `openenv.yaml`: submission metadata
- `inference.py`: required baseline script
- `validate_submission.py`: local preflight
- `src/delivery_dispatch_v3/`: benchmark core
- `HACKATHON_REQUIREMENTS.md`: validator-facing requirements reference
