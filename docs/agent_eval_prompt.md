# Black-Box Agent Evaluation Prompt

Use this prompt when evaluating an external agent against Fleetmind without giving it source-code access.

## Goal

Maximize cumulative reward in the live environment by interacting only through the HTTP API.

## Instructions

You are interacting with a live delivery-dispatch environment as a black-box external agent.

You may use any reasoning tools available to you, including calculations, helper code, or temporary scripts, but you must not inspect the environment source code or hidden files. Treat the HTTP API as the only interface to the environment.

Use only these endpoints:
- `GET /health`
- `POST /reset`
- `GET /state`
- `POST /step`

## Required Workflow

1. Call `GET /health` to confirm the service is live.
2. Start a fresh episode with `POST /reset`.
3. Play the episode entirely through repeated `POST /step` calls until `done = true`.
4. Use `GET /state` only if needed for recovery or inspection.
5. Choose assignments and rejections based only on API observations and returned feedback.

## Constraints

- Do not inspect local repository files or source code.
- Do not assume hidden future orders or hidden reward terms beyond what can be inferred from the API.
- Do not modify the environment implementation.
- You may write local scratch logic for your own planning, but the environment must be treated as a black box.

## Final Report

At the end of the episode, report:
- final cumulative reward
- the policy or strategy you converged on
- key assignment and rejection decisions
- what the API feedback taught you
- what felt confusing, too easy, too derived, or gameable
