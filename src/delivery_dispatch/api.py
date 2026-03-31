from __future__ import annotations

from fastapi import FastAPI

from .environment import DeliveryDispatchEnv
from .models import Action


app = FastAPI(title="Delivery Dispatch OpenEnv")
_env = DeliveryDispatchEnv()


@app.get("/health")
def health() -> dict[str, str]:
    return {"status": "ok"}


@app.post("/reset")
def reset(task_id: str | None = None, max_decision_steps: int | None = None) -> dict:
    observation = _env.reset(task_id=task_id, max_decision_steps=max_decision_steps)
    return observation.model_dump(mode="json")


@app.get("/state")
def state() -> dict:
    return _env.state().model_dump(mode="json")


@app.post("/step")
def step(action: Action) -> dict:
    return _env.step(action).model_dump(mode="json")
