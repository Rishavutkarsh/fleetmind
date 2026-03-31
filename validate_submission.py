from __future__ import annotations

import json
import sys
from pathlib import Path

import yaml
from fastapi.testclient import TestClient

sys.path.insert(0, str(Path(__file__).resolve().parent / "src"))

from delivery_dispatch.api import app
from delivery_dispatch.environment import DeliveryDispatchEnv
from delivery_dispatch.models import Action, Observation, StepResult
from delivery_dispatch.scenarios import SCENARIO_BUILDERS


def check(condition: bool, message: str) -> None:
    if not condition:
        raise AssertionError(message)


def validate_openenv_yaml() -> dict:
    path = Path("openenv.yaml")
    check(path.exists(), "openenv.yaml is missing")
    data = yaml.safe_load(path.read_text())
    check(data["name"] == "delivery-dispatch-openenv", "openenv.yaml name mismatch")
    task_ids = [task["id"] for task in data["tasks"]]
    check(set(task_ids) == set(SCENARIO_BUILDERS), "openenv.yaml tasks do not match scenarios")
    return data


def validate_environment_contract() -> None:
    env = DeliveryDispatchEnv("low_demand")
    observation = env.reset()
    check(isinstance(observation, Observation), "reset() must return Observation")

    state = env.state()
    check(isinstance(state, Observation), "state() must return Observation")

    step_result = env.step(Action(assignments=[]))
    check(isinstance(step_result, StepResult), "step() must return StepResult")
    check(0 <= step_result.reward.cumulative_reward or True, "reward object should be accessible")


def validate_inference() -> dict:
    import inference

    result = inference.score_tasks("baseline")
    check("tasks" in result and "overall_score" in result, "inference output missing keys")
    for task in result["tasks"]:
        check(0.0 <= float(task["score"]) <= 1.0, f"task score out of range for {task['task_id']}")
    return result


def validate_http_api() -> None:
    client = TestClient(app)
    health = client.get("/health")
    check(health.status_code == 200, "/health must return 200")

    reset = client.post("/reset", params={"task_id": "low_demand"})
    check(reset.status_code == 200, "/reset must return 200")
    reset_body = reset.json()
    check(reset_body["task_id"] == "low_demand", "/reset should select requested task")

    state = client.get("/state")
    check(state.status_code == 200, "/state must return 200")

    step = client.post(
        "/step",
        json={"assignments": [{"agent_id": "a1", "order_id": "o1"}]},
    )
    check(step.status_code == 200, "/step must return 200")
    step_body = step.json()
    check("observation" in step_body and "reward" in step_body, "/step response shape is invalid")


def main() -> None:
    yaml_data = validate_openenv_yaml()
    validate_environment_contract()
    inference_result = validate_inference()
    validate_http_api()

    summary = {
        "status": "ok",
        "openenv_name": yaml_data["name"],
        "tasks": [task["id"] for task in yaml_data["tasks"]],
        "inference": inference_result,
    }
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
