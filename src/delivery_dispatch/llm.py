from __future__ import annotations

import json
import os
from typing import Any

from openai import OpenAI

from .models import Action, Observation


SYSTEM_PROMPT = (
    "You are a delivery dispatch policy. "
    "Return JSON only with the shape "
    "{\"assignments\": [{\"agent_id\": \"...\", \"order_id\": \"...\"}], \"rejections\": [\"order_id\"]}. "
    "Assign only idle agents to visible unassigned orders and reject only visible unassigned orders. "
    "Do not include explanations."
)


def llm_configured() -> bool:
    return all(os.getenv(name) for name in ("API_BASE_URL", "MODEL_NAME", "HF_TOKEN"))


def build_client() -> OpenAI:
    return OpenAI(
        base_url=os.environ["API_BASE_URL"],
        api_key=os.environ["HF_TOKEN"],
    )


def choose_action_with_llm(observation: Observation) -> Action:
    client = build_client()
    response = client.chat.completions.create(
        model=os.environ["MODEL_NAME"],
        messages=[
            {"role": "system", "content": SYSTEM_PROMPT},
            {
                "role": "user",
                "content": json.dumps(observation.model_dump(mode="json"), separators=(",", ":")),
            },
        ],
        temperature=0.1,
    )
    raw_text = response.choices[0].message.content or ""
    return parse_action(raw_text)


def parse_action(raw_text: str) -> Action:
    try:
        payload: dict[str, Any] = json.loads(raw_text)
    except json.JSONDecodeError:
        start = raw_text.find("{")
        end = raw_text.rfind("}")
        if start == -1 or end == -1 or end <= start:
            return Action()
        try:
            payload = json.loads(raw_text[start : end + 1])
        except json.JSONDecodeError:
            return Action()
    try:
        return Action.model_validate(payload)
    except Exception:
        return Action()
