"""Named tasks: which tier each one needs and, for /v1/tasks/{name}, the prompt PulseIQ8_AI writes around the data.

Callers that write their own prompt (PulseIQ8 analytics sends system + user) just pass `task` to /v1/chat so the
request is routed to the right tier and counted under that task.
"""
from __future__ import annotations

import json
from dataclasses import dataclass


@dataclass(frozen=True)
class Task:
    tier: str
    system: str
    instruction: str
    json_output: bool = True
    sensitive: bool = False
    max_tokens: int = 1800


ANALYST = ("You are PulseIQ8 Intelligence, a senior mobile-banking product analyst and iOS / Android engineer for "
           "Co-op Bank of Kenya. You read anonymised app analytics and explain in plain business English what is going "
           "wrong for customers, the likely cause and what the team should do first. Use only numbers present in the "
           "data; mark hypotheses as \"likely\". Prioritise failed transactions and onboarding drop-offs, then errors and "
           "slow APIs, then friction.")
ENGINEER = ("You are a senior mobile and backend engineer (Swift / SwiftUI, Kotlin / Jetpack Compose, Java / Spring "
            "Boot, REST APIs). Be precise, show code when it helps, prefer the smallest safe change, and mention the "
            "tests that prove it.")
ASSISTANT = ("You are Nexora Intelligence, a helpful, accurate assistant for a Kenyan bank's technology and business "
             "teams. Be clear and concise; say when you are unsure.")

TASKS: dict[str, Task] = {
    # ---- PulseIQ8 mobile analytics ----
    "analytics.insights": Task("general", ANALYST,
        "Analyse the app for the period and report the most important findings. Return JSON: {\"headline\": str, "
        "\"health\": \"good\"|\"watch\"|\"critical\", \"findings\": [{\"title\", \"severity\": \"critical\"|\"high\"|"
        "\"medium\"|\"low\", \"area\", \"where\", \"evidence\", \"likely_cause\", \"recommendation\", \"impact\"}], "
        "\"quick_wins\": [str], \"watch\": [str]}. 3 to 7 findings, worst first."),
    "analytics.ask": Task("general", ANALYST,
        "Answer the question using only the data. Return JSON: {\"answer\": str, \"evidence\": [str], \"follow_up\": [str]}."),
    "analytics.session": Task("general", ANALYST,
        "Explain what this ONE customer experienced in the session (steps oldest first, t = seconds from start). Return "
        "JSON: {\"summary\", \"goal\", \"outcome\": \"completed\"|\"failed\"|\"gave_up\"|\"browsing\", \"struggle_score\": "
        "0-100, \"moments\": [{\"t\", \"what\"}], \"likely_cause\", \"fix\"}."),
    "analytics.anomaly": Task("reasoning", ANALYST,
        "An automatic monitor flagged this anomaly. Investigate the most likely root cause, urgency and owner. Return "
        "JSON: {\"explanation\": str, \"urgency\": \"act now\"|\"today\"|\"watch\", \"owner\": \"backend\"|\"iOS\"|"
        "\"Android\"|\"product\"|\"operations\", \"checks\": [str]}."),
    "analytics.remedy": Task("fast", ANALYST,
        "Write ONE short, calm message (max 140 characters) for customers on the affected screen while the problem is "
        "fixed. Plain English, no technical words, no blame, no promised times, no numbers. Return JSON: {\"message\": str}.",
        max_tokens=200),
    # ---- coding ----
    "code.review": Task("coding", ENGINEER,
        "Review this code. Return JSON: {\"summary\": str, \"issues\": [{\"severity\", \"line\", \"problem\", \"fix\"}], "
        "\"tests\": [str]}."),
    "code.crash": Task("coding", ENGINEER,
        "Explain this crash / stack trace / error: the cause, the exact fix and how to prevent it. Return JSON: "
        "{\"cause\": str, \"fix\": str, \"code\": str, \"prevention\": str}."),
    "code.tests": Task("coding", ENGINEER, "Write unit tests for this code. Reply with the test code only.", json_output=False),
    # ---- vision ----
    "vision.analyze": Task("vision", ASSISTANT,
        "Look at the image(s) and answer the question. Read any text, error messages, UI, charts or diagrams in them. "
        "Return JSON: {\"answer\": str, \"seen\": [str], \"next_steps\": [str]}."),
    "vision.ui_review": Task("vision", ENGINEER,
        "Review this app screenshot for usability, accessibility, layout bugs and confusing text. Return JSON: "
        "{\"summary\": str, \"issues\": [{\"severity\", \"where\", \"problem\", \"fix\"}]}."),
    # ---- fast utilities ----
    "text.summarize": Task("fast", ASSISTANT, "Summarise in at most 5 bullet points.", json_output=False, max_tokens=400),
    "text.title": Task("fast", ASSISTANT, "Give a chat title of at most 6 words. Return JSON: {\"title\": str}.", max_tokens=40),
    "text.classify": Task("fast", ASSISTANT,
        "Classify the text into one of the given labels. Return JSON: {\"label\": str, \"confidence\": 0-1}.", max_tokens=60),
    "text.intent": Task("fast", ASSISTANT,
        "Detect the user's intent. Return JSON: {\"intent\": str, \"entities\": {}, \"confidence\": 0-1}.", max_tokens=120),
    # ---- reasoning ----
    "reasoning.compare": Task("reasoning", ASSISTANT,
        "Compare the options against the goal. Return JSON: {\"recommendation\": str, \"options\": [{\"name\", \"pros\", "
        "\"cons\", \"risk\"}], \"why\": str}."),
    "reasoning.plan": Task("reasoning", ASSISTANT,
        "Make a step-by-step plan. Return JSON: {\"goal\": str, \"steps\": [{\"step\", \"owner\", \"done_when\"}], \"risks\": [str]}."),
}

TASK_TIERS = {name: t.tier for name, t in TASKS.items()}


def build_user_message(task: Task, data, question: str | None) -> str:
    parts = [task.instruction]
    if question:
        parts.append(f"<question>\n{question}\n</question>")
    if data is not None:
        payload = data if isinstance(data, str) else json.dumps(data, ensure_ascii=False, default=str)
        parts.append(f"<data>\n{payload}\n</data>")
    return "\n\n".join(parts)
