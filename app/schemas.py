"""Request / response shapes of the PulseIQ8_AI API."""
from __future__ import annotations

from typing import Any, Literal, Optional

from pydantic import BaseModel, Field

TierName = Literal["auto", "fast", "coding", "general", "reasoning", "vision"]


class Image(BaseModel):
    data: str = Field(..., description="base64 image bytes (no data: prefix)")
    media_type: str = "image/png"


class Message(BaseModel):
    role: Literal["system", "user", "assistant"]
    content: str
    images: list[Image] = Field(default_factory=list)


class ChatRequest(BaseModel):
    messages: list[Message]
    system: Optional[str] = None
    tier: TierName = "auto"
    task: Optional[str] = Field(None, description="e.g. analytics.insights — used for routing and usage stats")
    json_output: bool = Field(False, alias="json", description="ask for one JSON object back")
    max_tokens: int = Field(1800, ge=1, le=16000)
    temperature: float = Field(0.2, ge=0, le=2)
    sensitive: bool = Field(False, description="true = local models only, never a cloud provider")
    escalate: Optional[bool] = None
    model: Optional[str] = Field(None, description="answer with this exact model (e.g. qwen3:8b or ollama:qwen3:8b) "
                                                   "instead of routing; must be installed / configured")

    model_config = {"populate_by_name": True, "protected_namespaces": ()}


class Attempt(BaseModel):
    tier: str
    model: str
    ok: bool
    ms: int
    error: Optional[str] = None


class ChatResponse(BaseModel):
    text: str
    json_data: Optional[Any] = Field(None, alias="json")
    tier: str
    model: str
    provider: str
    route_reason: str
    latency_ms: int
    attempts: list[Attempt]
    usage: dict[str, int] = Field(default_factory=dict)

    model_config = {"populate_by_name": True}


class AssistantRequest(BaseModel):
    model_config = {"protected_namespaces": ()}

    """Data assistant: a conversation about the sales data (the last user message is the question)."""
    messages: list[Message]
    context: dict[str, Any] = Field(default_factory=dict, description="what the user is looking at, e.g. {from, to, region, segment, rep, page}")
    tier: Literal["auto", "fast", "general", "reasoning"] = "auto"
    model: Optional[str] = Field(None, description="preferred local model, e.g. qwen3:8b or deepseek-coder-v2:16b (default: by mode)")
    local_only: Optional[bool] = Field(None, description="true = local models only (default from ASSISTANT_LOCAL_ONLY, true)")


class TaskRequest(BaseModel):
    data: Optional[Any] = None
    question: Optional[str] = None
    images: list[Image] = Field(default_factory=list)
    sensitive: Optional[bool] = None
    max_tokens: Optional[int] = None
