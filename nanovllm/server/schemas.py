from typing import Annotated, Literal
from pydantic import BaseModel, ConfigDict, Field


class StrictModel(BaseModel):
    model_config = ConfigDict(extra='forbid', allow_inf_nan=False)


class StreamOptions(StrictModel):
    include_usage: bool = False


class GenerationRequest(StrictModel):
    model: str
    max_tokens: int = Field(16, ge=1, strict=True)
    temperature: float = Field(1.0, ge=0, le=2)
    top_p: float = Field(1.0, gt=0, le=1)
    top_k: int = Field(0, ge=0, strict=True)
    stream: bool = False
    stream_options: StreamOptions | None = None
    n: Literal[1] = 1
    stop: None = None


class CompletionRequest(GenerationRequest):
    prompt: str | list[Annotated[int, Field(ge=0, strict=True)]] = Field(min_length=1)
    echo: Literal[False] = False


class Message(StrictModel):
    role: Literal['system', 'developer', 'user', 'assistant']
    content: str


class ChatCompletionRequest(GenerationRequest):
    messages: list[Message] = Field(min_length=1)
