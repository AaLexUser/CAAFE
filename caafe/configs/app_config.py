from pathlib import Path
from typing import Any, Dict, List, Optional
from uuid import uuid4

from pydantic import BaseModel, Field

from caafe.constants import PACKAGE_PATH


class CachingConfig(BaseModel):
    enabled: bool = True
    dir_path: str = Field(default=str(Path(PACKAGE_PATH) / "cache"))


class LLMConfig(BaseModel):
    model_name: str = "openai/gpt-4o"
    base_url: Optional[str] = None
    api_key: Optional[str] = None
    caching: CachingConfig = Field(default_factory=CachingConfig)
    max_completion_tokens: Optional[int] = None
    temperature: Optional[float] = None
    stop_sequences: Optional[List[str]] = None
    extra_headers: Dict[str, Any] = Field(default_factory=dict)
    completion_params: Dict[str, Any] = Field(default_factory=dict)


class LangfuseConfig(BaseModel):
    host: str = "https://cloud.langfuse.com"
    public_key: Optional[str] = None
    secret_key: Optional[str] = None


class AppConfig(BaseModel):
    llm: LLMConfig = Field(default_factory=LLMConfig)
    langfuse: LangfuseConfig = Field(default_factory=LangfuseConfig)
    session_id: str = Field(default_factory=lambda: uuid4().hex)
