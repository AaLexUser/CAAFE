from pydantic import BaseModel, Field
from typing import List, Optional

class Observation(BaseModel):
    """Represents the result of a tool execution."""

    is_success: bool = Field(default=True)
    message: str = Field(default="")
    base64_images: Optional[List[str]] = Field(default=None)