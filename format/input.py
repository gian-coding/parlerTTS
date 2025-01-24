from pydantic import BaseModel
from typing import Optional

class inputPayload(BaseModel):
    text: str
    language: str
    description: Optional[str]

