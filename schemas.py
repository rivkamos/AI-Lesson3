from pydantic import BaseModel

class AnswerSchema(BaseModel):
    answer: str
    sources_used: list[str] | None = None