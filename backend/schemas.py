#Pydantic models

from pydantic import BaseModel

class QueryRequest(BaseModel):
    query: str

class QueryResponse(BaseModel):
    response: str
    sources: list[str] = []
