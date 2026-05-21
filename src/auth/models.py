from enum import Enum
from pydantic import BaseModel


class UserRole(str, Enum):
    ADMIN = "admin"
    USER = "user"


class TokenData(BaseModel):
    username: str
    role: UserRole
