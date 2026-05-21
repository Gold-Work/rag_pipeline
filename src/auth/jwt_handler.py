import os
from datetime import datetime, timedelta, timezone

from jose import JWTError, jwt

from src.auth.models import TokenData, UserRole


def _secret() -> str:
    secret = os.getenv("JWT_SECRET_KEY", "")
    if not secret:
        raise RuntimeError("JWT_SECRET_KEY manquant dans les variables d'environnement")
    return secret


def _algo() -> str:
    return os.getenv("JWT_ALGORITHM", "HS256")


def _expire_minutes() -> int:
    return int(os.getenv("JWT_EXPIRE_MINUTES", "60"))


def create_access_token(username: str, role: UserRole) -> str:
    payload = {
        "sub": username,
        "role": role.value,
        "exp": datetime.now(timezone.utc) + timedelta(minutes=_expire_minutes()),
    }
    return jwt.encode(payload, _secret(), algorithm=_algo())


def decode_token(token: str) -> TokenData:
    payload = jwt.decode(token, _secret(), algorithms=[_algo()])
    username: str | None = payload.get("sub")
    role_str: str | None = payload.get("role")
    if not username or not role_str:
        raise JWTError("Payload JWT incomplet : 'sub' ou 'role' manquant")
    try:
        role = UserRole(role_str)
    except ValueError:
        raise JWTError(f"Rôle inconnu dans le token : '{role_str}'")
    return TokenData(username=username, role=role)
