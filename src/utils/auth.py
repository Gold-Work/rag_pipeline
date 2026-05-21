"""Re-exports pour compatibilité ascendante.

Le module d'authentification a été déplacé dans src/auth/.
Ce fichier maintient les imports existants fonctionnels.
"""
from src.auth.dependencies import oauth2_scheme, require_authenticated_user, require_admin  # noqa: F401
from src.auth.jwt_handler import create_access_token, decode_token  # noqa: F401
from src.auth.models import TokenData, UserRole  # noqa: F401
from src.auth.password import hash_password, verify_password  # noqa: F401


def verify_token(token: str = None) -> str:  # type: ignore[assignment]
    """Compat ascendante — préférer require_authenticated_user."""
    from fastapi import Depends
    from src.auth.dependencies import oauth2_scheme
    from src.auth.jwt_handler import decode_token
    from jose import JWTError
    from fastapi import HTTPException, status

    try:
        data = decode_token(token)  # type: ignore[arg-type]
        return data.username
    except JWTError:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Token invalide ou expiré",
            headers={"WWW-Authenticate": "Bearer"},
        )
