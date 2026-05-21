import os

from fastapi import APIRouter, Depends, HTTPException, Request, status
from fastapi.security import OAuth2PasswordRequestForm
from pydantic import BaseModel

from src.auth.jwt_handler import create_access_token
from src.auth.models import UserRole
from src.auth.password import verify_password
from src.utils.logger import get_logger
from src.utils.rate_limiter import limiter

logger = get_logger("auth")
router = APIRouter()


class TokenResponse(BaseModel):
    access_token: str
    token_type: str
    role: str


def _get_users() -> dict[str, tuple[str, UserRole]]:
    """Construit la table des utilisateurs depuis les variables d'environnement.

    Retourne {username: (hashed_password, role)}.
    """
    users: dict[str, tuple[str, UserRole]] = {}

    admin_username = os.getenv("ADMIN_USERNAME")
    admin_hash = os.getenv("ADMIN_PASSWORD_HASH")
    if admin_username and admin_hash:
        users[admin_username] = (admin_hash, UserRole.ADMIN)

    user_username = os.getenv("USER_USERNAME")
    user_hash = os.getenv("USER_PASSWORD_HASH")
    if user_username and user_hash:
        users[user_username] = (user_hash, UserRole.USER)

    return users


@router.post("/api/auth/token", response_model=TokenResponse)
@limiter.limit("10/minute")
async def login(
    request: Request,
    form_data: OAuth2PasswordRequestForm = Depends(),
) -> TokenResponse:
    users = _get_users()

    if not users:
        logger.error("❌ Aucun utilisateur configuré — vérifiez ADMIN_USERNAME/ADMIN_PASSWORD_HASH dans .env")
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Service d'authentification non configuré",
        )

    entry = users.get(form_data.username)
    if entry is None or not verify_password(form_data.password, entry[0]):
        logger.warning(f"⚠️ Échec de connexion pour : '{form_data.username}'")
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Identifiants invalides",
            headers={"WWW-Authenticate": "Bearer"},
        )

    _, role = entry
    token = create_access_token(username=form_data.username, role=role)
    logger.info(f"✅ Connexion réussie : '{form_data.username}' (role={role.value})")
    return TokenResponse(access_token=token, token_type="bearer", role=role.value)
