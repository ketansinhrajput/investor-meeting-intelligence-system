"""
Authentication route: login endpoint.
"""

from datetime import datetime

from fastapi import APIRouter, Depends, HTTPException, status
from sqlalchemy.orm import Session

from database import get_db
from models.user import User
from api.auth import verify_password, create_access_token
from api.schemas.requests import LoginRequest
from api.schemas.responses import LoginResponse

router = APIRouter()


@router.post("/auth/login", response_model=LoginResponse)
async def login(request: LoginRequest, db: Session = Depends(get_db)):
    """Authenticate with username and password, returns a JWT."""
    user = db.query(User).filter(User.username == request.username).first()

    if user is None or not verify_password(request.password, user.password_hash):
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid username or password",
        )

    if not user.is_active:
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="Account is disabled. Contact an administrator.",
        )

    # Update last_login timestamp
    user.last_login = datetime.utcnow()
    db.commit()

    token = create_access_token(user.username, user.role)

    return LoginResponse(
        access_token=token,
        token_type="bearer",
        username=user.username,
        role=user.role,
    )
