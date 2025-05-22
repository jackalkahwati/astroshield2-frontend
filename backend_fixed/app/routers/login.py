from fastapi import APIRouter, Depends, HTTPException, status
from fastapi.security import OAuth2PasswordRequestForm
from sqlalchemy.orm import Session
from datetime import timedelta, datetime

from app.core.security import create_access_token, authenticate_user, ACCESS_TOKEN_EXPIRE_MINUTES
from app.db.session import get_db
from app.schemas.token import Token # We'll create this schema

router = APIRouter()

@router.post("/token", response_model=Token)
async def login_for_access_token(
    db: Session = Depends(get_db), 
    form_data: OAuth2PasswordRequestForm = Depends()
):
    user = await authenticate_user(email=form_data.username, password=form_data.password, db=db) # form_data.username is the email
    if not user:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Incorrect email or password",
            headers={"WWW-Authenticate": "Bearer"},
        )
    if not user.is_active:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST, 
            detail="Inactive user"
        )
    
    access_token_expires = timedelta(minutes=ACCESS_TOKEN_EXPIRE_MINUTES)
    
    # Prepare data for token creation. Ensure 'sub' is the email.
    # Include other claims you might want directly in the token if not too large,
    # otherwise rely on get_current_user to fetch fresh data from DB.
    token_data = {
        "sub": user.email,
        "is_superuser": user.is_superuser, # Example: include superuser status
        "roles": user.roles # Example: include roles
    }

    access_token = create_access_token(
        data=token_data,
        expires_delta=access_token_expires
    )
    return {"access_token": access_token, "token_type": "bearer", "expires_at": datetime.utcnow() + access_token_expires} 