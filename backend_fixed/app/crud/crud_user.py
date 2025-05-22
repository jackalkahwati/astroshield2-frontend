from sqlalchemy.orm import Session
from typing import Optional, List, Any

from app.models.user import User as UserModel # SQLAlchemy model
from app.schemas.user import UserCreate, UserUpdate # Pydantic schemas
from app.core.security import get_password_hash # For hashing password

class CRUDUser:
    def get_user(self, db: Session, user_id: int) -> Optional[UserModel]:
        return db.query(UserModel).filter(UserModel.id == user_id).first()

    def get_user_by_email(self, db: Session, email: str) -> Optional[UserModel]:
        return db.query(UserModel).filter(UserModel.email == email).first()

    def get_users(self, db: Session, skip: int = 0, limit: int = 100) -> List[UserModel]:
        return db.query(UserModel).offset(skip).limit(limit).all()

    def create_user(self, db: Session, user_in: UserCreate) -> UserModel:
        hashed_password = get_password_hash(user_in.password)
        db_user = UserModel(
            email=user_in.email,
            full_name=user_in.full_name,
            hashed_password=hashed_password,
            is_active=user_in.is_active if user_in.is_active is not None else True,
            is_superuser=user_in.is_superuser if user_in.is_superuser is not None else False,
            roles=user_in.roles if user_in.roles is not None else []
        )
        db.add(db_user)
        db.commit()
        db.refresh(db_user)
        return db_user

    def update_user(self, db: Session, db_user: UserModel, user_in: UserUpdate) -> UserModel:
        update_data = user_in.model_dump(exclude_unset=True) # Use model_dump for Pydantic v2
        if "password" in update_data and update_data["password"]:
            hashed_password = get_password_hash(update_data["password"])
            del update_data["password"] # Don't store plain password
            update_data["hashed_password"] = hashed_password
        
        for field, value in update_data.items():
            setattr(db_user, field, value)
            
        db.add(db_user)
        db.commit()
        db.refresh(db_user)
        return db_user

    def delete_user(self, db: Session, user_id: int) -> Optional[UserModel]:
        user = db.query(UserModel).filter(UserModel.id == user_id).first()
        if user:
            db.delete(user)
            db.commit()
        return user

user_crud = CRUDUser() # Singleton instance 