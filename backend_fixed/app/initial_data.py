import asyncio
from app.db.session import SessionLocal, engine
from app.db.base_class import Base
from app.core.config import settings
from app.schemas.user import UserCreate
from app.crud.crud_user import user_crud
from app.core.roles import Roles

async def create_admin_user():
    db = SessionLocal()
    try:
        Base.metadata.create_all(bind=engine)
        user = user_crud.get_user_by_email(db, email=settings.DEFAULT_ADMIN_EMAIL)
        if user:
            print("Admin user already exists: ", settings.DEFAULT_ADMIN_EMAIL)
            return
        user_in = UserCreate(
            email=settings.DEFAULT_ADMIN_EMAIL,
            password=settings.DEFAULT_ADMIN_USER_PASSWORD,
            full_name="Administrator",
            is_superuser=True,
            roles=[Roles.admin.value, Roles.system_admin.value]
        )
        user_crud.create_user(db, user_in)
        print("Admin user created: ", settings.DEFAULT_ADMIN_EMAIL)
    finally:
        db.close()

if __name__ == "__main__":
    asyncio.run(create_admin_user()) 