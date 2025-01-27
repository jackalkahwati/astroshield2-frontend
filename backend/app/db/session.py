from typing import Generator

def get_db() -> Generator:
    """
    Mock database session generator.
    In production, this would yield a real database session.
    """
    try:
        yield None
    finally:
        pass 