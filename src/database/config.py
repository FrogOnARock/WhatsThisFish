import os

from sqlalchemy import create_engine
from sqlalchemy.ext.asyncio import AsyncSession, create_async_engine
from sqlalchemy.orm import sessionmaker


def get_database_url(async_: bool = False) -> str:
    url = os.environ.get("DATABASE_URL")
    if not url:
        raise RuntimeError(
            "DATABASE_URL environment variable is not set. "
            "Expected format: postgresql://user:pass@host:port/dbname"
        )
    if async_:
        url = url.replace("postgresql://", "postgresql+asyncpg://", 1)
    return url


def get_engine(echo: bool = False):
    return create_engine(get_database_url(), echo=echo)


def get_async_engine(echo: bool = False):
    return create_async_engine(get_database_url(async_=True), echo=echo)


def get_session_factory(engine=None):
    if engine is None:
        engine = get_engine()
    return sessionmaker(bind=engine)


def get_async_session_factory(engine=None):
    if engine is None:
        engine = get_async_engine()
    return sessionmaker(bind=engine, class_=AsyncSession, expire_on_commit=False)
