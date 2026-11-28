from pydantic import SecretStr
from sqlalchemy.ext.asyncio import AsyncEngine, AsyncSession

from codeguard.db.database import (
    create_database_engine,
    create_session_factory,
)


async def test_create_database_engine_uses_postgresql_psycopg() -> None:
    engine = create_database_engine(
        SecretStr(
            "postgresql+psycopg://test_user:test_password@localhost:5434/codeguard"
        )
    )

    try:
        assert isinstance(engine, AsyncEngine)
        assert engine.dialect.name == "postgresql"
        assert engine.dialect.driver == "psycopg"
    finally:
        await engine.dispose()


async def test_create_session_factory_uses_async_sessions() -> None:
    engine = create_database_engine(
        SecretStr(
            "postgresql+psycopg://test_user:test_password@localhost:5434/codeguard"
        )
    )

    try:
        session_factory = create_session_factory(engine)

        assert session_factory.class_ is AsyncSession
        assert session_factory.kw["expire_on_commit"] is False
    finally:
        await engine.dispose()
