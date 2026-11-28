from uuid import UUID

from sqlalchemy import DateTime, String
from sqlalchemy.dialects.postgresql import UUID as PostgreSQLUUID
from sqlalchemy.orm import DeclarativeBase, Mapped, mapped_column

from codeguard.db.base import Base
from codeguard.db.mixins import (
    CreatedAtMixin,
    UpdatedAtMixin,
    UUIDPrimaryKeyMixin,
)


class _TestBase(DeclarativeBase):
    """Isolated test metadata for model mixins."""


class _MixinProbe(
    UUIDPrimaryKeyMixin,
    CreatedAtMixin,
    UpdatedAtMixin,
    _TestBase,
):
    """Temporary mapped model used to inspect the mixins."""

    __tablename__ = "model_mixin_probe"

    name: Mapped[str] = mapped_column(String(50), nullable=False)


def test_uuid_primary_key_configuration() -> None:
    column = _MixinProbe.__table__.c.id

    assert isinstance(column.type, PostgreSQLUUID)
    assert column.type.as_uuid is True
    assert column.primary_key is True
    assert column.default is not None
    assert callable(column.default.arg)

    first_id = column.default.arg(None)
    second_id = column.default.arg(None)

    assert isinstance(first_id, UUID)
    assert isinstance(second_id, UUID)
    assert first_id.version == 4
    assert second_id.version == 4
    assert first_id != second_id


def test_created_at_configuration() -> None:
    column = _MixinProbe.__table__.c.created_at

    assert isinstance(column.type, DateTime)
    assert column.type.timezone is True
    assert column.nullable is False
    assert column.server_default is not None


def test_updated_at_configuration() -> None:
    column = _MixinProbe.__table__.c.updated_at

    assert isinstance(column.type, DateTime)
    assert column.type.timezone is True
    assert column.nullable is False
    assert column.server_default is not None
    assert column.onupdate is not None


def test_base_has_constraint_naming_convention() -> None:
    convention = Base.metadata.naming_convention

    assert convention["pk"] == "pk_%(table_name)s"
    assert convention["ix"] == "ix_%(table_name)s_%(column_0_name)s"
    assert convention["ck"] == "ck_%(table_name)s_%(constraint_name)s"
