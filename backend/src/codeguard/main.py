from fastapi import FastAPI

from codeguard.api.v1.health import router as health_router
from codeguard.api.v1.reviews import router as reviews_router
from codeguard.core.config import Settings, get_settings
from codeguard.core.logging_config import configure_logging


def create_app(settings: Settings | None = None) -> FastAPI:
    """Create and configure the CodeGuard FastAPI application."""

    app_settings = settings or get_settings()

    logger = configure_logging(app_settings.log_level)

    app = FastAPI(
        title=app_settings.app_name,
        version=app_settings.app_version,
        description=app_settings.app_description,
        debug=app_settings.debug,
    )

    app.dependency_overrides[get_settings] = lambda: app_settings

    app.include_router(health_router)
    app.include_router(reviews_router)

    logger.info(
        "application_created",
        extra={
            "environment": app_settings.environment,
        },
    )

    return app


app = create_app()
