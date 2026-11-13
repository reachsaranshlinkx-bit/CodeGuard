from fastapi import APIRouter

router = APIRouter(tags=["health"])


@router.get("/health")
async def health_check() -> dict[str, str]:
    """Return the health status of the CodeGuard API."""
    return {
        "status": "healthy",
        "service": "CodeGuard API",
    }
