
# Use Python 3.12
FROM python:3.12-slim

# Install the pinned UV version
COPY --from=ghcr.io/astral-sh/uv:0.12.17 /uv /uvx /bin/

# Python and UV configuration
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    UV_COMPILE_BYTECODE=1 \
    UV_LINK_MODE=copy \
    PATH="/app/.venv/bin:$PATH"

# Application directory
WORKDIR /app

# Copy dependency files first
COPY pyproject.toml uv.lock ./

# Install production dependencies
RUN uv sync --frozen --no-dev --no-install-project

# Copy application source
COPY src ./src

# Install CodeGuard itself
RUN uv sync --frozen --no-dev --no-editable

# Create a non-root application user
RUN groupadd --system app && \
    useradd --system --gid app --create-home app && \
    chown -R app:app /app

USER app

# API port
EXPOSE 8000

# Start production application
CMD ["uvicorn", "codeguard.main:app", "--host", "0.0.0.0", "--port", "8000"]
