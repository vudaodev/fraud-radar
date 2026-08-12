# Multi-stage Dockerfile that utilises BuildKit

# -- BUILDER STAGE --
FROM python:3.13-slim AS builder
# Install UV
COPY --from=ghcr.io/astral-sh/uv:0.9.7 /uv /uvx /bin/
# Add a working directory
WORKDIR /app
# Copy configuration and lock files
COPY pyproject.toml uv.lock ./
# Create an updated python virtual environment
RUN uv sync --no-group notebooks --no-group dev --frozen --no-install-project

# -- RUNTIME STAGE --
FROM python:3.13-slim 
WORKDIR /app
# Copy virtual env from the Builder
COPY --from=builder /app/.venv /app/.venv
# Copy files that the app needs
COPY common/ common/
COPY api/ api/
COPY model/artifacts/isolation_forest.pkl model/artifacts/isolation_forest.pkl
# Set root
ENV FRAUD_RADAR_ROOT=/app
# Set Python Path so that modules can be found
ENV PYTHONPATH=/app
# Set Path so FASTAPI resolves without activating
ENV PATH="/app/.venv/bin:$PATH"
# Run the API, binding the port Heroku assigns (defaults to 8000 locally)
CMD fastapi run api/main.py --host 0.0.0.0 --port ${PORT:-8000}